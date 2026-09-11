/* DrBrain WebUI behavior: stream lifecycle, form guards, small interactions.
   No inline handlers — everything binds via data attributes so the CSP can
   stay strict. */

(function () {
  "use strict";

  const MAX_EVENT_NODES = 500;

  // ── project switcher: keep the current page, swap scope ──
  document.addEventListener("change", function (event) {
    const select = event.target.closest("[data-project-switcher]");
    if (!select) return;
    const url = new URL(window.location.href);
    url.searchParams.set("project_id", select.value);
    // Scope-specific detail routes resolve their own project; use the list
    // route for those so the switch never points at another project's entity.
    window.location.assign(url.pathname + "?" + url.searchParams.toString());
  });

  // ── submit guards: disable once, restore on success/failure/back ──
  function disableOnce(form) {
    form.querySelectorAll("[data-submit-once]").forEach(function (button) {
      button.disabled = true;
      button.dataset.originalText = button.textContent;
      button.textContent = "处理中…";
    });
  }
  function restore(form) {
    form.querySelectorAll("[data-submit-once]").forEach(function (button) {
      button.disabled = false;
      if (button.dataset.originalText) button.textContent = button.dataset.originalText;
    });
  }
  document.addEventListener("submit", function (event) {
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    if (form.dataset.confirm && !window.confirm(form.dataset.confirm)) {
      event.preventDefault();
      return;
    }
    if (form.hasAttribute("hx-post")) return; // htmx owns its lifecycle
    disableOnce(form);
    // Non-AJAX navigation: restore when the page comes back from bfcache.
    window.addEventListener(
      "pageshow",
      function () {
        restore(form);
      },
      { once: true }
    );
  });
  document.body.addEventListener("htmx:afterRequest", function (event) {
    const elt = event.detail && event.detail.elt;
    const form = elt && elt.closest ? elt.closest("form") : null;
    if (form) restore(form);
  });

  // ── chat: clear the composer after a successful swap, keep failures ──
  document.body.addEventListener("htmx:afterSwap", function (event) {
    const messages = document.getElementById("messages");
    const target = event.detail.target;
    if (messages && (target === messages || messages.contains(target))) {
      const input = document.getElementById("question");
      if (input) input.value = "";
      messages.scrollTop = messages.scrollHeight;
    }
  });

  function showAlert(message) {
    const main = document.getElementById("main");
    if (!main) return;
    const banner = document.createElement("div");
    banner.className = "alert alert-bad";
    banner.setAttribute("role", "alert");
    banner.textContent = message;
    main.insertBefore(banner, main.firstChild);
    window.setTimeout(function () {
      banner.remove();
    }, 8000);
  }

  document.body.addEventListener("htmx:responseError", function (event) {
    const target = event.detail.target;
    if (target && target.id === "messages") {
      const alerts = document.getElementById("chat-alerts");
      if (alerts) {
        alerts.innerHTML = "";
        const banner = document.createElement("div");
        banner.className = "alert alert-bad";
        banner.setAttribute("role", "alert");
        banner.textContent =
          "请求失败（" +
          event.detail.xhr.status +
          "）。请重试；若登录已过期请重新登录。";
        alerts.appendChild(banner);
      }
      return;
    }
    const xhr = event.detail.xhr;
    if (!xhr || xhr.status === 401) return; // HX-Redirect handles auth
    let message = "请求失败（" + xhr.status + "）。请重试或检查筛选条件。";
    try {
      const parsed = JSON.parse(xhr.responseText);
      if (parsed && parsed.error) message = parsed.error;
    } catch (err) {
      /* non-JSON error body */
    }
    showAlert(message);
  });

  // ── run detail: SSE lifecycle ──
  function buildEventRow(event) {
    const seq = Number(event.seq);
    const item = document.createElement("li");
    item.className = "event";
    item.dataset.seq = String(Number.isFinite(seq) ? seq : 0);

    const head = document.createElement("div");
    head.className = "event-head";

    const seqNode = document.createElement("span");
    seqNode.className = "event-seq";
    seqNode.textContent = Number.isFinite(seq) ? "#" + seq : "#?";

    const actorNode = document.createElement("span");
    actorNode.className = "event-actor";
    actorNode.textContent = event.actor || "";

    const typeNode = document.createElement("span");
    typeNode.className = "event-type";
    typeNode.textContent = event.type || "";

    const timeNode = document.createElement("time");
    timeNode.className = "muted small";
    const created = Number(event.created_at);
    timeNode.textContent = Number.isFinite(created)
      ? new Date(created * 1000).toLocaleString()
      : "";

    head.append(seqNode, actorNode, typeNode, timeNode);
    item.appendChild(head);

    if (event.payload && Object.keys(event.payload).length) {
      let payloadText = "";
      try {
        payloadText = JSON.stringify(event.payload, null, 2);
      } catch (err) {
        payloadText = "";
      }
      if (payloadText) {
        const details = document.createElement("details");
        details.className = "event-payload";
        const summary = document.createElement("summary");
        summary.textContent = "payload";
        const pre = document.createElement("pre");
        pre.textContent = payloadText;
        details.append(summary, pre);
        item.appendChild(details);
      }
    }
    return item;
  }

  function initRunStream(root) {
    if (!root || root.dataset.streamStarted === "1") return;
    if (root.dataset.streamState === "closed") return; // terminal run: history only
    root.dataset.streamStarted = "1";

    const runId = root.dataset.runId;
    const projectId = root.dataset.projectId;
    const after = root.dataset.after || "0";
    const streamUrl = root.dataset.streamUrl;
    const loginUrl = document.body.dataset.loginUrl || "/login";
    const list = document.getElementById("event-list");
    const state = document.getElementById("stream-state");
    const followButton = document.getElementById("follow-latest");
    const seen = new Set();
    let pending = 0;
    let follow = true;

    if (list) {
      list.querySelectorAll("[data-seq]").forEach(function (node) {
        const seq = Number(node.dataset.seq);
        if (Number.isFinite(seq)) seen.add(seq);
      });
    }

    function nearBottom() {
      if (!list) return true;
      return list.scrollHeight - list.scrollTop - list.clientHeight < 80;
    }

    function setState(text, tone) {
      if (!state) return;
      state.textContent = text;
      state.className = "badge badge-" + tone;
    }

    function appendEvent(event) {
      if (!list) return;
      const seq = Number(event.seq);
      if (Number.isFinite(seq) && seen.has(seq)) return;
      if (Number.isFinite(seq)) seen.add(seq);
      list.appendChild(buildEventRow(event));
      while (list.children.length > MAX_EVENT_NODES) {
        const removed = list.removeChild(list.firstChild);
        const removedSeq = Number(removed.dataset ? removed.dataset.seq : NaN);
        if (Number.isFinite(removedSeq)) seen.delete(removedSeq);
      }
      if (follow) {
        list.scrollTop = list.scrollHeight;
      } else {
        pending += 1;
        if (followButton) {
          followButton.textContent = "新增 " + pending + " 条 / 回到最新";
          followButton.classList.remove("is-hidden");
        }
      }
    }

    if (!streamUrl) return;
    const query =
      "?project_id=" +
      encodeURIComponent(projectId || "") +
      "&after=" +
      encodeURIComponent(after);
    const source = new EventSource(streamUrl + query);
    source.addEventListener("message", function (event) {
      try {
        appendEvent(JSON.parse(event.data));
      } catch (err) {
        /* ignore malformed frame; the next event re-syncs */
      }
    });
    source.addEventListener("status", function (event) {
      try {
        const data = JSON.parse(event.data);
        setState(data.label || data.display_status || "", data.tone || "muted");
        const counts = {
          "kpi-events": data.events,
          "kpi-claims": data.claims,
          "kpi-verified": data.verified,
          "kpi-experiments": data.experiments,
        };
        Object.keys(counts).forEach(function (id) {
          const el = document.getElementById(id);
          if (el && counts[id] !== undefined && counts[id] !== null) {
            el.textContent = counts[id];
          }
        });
      } catch (err) {
        /* ignore */
      }
    });
    source.addEventListener("end", function () {
      setState("已结束", "muted");
      source.close();
    });
    source.addEventListener("auth-expired", function () {
      source.close();
      window.location.assign(
        loginUrl +
          "?next=" +
          encodeURIComponent(window.location.pathname + window.location.search)
      );
    });
    // Transient network failures only update the badge; EventSource retries on
    // its own.  A revoked login arrives as an explicit auth-expired event.
    source.onerror = function () {
      if (source.readyState === EventSource.CLOSED) {
        setState("连接已关闭", "muted");
      } else {
        setState("连接中断，重连中…", "warn");
      }
    };

    if (list) {
      list.addEventListener("scroll", function () {
        follow = nearBottom();
        if (follow) {
          pending = 0;
          if (followButton) followButton.classList.add("is-hidden");
        } else if (followButton && pending > 0) {
          followButton.classList.remove("is-hidden");
        }
      });
    }
    if (followButton) {
      followButton.addEventListener("click", function () {
        follow = true;
        pending = 0;
        followButton.classList.add("is-hidden");
        if (list) list.scrollTop = list.scrollHeight;
      });
    }
    window.addEventListener("beforeunload", function () {
      source.close();
    });
  }

  function boot() {
    document.querySelectorAll("[data-run-stream]").forEach(initRunStream);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
  document.body.addEventListener("htmx:afterSwap", function () {
    document.querySelectorAll("[data-run-stream]").forEach(initRunStream);
  });
})();
