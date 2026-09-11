/* DrBrain WebUI behavior: stream lifecycle, form guards, small interactions.
   No inline handlers — everything binds via data attributes so the CSP can
   stay strict. */

(function () {
  "use strict";

  // ── project switcher: keep the current page, swap scope ──
  document.addEventListener("change", function (event) {
    var select = event.target.closest("[data-project-switcher]");
    if (!select) return;
    var url = new URL(window.location.href);
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
    var form = event.target;
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
      function () { restore(form); },
      { once: true }
    );
  });
  document.body.addEventListener("htmx:afterRequest", function (event) {
    var elt = event.detail && event.detail.elt;
    var form = elt && elt.closest ? elt.closest("form") : null;
    if (form) restore(form);
  });

  // ── chat: clear the composer after a successful swap, keep failures ──
  document.body.addEventListener("htmx:afterSwap", function (event) {
    var messages = document.getElementById("messages");
    if (messages && messages.contains(event.detail.target)) {
      var input = document.getElementById("question");
      if (input) input.value = "";
      var list = document.getElementById("messages");
      if (list) list.scrollTop = list.scrollHeight;
    }
  });
  document.body.addEventListener("htmx:responseError", function (event) {
    var target = event.detail.target;
    if (target && target.id === "messages") {
      var alerts = document.getElementById("chat-alerts");
      if (alerts) {
        alerts.innerHTML =
          '<div class="alert alert-bad" role="alert">请求失败（' +
          event.detail.xhr.status +
          "）。请重试；若登录已过期请重新登录。</div>";
      }
      return;
    }
    var xhr = event.detail.xhr;
    if (!xhr || xhr.status === 401) return; // HX-Redirect handles auth
    var message = "请求失败（" + xhr.status + "）。请重试或检查筛选条件。";
    try {
      var parsed = JSON.parse(xhr.responseText);
      if (parsed && parsed.error) message = parsed.error;
    } catch (err) {
      /* non-JSON error body */
    }
    showAlert(message);
  });

  function showAlert(message) {
    var main = document.getElementById("main");
    if (!main) return;
    var banner = document.createElement("div");
    banner.className = "alert alert-bad";
    banner.setAttribute("role", "alert");
    banner.textContent = message;
    main.insertBefore(banner, main.firstChild);
    window.setTimeout(function () {
      banner.remove();
    }, 8000);
  }

  // ── run detail: SSE lifecycle ──
  function initRunStream(root) {
    if (!root || root.dataset.streamStarted === "1") return;
    root.dataset.streamStarted = "1";
    var runId = root.dataset.runId;
    var projectId = root.dataset.projectId;
    var after = root.dataset.after || "0";
    var list = document.getElementById("event-list");
    var state = document.getElementById("stream-state");
    var followButton = document.getElementById("follow-latest");
    var seen = new Set();
    var pending = 0;
    var closed = root.dataset.streamState === "closed";

    if (closed) return;

    function nearBottom() {
      return window.innerHeight + window.scrollY >= document.body.offsetHeight - 120;
    }
    var follow = true;

    function setState(text, tone) {
      if (!state) return;
      state.textContent = text;
      state.className = "badge badge-" + tone;
    }

    function appendEvent(event) {
      if (!list || seen.has(event.seq)) return;
      seen.add(event.seq);
      var item = document.createElement("li");
      item.className = "event";
      item.dataset.seq = event.seq;
      var payload = "";
      try {
        payload = JSON.stringify(event.payload, null, 2);
      } catch (err) {
        payload = "";
      }
      item.innerHTML =
        '<div class="event-head"><span class="event-seq">#' +
        event.seq +
        '</span><span class="event-actor"></span><span class="event-type"></span><time class="muted small">' +
        new Date((event.created_at || 0) * 1000).toLocaleString() +
        "</time></div>" +
        (payload
          ? '<details class="event-payload"><summary>payload</summary><pre></pre></details>'
          : "");
      item.querySelector(".event-actor").textContent = event.actor || "";
      item.querySelector(".event-type").textContent = event.type || "";
      if (payload) item.querySelector("pre").textContent = payload;
      list.appendChild(item);
      while (list.children.length > 500) list.removeChild(list.firstChild);
      if (follow) {
        list.scrollTop = list.scrollHeight;
      } else {
        pending += 1;
        if (followButton) {
          followButton.classList.remove("is-hidden");
          followButton.textContent = "新增 " + pending + " 条 / 回到最新";
        }
      }
    }

    var source = new EventSource(
      "/api/runs/" + encodeURIComponent(runId) + "/stream?project_id=" +
        encodeURIComponent(projectId) + "&after=" + encodeURIComponent(after)
    );
    var errors = 0;
    source.addEventListener("message", function (event) {
      errors = 0;
      try {
        appendEvent(JSON.parse(event.data));
      } catch (err) {
        /* ignore malformed frame; the next event re-syncs */
      }
    });
    source.addEventListener("status", function (event) {
      try {
        var data = JSON.parse(event.data);
        var map = { running: "run", created: "muted", paused: "warn", succeeded: "ok", failed: "bad", cancelled: "muted", interrupted: "warn" };
        setState(
          { running: "运行中", created: "待运行", paused: "已暂停", succeeded: "成功", failed: "失败", cancelled: "已取消", interrupted: "已中断" }[data.display_status] || data.display_status,
          map[data.display_status] || "muted"
        );
        var counts = { "kpi-events": data.events, "kpi-claims": data.claims, "kpi-verified": data.verified, "kpi-experiments": data.experiments };
        Object.keys(counts).forEach(function (id) {
          var el = document.getElementById(id);
          if (el && counts[id] !== undefined && counts[id] !== null) el.textContent = counts[id];
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
        "/login?next=" + encodeURIComponent(window.location.pathname + window.location.search)
      );
    });
    source.onerror = function () {
      if (source.readyState === EventSource.CLOSED) {
        setState("连接已关闭", "muted");
        return;
      }
      errors += 1;
      setState("连接中断，重连中…", "warn");
      // A revoked login rejects every reconnect with 401; three failures in a
      // row send the tab back to the login page instead of retrying forever.
      if (errors >= 3) {
        source.close();
        window.location.assign(
          "/login?next=" + encodeURIComponent(window.location.pathname + window.location.search)
        );
      }
    };

    window.addEventListener("scroll", function () {
      follow = nearBottom();
      if (follow) {
        pending = 0;
        if (followButton) followButton.classList.add("is-hidden");
      }
    });
    if (followButton) {
      followButton.addEventListener("click", function () {
        follow = true;
        pending = 0;
        followButton.classList.add("is-hidden");
        if (list) list.scrollTop = list.scrollHeight;
        window.scrollTo({ top: document.body.scrollHeight });
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
