"""Concept map visualization: UMAP 2D projection + interactive HTML export.

Produces a "map of science"-style 2D layout of concept embeddings (cf. Marwitz
et al. 2026, Fig. 2) and exports a self-contained interactive HTML page
rendered with **sigma.js** (WebGL, the same engine Gephi's web exporters use):

* **community coloring** — a *recursive* DBSCAN pass over the UMAP layout
  splits oversized clusters into smaller sub-communities (a plain DBSCAN run
  tends to merge the dense centre of the map into one ~100k-point blob); the
  largest K communities get distinct hues, the rest merge into a grey
  "other" bucket;
* **point size mapping** — marker radius grows with the concept's
  ``doc_freq`` (square-root scale), so frequent concepts stand out;
* **interactive** — light theme, drag nodes, wheel zoom, hover tooltips,
  a clickable community legend (toggle visibility), and a display switcher
  (community colour / density shading / uniform size);
* **density shading** — a 2D grid-kernel density estimate can drive point
  alpha (deep colour = crowded region), mirroring the paper's figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from drbrain.concept_graph.embeddings import load_concept_embeddings
from drbrain.storage.database import Database

_VENDOR_DIR = Path(__file__).parent / "vendor"

# ---------------------------------------------------------------------------
# Layout / analysis helpers
# ---------------------------------------------------------------------------


def _normalize_xy(xy: np.ndarray) -> np.ndarray:
    """Scale 2D points to unit variance per axis (stable coordinate scale)."""
    return xy / np.maximum(xy.std(axis=0, keepdims=True), 1e-9)


def density_field(xy: np.ndarray, *, bins: int = 384, sigma: float = 2.0) -> np.ndarray:
    """Per-point 2D kernel density estimate via a smoothed grid histogram.

    Args:
        xy: ``(n, 2)`` UMAP coordinates.
        bins: Grid resolution per axis.
        sigma: Gaussian smoothing width (in grid cells).

    Returns:
        Density in ``[0, 1]`` for each point (1 = densest region).
    """
    from scipy.ndimage import gaussian_filter

    if len(xy) == 0:
        return np.zeros(0, dtype=np.float64)
    xy = _normalize_xy(xy)
    g = np.clip(xy, -4.0, 4.0)
    gi = np.floor((g + 4.0) / 8.0 * bins).astype(np.int64)
    gi = np.clip(gi, 0, bins - 1)
    grid = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(grid, (gi[:, 0], gi[:, 1]), 1.0)
    smoothed = gaussian_filter(grid, sigma=sigma)
    dens = smoothed[gi[:, 0], gi[:, 1]]
    lo, hi = float(dens.min()), float(dens.max())
    if hi > lo:
        dens = (dens - lo) / (hi - lo)
    return dens


def detect_communities(
    xy: np.ndarray,
    *,
    eps: float = 0.35,
    min_samples: int = 8,
    top_k: int = 15,
    max_frac: float = 0.12,
    min_comm: int = 500,
    eps_min: float = 0.001,
    keep_all: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Cluster the normalized UMAP layout with *recursive* DBSCAN.

    A single DBSCAN pass merges the dense centre of a UMAP layout into one
    giant blob (observed: a ~107k-point community out of 109k points).  This
    variant re-clusters any community larger than ``max_frac * n`` with a
    shrinking ``eps`` (×0.7 each level) until every community is below the
    size cap, smaller than ``min_comm``, or ``eps`` drops below ``eps_min``.

    Args:
        xy: ``(n, 2)`` UMAP coordinates.
        eps: Starting DBSCAN neighbourhood radius (unit-variance scale).
        min_samples: DBSCAN core-point threshold.
        top_k: Number of largest communities to keep as distinct; smaller
            clusters and noise fold into ``-1`` ("other").
        max_frac: Communities larger than this fraction of the data are
            split recursively.
        min_comm: Lower bound for considering a sub-cluster worth splitting.
        eps_min: Stop shrinking ``eps`` below this value.  Must be well below
            the ``eps`` needed to resolve the map's dense centre (0.02 is
            *not* enough — the giant blob still survives; 0.001 resolves it
            into ~1000 leaf communities).
        keep_all: When True, return *every* leaf community (ids assigned in
            size order, ``-1`` kept only for true DBSCAN noise) instead of
            folding non-``top_k`` leaves into ``-1``.

    Returns:
        ``(labels, communities)`` where ``labels[i]`` is the community id
        (``-1`` for merged/other) and ``communities`` lists dicts with
        ``id`` and ``size``, ordered by size descending.
    """
    from sklearn.cluster import DBSCAN

    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=np.int64), []
    if n < min_samples:
        return np.full(n, -1, dtype=np.int64), []
    xyn = _normalize_xy(xy)

    leaves: list[np.ndarray] = []

    def split(idx: np.ndarray, eps_level: float) -> None:
        sub = xyn[idx]
        lab = DBSCAN(eps=eps_level, min_samples=min_samples).fit_predict(sub)
        for c in sorted(set(int(x) for x in lab)):
            if c < 0:
                continue  # noise -> other
            m = idx[lab == c]
            if len(m) > max_frac * n and len(m) >= min_comm and eps_level > eps_min:
                split(m, eps_level * 0.7)
            else:
                leaves.append(m)

    split(np.arange(n), eps)
    leaves.sort(key=len, reverse=True)

    labels = np.full(n, -1, dtype=np.int64)
    keep = leaves if keep_all else leaves[:top_k]
    for i, m in enumerate(keep):
        labels[m] = i
    communities = [{"id": i, "size": int(len(m))} for i, m in enumerate(keep)]
    return labels, communities


def _community_names(
    labels: list[str],
    freq: np.ndarray,
    comm_labels: np.ndarray,
    communities: list[dict],
    top_concepts: int = 3,
) -> list[str]:
    """Name each community after its top-frequency concept labels."""
    names: list[str] = []
    for c in communities:
        members = np.where(comm_labels == c["id"])[0]
        order = np.argsort(freq[members])[::-1][:top_concepts]
        top = [labels[members[i]] for i in order]
        names.append(" · ".join(top) if top else f"Community {c['id']}")
    return names


def _vendor_js(name: str) -> str:
    """Read a bundled vendor JS file (sigma.min.js / graphology.min.js)."""
    path = _VENDOR_DIR / name
    js = path.read_text(encoding="utf-8")
    return js.replace("</script", "<\\/script")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def umap_project(
    db: Database, *, n_components: int = 2, random_state: int = 42
) -> dict[str, list[float]]:
    """Project concept embeddings to ``n_components`` dimensions via UMAP.

    Args:
        db: Database handle.
        n_components: Target dimensionality (default 2 for a map).
        random_state: Seed for reproducibility.

    Returns:
        ``{label: [coords...]}``. Empty when fewer than 2 embeddings exist.
    """
    embeddings = load_concept_embeddings(db)
    labels = list(embeddings.keys())
    if len(labels) < 2:
        logger.warning("[cg.map] need >= 2 concept embeddings for UMAP, got {}", len(labels))
        return {}

    import umap

    matrix = np.asarray([embeddings[lab] for lab in labels], dtype=np.float32)
    n_samples, n_features = matrix.shape
    reducer = umap.UMAP(
        n_components=min(n_components, n_features - 1, n_samples - 1) or 1,
        n_neighbors=min(15, n_samples - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    reduced = reducer.fit_transform(matrix)
    return {label: reduced[i].tolist() for i, label in enumerate(labels)}


def export_html(
    db: Database,
    output: str | Path,
    *,
    coords: dict[str, list[float]] | None = None,
    top_communities: int = 150,
    density_bins: int = 384,
    random_state: int = 42,
) -> Path:
    """Export an interactive 2D concept map to a self-contained HTML file.

    Args:
        db: Database handle.
        output: Destination ``.html`` path.
        coords: Optional precomputed ``{label: [x, y]}``; computed via UMAP if
            omitted.
        top_communities: Number of largest communities shown in the legend
            with a distinct name/colour; every other leaf community is still
            rendered with a low-saturation colour — only true DBSCAN noise
            stays grey.
        density_bins: Grid resolution for the density shading.
        random_state: UMAP seed.

    Returns:
        The written path.
    """
    if coords is None:
        coords = umap_project(db, n_components=2, random_state=random_state)
    labels = [lab for lab, xy in coords.items() if len(xy) >= 2]
    if not labels:
        logger.warning("[cg.map] no 2D coordinates to plot")
        out_path = Path(output)
        out_path.write_text(_render_html([], []), encoding="utf-8")
        return out_path

    xy = np.asarray([coords[lab][:2] for lab in labels], dtype=np.float64)
    freq_map = dict(db.conn.execute("SELECT label, doc_freq FROM concept_nodes").fetchall())
    freq = np.asarray([freq_map.get(lab, 0) for lab in labels], dtype=np.float64)

    dens = density_field(xy, bins=density_bins)
    comm_labels, all_comms = detect_communities(xy, top_k=top_communities, keep_all=True)
    named = all_comms[:top_communities]
    names = _community_names(labels, freq, comm_labels, named)
    for c, name in zip(named, names):
        c["name"] = name

    points = [
        {
            "label": lab,
            "x": float(x),
            "y": float(y),
            "freq": int(f),
            "density": float(d),
            "community": int(c),
        }
        for lab, (x, y), f, d, c in zip(labels, xy, freq, dens, comm_labels)
    ]
    out_path = Path(output)
    out_path.write_text(_render_html(points, named), encoding="utf-8")
    logger.info(
        "[cg.map] wrote {} ({} points, {} named communities, {} leaves total)",
        out_path,
        len(points),
        len(named),
        len(all_comms),
    )
    return out_path


# ---------------------------------------------------------------------------
# HTML rendering (sigma.js / WebGL)
# ---------------------------------------------------------------------------


def _community_colors(n_communities: int) -> list[str]:
    """Distinct hues for communities via golden-angle rotation."""
    colors: list[str] = []
    for cid in range(n_communities):
        hue = (cid * 137.508) % 360.0
        colors.append(f"hsl({hue:.0f}, 62%, 45%)")
    return colors


def _render_html(points: list[dict], communities: list[dict]) -> str:
    """Render the self-contained HTML (sigma.js + graphology.js embedded)."""
    comm_colors = _community_colors(len(communities))
    for c, color in zip(communities, comm_colors):
        c["color"] = color
    if len(points) == 0:
        # Nothing to plot: still ship a valid page (sigma tolerates empty graph).
        pass

    nodes = [
        {
            "label": p["label"],
            "x": p["x"],
            "y": p["y"],
            "freq": p["freq"],
            "density": round(p["density"], 4),
            "community": p["community"],
        }
        for p in points
    ]
    data_json = json.dumps(
        {"nodes": nodes, "communities": communities, "total": len(nodes)},
        ensure_ascii=False,
    ).replace("<", "\\u003c")

    graphology_js = _vendor_js("graphology.min.js")
    sigma_js = _vendor_js("sigma.min.js")
    title = f"DrBrain Concept Map — {len(nodes):,} concepts (Qwen3-Embedding · UMAP)"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DrBrain Concept Map</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #f6f8fa; color: #24292f; font: 13px/1.45 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; overflow: hidden; }}
  #wrap {{ display: flex; height: 100vh; }}
  #sidebar {{ width: 300px; min-width: 300px; background: #fff; border-right: 1px solid #e1e4e8; display: flex; flex-direction: column; overflow: hidden; }}
  #map {{ flex: 1; background: #f6f8fa; position: relative; }}
  #map canvas {{ cursor: grab; }}
  #map canvas:active {{ cursor: grabbing; }}
  #logo {{ padding: 14px 16px; border-bottom: 1px solid #e1e4e8; }}
  #logo h1 {{ font-size: 15px; font-weight: 600; }}
  #logo p {{ color: #57606a; font-size: 12px; margin-top: 2px; }}
  #controls {{ padding: 10px 16px; border-bottom: 1px solid #e1e4e8; display: flex; gap: 6px; flex-wrap: wrap; }}
  #controls button {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 5px 10px; font-size: 12px; cursor: pointer; color: #24292f; }}
  #controls button:hover {{ background: #eaeef2; }}
  #controls button.active {{ background: #0969da; border-color: #0969da; color: #fff; }}
  #legend {{ flex: 1; overflow-y: auto; padding: 8px 0; }}
  #legend .item {{ display: flex; align-items: center; gap: 8px; padding: 5px 16px; cursor: pointer; }}
  #legend .item:hover {{ background: #f6f8fa; }}
  #legend .swatch {{ width: 12px; height: 12px; border-radius: 50%; flex: none; }}
  #legend .name {{ flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  #legend .count {{ color: #57606a; font-size: 11px; flex: none; }}
  #legend .item.off {{ opacity: .35; }}
  #legend h2 {{ font-size: 11px; text-transform: uppercase; color: #57606a; padding: 10px 16px 4px; }}
  #tooltip {{ position: absolute; display: none; pointer-events: none; background: rgba(255,255,255,.96); border: 1px solid #d0d7de; border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12); max-width: 320px; z-index: 10; }}
  #tooltip b {{ font-size: 12.5px; }}
  #tooltip .dim {{ color: #57606a; }}
  #statusbar {{ position: absolute; left: 10px; bottom: 8px; color: #57606a; font-size: 11px; background: rgba(255,255,255,.8); padding: 2px 8px; border-radius: 10px; }}
  .tip {{ padding: 0 16px 10px; color: #8c959f; font-size: 11px; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="sidebar">
    <div id="logo">
      <h1>DrBrain Concept Map</h1>
      <p>{title}</p>
    </div>
    <div id="controls">
      <button id="btnComm" class="active">社区色</button>
      <button id="btnDensity">密度色</button>
      <button id="btnUniform">统一大小</button>
      <button id="btnReset">重置视图</button>
    </div>
    <div id="legend"><h2>社区 · 点击开关</h2></div>
    <div class="tip">滚轮缩放 · 拖拽节点/空白平移 · 悬停查看概念 · 点大小 = 概念词频</div>
  </div>
  <div id="map">
    <div id="tooltip"></div>
    <div id="statusbar"></div>
  </div>
</div>
<script>{graphology_js}</script>
<script>{sigma_js}</script>
<script>
const DATA = {data_json};
const container = document.getElementById('map');
const tooltip = document.getElementById('tooltip');
const statusbar = document.getElementById('statusbar');

const commById = {{}};
for (const c of DATA.communities) commById[c.id] = c;
const namedCount = DATA.communities.length;
const MINOR_COLOR = '#7d94ad';
const NOISE_COLOR = '#9aa4af';
const nodeKey = c => (c < 0 ? 'noise' : (c < namedCount ? String(c) : 'minor'));
const nodeColor = c => (c < 0 ? NOISE_COLOR : (c < namedCount ? commById[c].color : MINOR_COLOR));

const maxFreq = Math.max(1, ...DATA.nodes.map(n => n.freq));
const sizeOf = n => Math.max(1.5, 1.5 + 6.5 * Math.sqrt(n.freq / maxFreq));

const graph = new graphology.Graph();
DATA.nodes.forEach((n, i) => {{
  graph.addNode(String(i), {{
    label: n.label, x: n.x, y: n.y, freq: n.freq,
    density: n.density, community: n.community,
    size: sizeOf(n),
    color: nodeColor(n.community),
    hidden: false,
  }});
}});

const renderer = new Sigma(graph, container, {{
  renderer: {{ container, type: 'webgl' }},
  settings: {{
    minNodeSize: 1.5, maxNodeSize: 12,
    defaultNodeColor: '#9aa4af',
    labelColor: {{ color: '#24292f' }},
    labelRenderedSizeThreshold: 9,
    labelDensity: 2,
    labelFont: '13px -apple-system, sans-serif',
    enableEdgeHovering: false,
  }},
}});

// --- display modes ---------------------------------------------------------
let mode = 'community';
function applyColors() {{
  if (mode === 'density') {{
    graph.forEachNode((_, a) => {{
      const alpha = 0.25 + 0.75 * a.density;
      graph.setNodeAttribute(_, 'color', `rgba(9, 105, 218, ${{alpha}})`.replace(' ', ''));
    }});
  }} else {{
    graph.forEachNode((_, a) => {{
      graph.setNodeAttribute(_, 'color', nodeColor(a.community));
    }});
  }}
  renderer.refresh();
}}
function applySizes(uniform) {{
  graph.forEachNode((_, a) => {{
    graph.setNodeAttribute(_, 'size', uniform ? 3 : sizeOf(a));
  }});
  renderer.refresh();
}}
document.getElementById('btnComm').onclick = () => {{ mode = 'community'; applyColors();
  document.getElementById('btnComm').classList.add('active');
  document.getElementById('btnDensity').classList.remove('active');
}};
document.getElementById('btnDensity').onclick = () => {{ mode = 'density'; applyColors();
  document.getElementById('btnDensity').classList.add('active');
  document.getElementById('btnComm').classList.remove('active');
}};
document.getElementById('btnUniform').onclick = () => {{ applySizes(true); }};
document.getElementById('btnComm').onclick();  // no-op keeps state coherent
document.getElementById('btnReset').onclick = () => renderer.getCamera().animatedReset();

// re-enable real uniform toggle semantics
let uniform = false;
document.getElementById('btnUniform').onclick = () => {{
  uniform = !uniform;
  applySizes(uniform);
}};

// --- community legend ------------------------------------------------------
const legend = document.getElementById('legend');
const hiddenComms = new Set();
function toggleComm(key, item) {{
  if (hiddenComms.has(key)) {{ hiddenComms.delete(key); item.classList.remove('off'); }}
  else {{ hiddenComms.add(key); item.classList.add('off'); }}
  graph.forEachNode((_, a) => {{
    graph.setNodeAttribute(_, 'hidden', hiddenComms.has(nodeKey(a.community)));
  }});
  renderer.refresh();
}}
function legendItem(swatchColor, text, countText) {{
  const item = document.createElement('div');
  item.className = 'item';
  const sw = document.createElement('span');
  sw.className = 'swatch'; sw.style.background = swatchColor;
  const nm = document.createElement('span');
  nm.className = 'name'; nm.textContent = text; nm.title = text;
  const ct = document.createElement('span');
  ct.className = 'count'; ct.textContent = countText;
  item.appendChild(sw); item.appendChild(nm); item.appendChild(ct);
  legend.appendChild(item);
  return item;
}}
for (const c of DATA.communities) {{
  const item = legendItem(c.color, c.name, c.size.toLocaleString());
  item.onclick = () => toggleComm(String(c.id), item);
}}
const minorClusters = new Set(DATA.nodes.filter(n => n.community >= namedCount).map(n => n.community)).size;
const minorCount = DATA.nodes.filter(n => n.community >= namedCount).length;
const minorItem = legendItem(MINOR_COLOR, `其他小簇 (${{minorClusters}} 个)`, minorCount.toLocaleString());
minorItem.onclick = () => toggleComm('minor', minorItem);
const noiseCount = DATA.nodes.filter(n => n.community < 0).length;
const noiseItem = legendItem(NOISE_COLOR, '噪声', noiseCount.toLocaleString());
noiseItem.onclick = () => toggleComm('noise', noiseItem);

// --- hover tooltip ---------------------------------------------------------
let hovered = null;
renderer.on('enterNode', ({{ node }}) => {{
  const a = graph.getNodeAttributes(node);
  hovered = a;
  const comm = a.community < 0 ? '噪声' : (a.community < namedCount ? commById[a.community].name : `小簇 #${{a.community}}`);
  tooltip.innerHTML =
    `<b>${{a.label}}</b><br>` +
    `<span class="dim">词频 ${{a.freq.toLocaleString()}} · 密度 ${{(a.density*100).toFixed(0)}}% · 社区: ${{comm}}</span>`;
  tooltip.style.display = 'block';
}});
renderer.on('moveNode', (e) => {{
  const p = renderer.graphToViewport({{ x: e.node }});
  if (p) {{
    tooltip.style.left = (p.x + 14) + 'px';
    tooltip.style.top = (p.y + 14) + 'px';
  }}
}});
renderer.on('leaveNode', () => {{ tooltip.style.display = 'none'; hovered = null; }});
renderer.on('clickStage', () => {{ tooltip.style.display = 'none'; }});
renderer.on('downStage', () => {{ tooltip.style.display = 'none'; }});

// --- drag nodes / pan canvas ----------------------------------------------
let dragMode = null;          // 'node' | 'pan'
let dragNodeId = null;
let panLast = null;
const captor = renderer.getMouseCaptor();
captor.on('mousedown', (e) => {{
  // hit-test in pixel space (graphToViewport) so scaling is handled
  let best = null, bestD = 1e9;
  graph.forEachNode((id, a) => {{
    const p = renderer.graphToViewport({{ x: a.x, y: a.y }});
    if (!p) return;
    const d = Math.hypot(p.x - e.x, p.y - e.y);
    const r = Math.max(a.size, 2) + 2;   // px hit radius
    if (d < r && d < bestD) {{ bestD = d; best = id; }}
  }});
  if (best) {{ dragMode = 'node'; dragNodeId = best; }}
  else {{ dragMode = 'pan'; panLast = {{ x: e.x, y: e.y }}; }}
}});
captor.on('mousemove', (e) => {{
  if (dragMode === 'node' && dragNodeId != null) {{
    const pos = renderer.viewportToGraph({{ x: e.x, y: e.y }});
    graph.setNodeAttribute(dragNodeId, 'x', pos.x);
    graph.setNodeAttribute(dragNodeId, 'y', pos.y);
  }} else if (dragMode === 'pan' && panLast) {{
    const cam = renderer.getCamera();
    const st = cam.getState();
    cam.setX(st.x - (e.x - panLast.x) * st.ratio);
    cam.setY(st.y - (e.y - panLast.y) * st.ratio);
    panLast = {{ x: e.x, y: e.y }};
  }}
}});
const endDrag = () => {{ dragMode = null; dragNodeId = null; panLast = null; }};
captor.on('mouseup', endDrag);
captor.on('mouseleave', endDrag);

// --- status + resize -------------------------------------------------------
statusbar.textContent = `${{DATA.total.toLocaleString()}} concepts · ${{DATA.communities.length}} communities · drag / wheel-zoom`;
window.addEventListener('resize', () => renderer.refresh());
</script>
</body>
</html>
"""
