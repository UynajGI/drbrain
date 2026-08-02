"""Concept map visualization: UMAP 2D projection + interactive HTML export.

Produces a "map of science"-style 2D layout of concept embeddings (cf. Marwitz
et al. 2026, Fig. 2) and exports a self-contained interactive scatter plot.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from loguru import logger

from drbrain.concept_graph.embeddings import load_concept_embeddings
from drbrain.storage.database import Database


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
    db: Database, output: str | Path, *, coords: dict[str, list[float]] | None = None
) -> Path:
    """Export an interactive 2D concept map to a self-contained HTML file.

    Args:
        db: Database handle.
        output: Destination ``.html`` path.
        coords: Optional precomputed ``{label: [x, y]}``; computed via UMAP if omitted.

    Returns:
        The written path.
    """
    if coords is None:
        coords = umap_project(db, n_components=2)

    # Attach document frequency for hover context.
    freq = dict(db.conn.execute("SELECT label, doc_freq FROM concept_nodes").fetchall())
    points = [
        {"label": label, "x": xy[0], "y": xy[1], "freq": int(freq.get(label, 0))}
        for label, xy in coords.items()
        if len(xy) >= 2
    ]
    out_path = Path(output)
    out_path.write_text(_render_html(points), encoding="utf-8")
    logger.info("[cg.map] wrote {} ({} points)", out_path, len(points))
    return out_path


def _render_html(points: list[dict]) -> str:
    # Escape "<" so a malicious label containing "</script>" cannot terminate the
    # inline script and inject JS (labels come from external corpus metadata).
    data_json = json.dumps(points, ensure_ascii=False).replace("<", "\\u003c")
    title = html.escape("DrBrain Concept Map")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; font-family: system-ui, sans-serif; background: #0d1117; color: #c9d1d9; }}
  #wrap {{ position: relative; width: 100vw; height: 100vh; }}
  svg {{ width: 100%; height: 100%; display: block; }}
  circle {{ fill: #58a6ff; fill-opacity: 0.6; stroke: #1f6feb; }}
  #tip {{ position: absolute; pointer-events: none; background: #161b22; border: 1px solid #30363d;
          padding: 4px 8px; border-radius: 6px; font-size: 12px; display: none; }}
  h1 {{ position: absolute; top: 12px; left: 16px; font-size: 16px; margin: 0; opacity: 0.8; }}
</style>
</head>
<body>
<div id="wrap">
  <h1>DrBrain Concept Map</h1>
  <svg id="map"></svg>
  <div id="tip"></div>
</div>
<script>
const POINTS = {data_json};
const svg = document.getElementById('map');
const tip = document.getElementById('tip');
function render() {{
  const w = svg.clientWidth, h = svg.clientHeight;
  svg.innerHTML = '';
  if (!POINTS.length) return;
  const xs = POINTS.map(p => p.x), ys = POINTS.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 40;
  const sx = x => pad + (x - minX) / (maxX - minX || 1) * (w - 2 * pad);
  const sy = y => h - pad - (y - minY) / (maxY - minY || 1) * (h - 2 * pad);
  const maxFreq = Math.max(...POINTS.map(p => p.freq), 1);
  for (const p of POINTS) {{
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', sx(p.x)); c.setAttribute('cy', sy(p.y));
    c.setAttribute('r', 3 + 6 * (p.freq / maxFreq));
    c.addEventListener('mousemove', e => {{
      tip.style.display = 'block';
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top = (e.clientY + 12) + 'px';
      tip.textContent = p.label + ' (freq ' + p.freq + ')';
    }});
    c.addEventListener('mouseleave', () => tip.style.display = 'none');
    svg.appendChild(c);
  }}
}}
window.addEventListener('resize', render);
render();
</script>
</body>
</html>
"""
