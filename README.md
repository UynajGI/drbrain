# DrBrain — Academic Knowledge Graph System

Vector-free, symbol-driven research discovery engine.

## Quick Start

```bash
uv sync
uv run drbrain ingest paper.pdf
uv run drbrain query "show gaps in long-range dependency"
uv run drbrain serve  # launch Streamlit UI
```

## Architecture

- **Parser**: MinerU PDF → Markdown, chapter-filtered
- **Extractor**: LLM structured extraction (Problem/Method/Gap/Debate/Conclusion)
- **Dedup**: Triple-ID resolution (DOI → arXiv → S2 → OpenAlex → title fuzzy)
- **Graph**: NetworkX in-memory + SQLite persistence, rule-based closure
- **Report**: Per-paper JSON with citation coverage & boundary alerts
