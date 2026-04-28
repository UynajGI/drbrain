"""Streamlit frontend — graph visualization & query."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="DrBrain", page_icon="🧠", layout="wide")

st.title("DrBrain — Academic Knowledge Graph")

DB_PATH = Path("data/drbrain.db")
REPORTS_DIR = Path("data/reports")

st.sidebar.header("Filters")
selected_type = st.sidebar.selectbox(
    "Concept type",
    ["All", "Problem", "Method", "Conclusion", "Debate", "Gap", "Actor"],
)
time_range = st.sidebar.slider("Year range", 2000, 2026, (2015, 2026))

if DB_PATH.exists():
    st.success(f"Database found at {DB_PATH}")
else:
    st.warning("No database yet. Run: uv run drbrain ingest <paper.pdf>")

st.header("Graph Overview")
st.caption("NetworkX visualization will render here once data is loaded")

if REPORTS_DIR.exists():
    reports = list(REPORTS_DIR.glob("*.json"))
    if reports:
        st.header(f"Reports ({len(reports)} papers)")
        for rp in sorted(reports)[:10]:
            st.markdown(f"- `{rp.stem}`")
        if len(reports) > 10:
            st.caption(f"...and {len(reports) - 10} more")
    else:
        st.info("No reports generated yet")
