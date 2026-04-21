"""DrBrain CLI — ingest, query, serve."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import typer

from brbrain.parser.pdf import extract_pdf, filter_sections
from brbrain.extractor.concept import extract_concepts
from brbrain.dedup.resolver import DedupEngine, PaperIDs
from brbrain.storage.database import Database
from brbrain.graph.engine import GraphEngine
from brbrain.report.generator import PaperReport

app = typer.Typer(help="DrBrain — Academic Knowledge Graph System")
db = Database()
graph = GraphEngine()
dedup = DedupEngine(db)


@app.command()
def ingest(pdf_path: str, model: str = "openai/gpt-4o", api_base: str | None = None):
    """Ingest a paper PDF into the knowledge graph."""
    parsed = extract_pdf(pdf_path)

    # ID resolution
    ids = PaperIDs(
        doi=parsed.doi,
        arxiv=parsed.arxiv,
    )
    local_id = dedup.resolve(ids, title=parsed.title, year=parsed.year)
    is_new = local_id is None

    if is_new:
        local_id = f"paper_{uuid.uuid4().hex[:8]}"
        db.insert_paper(local_id, parsed.title, parsed.year, "uploaded")
        db.insert_paper_ids(local_id, doi=ids.doi, arxiv=ids.arxiv)
        db.commit()
        typer.echo(f"[new] {local_id}: {parsed.title}")
    else:
        db.upgrade_placeholder(local_id)
        db.commit()
        typer.echo(f"[upgrade] {local_id}: {parsed.title}")

    # TODO: LLM extraction, concept insertion, edge creation
    typer.echo(f"  Status: {'new paper' if is_new else 'upgraded placeholder'}")
    typer.echo(f"  Sections: {len(filter_sections(parsed.raw_md))} high-signal blocks")


@app.command()
def query(text: str):
    """Query the knowledge graph (natural language)."""
    typer.echo(f"Query: {text}")
    typer.echo("Not yet implemented — coming soon: NL → SQL/graph traversal")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8501):
    """Launch Streamlit UI."""
    typer.echo(f"Starting Streamlit on {host}:{port}")
    typer.echo("Run: uv run streamlit run src/brbrain/api/app.py")


@app.command()
def stats():
    """Show graph statistics."""
    papers = db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    concepts = db.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    edges = db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    typer.echo(f"Papers: {papers}")
    typer.echo(f"Concepts: {concepts}")
    typer.echo(f"Edges: {edges}")


if __name__ == "__main__":
    app()
