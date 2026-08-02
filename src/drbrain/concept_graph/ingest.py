"""Corpus ingestion service for the concept graph layer.

Pulls paper metadata from a :class:`CorpusSource` and writes it into the existing
``papers`` / ``paper_ids`` tables plus the v9 ``corpus_sources`` provenance table.
Deduplication is ``unique_id``-first with DOI as a secondary key, enabling
incremental re-ingestion. Citation edges are harvested via ``fetch_relations``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from drbrain.concept_graph.sources.base import CorpusSource, PaperRecord
from drbrain.storage.database import Database


@dataclass
class IngestStats:
    """Counters summarising a corpus ingestion run."""

    fetched: int = 0
    inserted: int = 0
    skipped: int = 0
    citations: int = 0
    errors: list[str] = field(default_factory=list)


def _slugify(value: str) -> str:
    """Turn an arbitrary identifier into a compact, filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug[:120] or "paper"


def make_local_id(record: PaperRecord) -> str:
    """Derive a stable, unique ``local_id`` for a paper record.

    Prefers the DOI (globally unique) when present; otherwise falls back to a
    source-prefixed slug of the source ``unique_id``.
    """
    if record.doi:
        return record.doi.strip().lower()
    return f"{record.source or 'src'}-{_slugify(record.unique_id)}"


def ingest_corpus(
    db: Database,
    source: CorpusSource,
    query: str | None = None,
    *,
    year_from: int | None = None,
    year_to: int | None = None,
    venues: list[str] | None = None,
    limit: int = 100,
    commit_every: int = 50,
) -> IngestStats:
    """Ingest up to ``limit`` papers from ``source`` into the database.

    Deduplication: skip if the source ``unique_id`` is already in
    ``corpus_sources``; otherwise, if a DOI already maps to a paper, link the
    provenance to that existing ``local_id`` instead of inserting a duplicate.

    Args:
        db: Target database.
        source: Corpus adapter to read from.
        query: Optional free-text query.
        year_from: Minimum publication year (inclusive).
        year_to: Maximum publication year (inclusive).
        venues: Optional venue filter list.
        limit: Maximum number of records to fetch.
        commit_every: Flush a commit after this many inserts.

    Returns:
        An :class:`IngestStats` summary.
    """
    stats = IngestStats()
    for record in source.search(
        query, year_from=year_from, year_to=year_to, venues=venues, limit=limit
    ):
        stats.fetched += 1
        if not record.unique_id:
            stats.skipped += 1
            continue

        existing = db.find_corpus_source(source.name, record.unique_id)
        if existing:
            stats.skipped += 1
            continue

        local_id = make_local_id(record)
        if record.doi:
            doi_owner = db.find_local_id_by_doi(record.doi.strip().lower())
            if doi_owner:
                db.insert_corpus_source(doi_owner, source.name, record.unique_id)
                stats.skipped += 1
                continue

        _insert_paper(db, local_id, record)
        db.insert_corpus_source(local_id, source.name, record.unique_id)
        stats.inserted += 1
        if stats.inserted % commit_every == 0:
            db.conn.commit()

    db.conn.commit()
    logger.info(
        "[cg.ingest] {} fetched={} inserted={} skipped={}",
        source.name,
        stats.fetched,
        stats.inserted,
        stats.skipped,
    )
    return stats


def _insert_paper(db: Database, local_id: str, record: PaperRecord) -> None:
    authors = "; ".join(a.name for a in record.authors if a.name)
    db.insert_paper(
        local_id=local_id,
        title=record.title,
        year=record.year,
        status="extracted",
        paper_type="paper",
        journal=record.venue,
        citation_count=record.citation_count,
        authors=authors,
    )
    if record.abstract:
        db.set_paper_abstract(local_id, record.abstract)
    doi = record.doi.strip().lower() if record.doi else None
    openalex_id = record.unique_id if record.source == "openalex" else None
    db.insert_paper_ids(local_id, doi=doi, openalex_id=openalex_id)


def ingest_citations(
    db: Database,
    source: CorpusSource,
    *,
    limit: int | None = None,
    commit_every: int = 20,
) -> IngestStats:
    """Harvest citation edges for papers previously ingested from ``source``.

    Args:
        db: Target database.
        source: Corpus adapter used to resolve relations.
        limit: Cap on the number of papers to process (None = all).
        commit_every: Flush a commit after this many papers.

    Returns:
        An :class:`IngestStats` summary (``citations`` counts new edges).
    """
    stats = IngestStats()
    rows = db.conn.execute(
        "SELECT local_id, source_unique_id FROM corpus_sources WHERE source = ?",
        (source.name,),
    ).fetchall()
    if limit is not None:
        rows = rows[:limit]

    for local_id, source_unique_id in rows:
        relations = source.fetch_relations(source_unique_id)
        if relations is None:
            continue
        for item in relations.references + relations.citations:
            cited_id = item.get("id")
            if not cited_id:
                continue
            db.insert_paper_citation(local_id, cited_id, source=source.name)
            stats.citations += 1
        stats.fetched += 1
        if stats.fetched % commit_every == 0:
            db.conn.commit()

    db.conn.commit()
    logger.info(
        "[cg.ingest] citations for {} papers={} edges={}",
        source.name,
        stats.fetched,
        stats.citations,
    )
    return stats
