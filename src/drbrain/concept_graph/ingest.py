"""Corpus ingestion service for the concept graph layer.

Pulls paper metadata from a :class:`CorpusSource` and writes it into the existing
``papers`` / ``paper_ids`` tables plus the v9 ``corpus_sources`` provenance table.
Deduplication is ``unique_id``-first with DOI as a secondary key, enabling
incremental re-ingestion. Citation edges are harvested via ``fetch_relations``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from functools import wraps

from loguru import logger

from drbrain.concept_graph.sources.base import CorpusSource, PaperRecord
from drbrain.dedup.resolver import (
    PaperIDs,
    canonical_paper_id,
)
from drbrain.security import safe_error
from drbrain.storage.database import Database


@dataclass
class IngestStats:
    """Counters summarising a corpus ingestion run."""

    fetched: int = 0
    inserted: int = 0
    skipped: int = 0
    citations: int = 0
    errors: list[str] = field(default_factory=list)


def _serialized_ingest(func):
    """Hold the database write lock across an entire ingest batch."""

    @wraps(func)
    def wrapped(db: Database, *args, **kwargs):
        with db.write_lock():
            return func(db, *args, **kwargs)

    return wrapped


def _record_ids(record: PaperRecord) -> PaperIDs:
    """Map source-specific fields to the shared external-ID vocabulary.

    ``PaperRecord`` intentionally keeps a small, source-neutral shape.  A few
    adapters expose an identifier only through ``unique_id``; recognising
    those adapter names here keeps the canonical local ID and the persisted
    ``paper_ids`` row derived from exactly the same identity.
    """
    source = str(record.source or "").strip().lower()
    unique_id = str(record.unique_id or "").strip()
    if "\x00" in unique_id:
        raise ValueError("paper source identifier must not contain NUL bytes")
    ids = PaperIDs(doi=record.doi)
    if source == "openalex":
        ids.openalex_id = unique_id
    elif source in {"arxiv", "arxiv-latex", "arxiv_latex"}:
        ids.arxiv = unique_id
    elif source in {"s2", "semantic-scholar", "semanticscholar"}:
        ids.s2_id = unique_id
    return ids.normalized()


def make_local_id(record: PaperRecord) -> str:
    """Derive a stable, unique ``local_id`` for a paper record.

    All corpus sources use the same hashed canonical identity as the rest of
    DrBrain.  A source-qualified key is retained as the final fallback so two
    providers that reuse a short identifier cannot collide.
    """
    source = str(record.source or "source").strip().lower() or "source"
    unique_id = str(record.unique_id or "").strip()
    if not unique_id:
        raise ValueError("paper source identifier must not be empty")
    return canonical_paper_id(
        _record_ids(record),
        title=record.title,
        year=record.year,
        source_key=f"{source}:{unique_id}",
    )


@_serialized_ingest
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
    if commit_every <= 0:
        raise ValueError("commit_every must be a positive integer")

    stats = IngestStats()
    # A caller may batch corpus and citation work in one transaction.  Keep
    # that transaction open while still isolating each record with a
    # savepoint; standalone calls retain the historical auto-commit behavior.
    caller_transaction = db.conn.in_transaction
    for record in source.search(
        query, year_from=year_from, year_to=year_to, venues=venues, limit=limit
    ):
        stats.fetched += 1
        if not record.unique_id:
            stats.skipped += 1
            continue

        savepoint = f"cg_ingest_{uuid.uuid4().hex}"
        db.conn.execute(f"SAVEPOINT {savepoint}")
        record_inserted = False
        record_skipped = False
        try:
            existing = db.find_corpus_source(source.name, record.unique_id)
            if existing:
                record_skipped = True
            else:
                local_id = make_local_id(record)
                record_ids = _record_ids(record)
                owner = None
                for kind in ("doi", "arxiv", "s2_id", "openalex_id"):
                    value = getattr(record_ids, kind)
                    if value:
                        owner = db.get_paper_by_external_id(kind, value)
                        if owner:
                            break
                if owner:
                    db.insert_corpus_source(owner, source.name, record.unique_id)
                    record_skipped = True
                else:
                    _insert_paper(db, local_id, record)
                    db.insert_corpus_source(local_id, source.name, record.unique_id)
                    record_inserted = True

            db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception as exc:  # noqa: BLE001 - isolate one source record
            try:
                db.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            finally:
                db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            message = (
                f"{safe_error(record.unique_id, limit=120)}: "
                f"{safe_error(f'{type(exc).__name__}: {exc}')}"
            )
            stats.errors.append(message)
            logger.warning("[cg.ingest] record failed; rolled back {}", message)
            continue

        if record_inserted:
            stats.inserted += 1
        elif record_skipped:
            stats.skipped += 1
        if not caller_transaction and stats.inserted > 0 and stats.inserted % commit_every == 0:
            db.conn.commit()

    if not caller_transaction:
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
        strict=True,
    )
    if record.abstract:
        db.set_paper_abstract(local_id, record.abstract)
    ids = _record_ids(record)
    # Corpus ingestion is an identity-establishing path: silently dropping a
    # conflicting external ID would persist a paper that cannot be resolved
    # back to its source record.  Let the record savepoint roll the whole row
    # back instead.
    db.insert_paper_ids(
        local_id,
        doi=ids.doi,
        arxiv=ids.arxiv,
        s2_id=ids.s2_id,
        openalex_id=ids.openalex_id,
        strict=True,
    )
    for kw in record.keywords:
        if kw.strip():
            db.insert_paper_term(local_id, kw.strip(), kind="keyword")
    for topic in record.topics:
        if topic.strip():
            db.insert_paper_term(local_id, topic.strip(), kind="topic")


@_serialized_ingest
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
    if commit_every <= 0:
        raise ValueError("commit_every must be a positive integer")

    stats = IngestStats()
    rows = db.conn.execute(
        "SELECT local_id, source_unique_id FROM corpus_sources WHERE source = ?",
        (source.name,),
    ).fetchall()
    if limit is not None:
        rows = rows[:limit]

    caller_transaction = db.conn.in_transaction
    for local_id, source_unique_id in rows:
        stats.fetched += 1
        savepoint = f"cg_citations_{uuid.uuid4().hex}"
        db.conn.execute(f"SAVEPOINT {savepoint}")
        record_citations = 0
        try:
            relations = source.fetch_relations(source_unique_id)
            if relations is not None:
                # references: this paper cites item.id  ->  (local_id -> item.id)
                for item in relations.references:
                    cited_id = item.get("id")
                    if cited_id:
                        db.insert_paper_citation(local_id, cited_id, source=source.name)
                        record_citations += 1
                # citations: item.id cites this paper  ->  (item.id -> local_id)
                for item in relations.citations:
                    citing_id = item.get("id")
                    if citing_id:
                        db.insert_paper_citation(citing_id, local_id, source=source.name)
                        record_citations += 1
            db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception as exc:  # noqa: BLE001 - isolate one relation response
            try:
                db.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            finally:
                db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            message = f"{source_unique_id}: {type(exc).__name__}: {exc}"
            stats.errors.append(message)
            logger.warning("[cg.ingest] citation record failed; rolled back {}", message)
            continue
        stats.citations += record_citations
        if not caller_transaction and stats.fetched % commit_every == 0:
            db.conn.commit()

    if not caller_transaction:
        db.conn.commit()
    logger.info(
        "[cg.ingest] citations for {} papers={} edges={}",
        source.name,
        stats.fetched,
        stats.citations,
    )
    return stats
