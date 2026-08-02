"""CLI commands for the concept graph (co-occurrence) layer.

Exposed as the ``drbrain cg`` sub-app. Additive to the existing CLI; does not
touch the full-text parser or RAG commands.
"""

from __future__ import annotations

import json

import typer
from loguru import logger

from drbrain.cli._common import open_db

cg_app = typer.Typer(help="Concept co-occurrence graph layer (build / embed / predict / recommend)")


@cg_app.command("ingest")
def cg_ingest_cmd(
    ctx: typer.Context,
    query: str = typer.Argument(None, help="Free-text search query (optional with filters)"),
    source: str = typer.Option(
        "sciverse", "--source", "-s", help="Corpus source: sciverse | openalex"
    ),
    year_from: int = typer.Option(None, "--year-from", help="Minimum publication year"),
    year_to: int = typer.Option(None, "--year-to", help="Maximum publication year"),
    venue: list[str] = typer.Option(None, "--venue", help="Venue filter (repeatable)"),
    limit: int = typer.Option(100, "--limit", "-n", help="Max records to fetch"),
    with_citations: bool = typer.Option(
        False, "--with-citations", help="Also harvest citation edges"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON stats"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Count matches without writing"),
) -> None:
    """Ingest paper metadata from an external academic API into the library."""
    from drbrain.concept_graph.ingest import ingest_citations, ingest_corpus
    from drbrain.concept_graph.sources.registry import get_source

    cfg = ctx.obj["config"]
    try:
        src = get_source(source)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if dry_run:
        count = sum(
            1
            for _ in src.search(
                query, year_from=year_from, year_to=year_to, venues=venue or None, limit=limit
            )
        )
        payload = {"dry_run": True, "source": source, "would_fetch": count}
        typer.echo(json.dumps(payload) if json_output else f"[dry-run] {source}: {count} matches")
        return

    with open_db(cfg) as db:
        stats = ingest_corpus(
            db, src, query, year_from=year_from, year_to=year_to, venues=venue or None, limit=limit
        )
        if with_citations:
            cit = ingest_citations(db, src)
            stats.citations = cit.citations

    payload = {
        "source": source,
        "fetched": stats.fetched,
        "inserted": stats.inserted,
        "skipped": stats.skipped,
        "citations": stats.citations,
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        typer.echo(
            f"[cg.ingest] {source}: fetched={stats.fetched} inserted={stats.inserted} "
            f"skipped={stats.skipped} citations={stats.citations}"
        )
    logger.info("[cg.ingest] done: {}", payload)


@cg_app.command("build")
def cg_build_cmd(
    ctx: typer.Context,
    source: str = typer.Option(
        "terms", "--source", "-s", help="Concept source: terms | concepts | abstract"
    ),
    min_freq: int = typer.Option(
        3, "--min-freq", help="Minimum document frequency to keep a concept"
    ),
    min_words: int = typer.Option(2, "--min-words", help="Minimum words in a concept label"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON stats"),
) -> None:
    """Build the concept co-occurrence graph (cliques + frequency filtering)."""
    from drbrain.concept_graph.builder import apply_filter, build_cliques

    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        edges = build_cliques(db, source=source)
        stats = apply_filter(db, min_freq=min_freq, min_words=min_words)

    payload = {"edges": edges, "total_concepts": stats["total_concepts"], "kept": stats["kept"]}
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        typer.echo(
            f"[cg.build] edges={edges} concepts={stats['total_concepts']} kept={stats['kept']}"
        )


@cg_app.command("embed")
def cg_embed_cmd(
    ctx: typer.Context,
    context: bool = typer.Option(
        False, "--context", help="Average label with containing-paper titles"
    ),
    model_name: str = typer.Option("", "--model", help="Model label recorded in the DB"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON stats"),
) -> None:
    """Compute semantic embeddings for concept nodes."""
    from drbrain.concept_graph.embeddings import compute_concept_embeddings

    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        count = compute_concept_embeddings(db, cfg, context=context, model_name=model_name)
    if json_output:
        typer.echo(json.dumps({"embedded": count}))
    else:
        typer.echo(f"[cg.embed] embedded {count} concepts")


@cg_app.command("neighbors")
def cg_neighbors_cmd(
    ctx: typer.Context,
    concept: str = typer.Argument(..., help="Query concept label"),
    top: int = typer.Option(10, "--top", "-k", help="Number of neighbours"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    """Show the nearest concepts to a query concept by embedding similarity."""
    from drbrain.concept_graph.embeddings import nearest_neighbors

    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        results = nearest_neighbors(db, concept, k=top)
    if json_output:
        typer.echo(
            json.dumps([{"label": l, "score": round(s, 4)} for l, s in results], ensure_ascii=False)
        )
        return
    if not results:
        typer.echo(f"No embedding found for '{concept}'.")
        return
    for label, score in results:
        typer.echo(f"  {score:.4f}  {label}")


@cg_app.command("map")
def cg_map_cmd(
    ctx: typer.Context,
    output: str = typer.Option("concept_map.html", "--output", "-o", help="Output HTML path"),
) -> None:
    """Export an interactive UMAP concept map to HTML."""
    from drbrain.concept_graph.map import export_html

    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        path = export_html(db, output)
    typer.echo(f"[cg.map] wrote {path}")


@cg_app.command("predict")
def cg_predict_cmd(
    ctx: typer.Context,
    feat_cutoff: int = typer.Option(
        ..., "--feat-cutoff", help="Feature snapshot year (G_t, leakage-free)"
    ),
    train_end: int = typer.Option(..., "--train-end", help="End of training label window"),
    test_end: int = typer.Option(..., "--test-end", help="End of test label window"),
    model: str = typer.Option("mixture", "--model", help="baseline | embed | mixture"),
    neg_ratio: int = typer.Option(1, "--neg-ratio", help="Negatives per positive"),
    top_k: int = typer.Option(50, "--top-k", help="k for Precision/Recall@k"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON metrics"),
) -> None:
    """Train/evaluate temporal link prediction on the concept graph."""
    import numpy as np

    from drbrain.concept_graph.dataset import feature_years, temporal_pairs
    from drbrain.concept_graph.embeddings import load_concept_embeddings
    from drbrain.concept_graph.eval import precision_recall_at_k, roc_auc, stratify_by_dprev
    from drbrain.concept_graph.features import semantic_features, topo_features, yearly_subgraph
    from drbrain.concept_graph.link_predict import MixtureEnsemble, MLPLinkClassifier

    if model not in ("baseline", "embed", "mixture"):
        typer.echo(f"Invalid model '{model}'.", err=True)
        raise typer.Exit(1)

    cfg = ctx.obj["config"]
    years = feature_years(feat_cutoff)
    with open_db(cfg) as db:
        train = temporal_pairs(db, feat_cutoff, train_end, neg_ratio=neg_ratio)
        test = temporal_pairs(db, train_end, test_end, neg_ratio=neg_ratio)
        embeddings = load_concept_embeddings(db)

        def _pairs_labels(data: dict) -> tuple[list[tuple[str, str]], np.ndarray]:
            pairs = list(data["positives"]) + list(data["negatives"])
            labels = np.array([1] * len(data["positives"]) + [0] * len(data["negatives"]))
            return pairs, labels

        train_pairs, y_train = _pairs_labels(train)
        test_pairs, y_test = _pairs_labels(test)

        use_topo = model in ("baseline", "mixture")
        use_sem = model in ("embed", "mixture") and bool(embeddings)
        if model in ("embed", "mixture") and not embeddings:
            typer.echo("[cg.predict] no concept embeddings; falling back to baseline", err=True)
            use_sem = False
            use_topo = True

        def _topo_matrix(pairs):
            return np.array([topo_features(db, u, v, years) for u, v in pairs])

        def _sem_matrix(pairs):
            return np.array([semantic_features(u, v, embeddings) for u, v in pairs])

        if use_topo and use_sem:
            ens = MixtureEnsemble(MLPLinkClassifier(), MLPLinkClassifier(), weight_a=0.6)
            ens.fit(_topo_matrix(train_pairs), _sem_matrix(train_pairs), y_train)
            scores = ens.predict_proba(_topo_matrix(test_pairs), _sem_matrix(test_pairs))
        elif use_sem:
            clf = MLPLinkClassifier().fit(_sem_matrix(train_pairs), y_train)
            scores = clf.predict_proba(_sem_matrix(test_pairs))
        else:
            clf = MLPLinkClassifier().fit(_topo_matrix(train_pairs), y_train)
            scores = clf.predict_proba(_topo_matrix(test_pairs))

        g_train_end = yearly_subgraph(db, train_end)
        prec, rec = precision_recall_at_k(y_test, scores, top_k)
        metrics = {
            "model": model,
            "test_pairs": int(len(y_test)),
            "test_positives": int(y_test.sum()),
            "auc": round(roc_auc(y_test, scores), 4),
            f"precision@{top_k}": round(prec, 4),
            f"recall@{top_k}": round(rec, 4),
            "by_dprev": {
                str(d): {
                    "count": v["count"],
                    "positives": v["positives"],
                    "auc": round(v["auc"], 4),
                }
                for d, v in stratify_by_dprev(test_pairs, y_test, scores, g_train_end).items()
            },
        }

    if json_output:
        typer.echo(json.dumps(metrics, ensure_ascii=False))
    else:
        typer.echo(
            f"[cg.predict] model={model} auc={metrics['auc']} "
            f"P@{top_k}={metrics[f'precision@{top_k}']} R@{top_k}={metrics[f'recall@{top_k}']}"
        )


@cg_app.command("recommend")
def cg_recommend_cmd(
    ctx: typer.Context,
    author: str = typer.Option(..., "--author", "-a", help="Author name (substring match)"),
    top: int = typer.Option(25, "--top", "-k", help="Suggestions per section"),
    sim_min: float = typer.Option(0.15, "--sim-min", help="Minimum similarity to keep"),
    sim_max: float = typer.Option(0.95, "--sim-max", help="Maximum similarity to keep"),
    max_hub_freq: int = typer.Option(
        None, "--max-hub-freq", help="Exclude concepts above this doc_freq"
    ),
    curate: bool = typer.Option(False, "--curate", help="Add LLM curation paragraph"),
    output: str = typer.Option(None, "--output", "-o", help="Write a markdown report to this path"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    """Recommend novel research-direction concept combinations for an author."""
    from drbrain.concept_graph.recommend import llm_curation, recommend_combinations

    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        result = recommend_combinations(
            db, author, top_k=top, sim_min=sim_min, sim_max=sim_max, max_hub_freq=max_hub_freq
        )
        curation = ""
        if curate:
            models = cfg.llm.models if hasattr(cfg, "llm") else []
            curation = llm_curation(result["own_x_other"], models)

    if json_output:
        result["curation"] = curation
        typer.echo(json.dumps(result, ensure_ascii=False))
        return

    lines = [f"# Research directions for: {author}", ""]
    lines.append(
        f"Own concepts ({len(result['c_own'])}): " + ", ".join(result["c_own"]) or "(none)"
    )
    lines.append("")
    lines.append("## Suggested new combinations (own × other)")
    for item in result["own_x_other"]:
        lines.append(f"- {item['concept']}  (score {item['score']})")
    if result["many_own_x_other"]:
        lines.append("")
        lines.append("## Connects to many own concepts")
        for item in result["many_own_x_other"]:
            lines.append(f"- {item['concept']}  (related own: {item['related_own_count']})")
    if curation:
        lines.append("")
        lines.append("## LLM curation")
        lines.append(curation)
    report = "\n".join(lines)
    if output:
        from pathlib import Path

        Path(output).write_text(report, encoding="utf-8")
        typer.echo(f"[cg.recommend] wrote {output}")
    else:
        typer.echo(report)
