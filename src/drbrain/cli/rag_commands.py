"""RAG subcommands: ``drbrain rag index`` / ``drbrain rag eval``.

The ``rag`` Typer sub-app hosts LlamaIndex-driven operations. T3 ships the
``index`` command (build/persist the vector + BM25 index from PageIndex
assets); T7 ships the ``eval`` command (golden-set retriever/ragas evaluation,
baseline report into ``docs/llamaindex-eval-baseline.md``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from drbrain.cli._common import open_db

rag_app = typer.Typer(help="LlamaIndex RAG layer operations")

console = Console()


@rag_app.command("index")
def rag_index_cmd(
    ctx: typer.Context,
    force: bool = typer.Option(
        False, "--force", "-f", help="Force full rebuild (ignore content_hash)"
    ),
    paper: list[str] = typer.Option(
        None, "--paper", help="Restrict to paper local_id (repeatable: --paper A --paper B)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Build (or incrementally update) the LlamaIndex vector + BM25 indexes.

    Reads each paper's tree.json/raw.md into LlamaIndex nodes, embeds only
    changed nodes, and persists everything under ``llamaindex.storage_dir``.
    Re-run to pick up changed papers; ``--force`` rebuilds from scratch.
    """
    cfg = ctx.obj["config"]

    available = False
    build = None
    try:
        from drbrain.rag.config import get_llamaindex_config
        from drbrain.rag.indexer import _LLAMA_INDEX_AVAILABLE, build_index

        available, build = _LLAMA_INDEX_AVAILABLE, build_index
        max_node_tokens = get_llamaindex_config(cfg).max_node_tokens
    except ImportError:  # pragma: no cover - defensive
        available, build, max_node_tokens = False, None, None

    if not available or build is None:
        typer.echo(
            "llama-index is not installed. Run: uv add llama-index-core llama-index-retrievers-bm25",
            err=True,
        )
        raise typer.Exit(1)

    with open_db(cfg) as db:
        stats = build(
            cfg,
            db,
            paper_ids=paper or None,
            force=force,
            max_node_tokens=max_node_tokens,
        )

    if json_output:
        typer.echo(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
        return

    table = Table(title="LlamaIndex RAG Index")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    for key, label in (
        ("papers", "Papers indexed"),
        ("nodes", "Indexed nodes"),
        ("chunked", "Long nodes split (T9)"),
        ("embedded", "Embedded this run"),
        ("unchanged", "Unchanged (cached)"),
        ("removed", "Removed nodes"),
        ("bm25_nodes", "BM25 documents"),
    ):
        table.add_row(label, str(stats.get(key, 0)))
    table.add_row("Storage dir", str(stats.get("storage_dir", "")))
    if max_node_tokens:
        table.add_row("Max node tokens", str(max_node_tokens))
    console.print(table)


@rag_app.command("health")
def rag_health_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output the readiness report as JSON"),
):
    """Check RAG index readiness without querying, embedding, or writing."""
    from drbrain.rag.indexer import get_index_health

    report = get_index_health(ctx.obj["config"])
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, default=str))
    else:
        typer.echo(f"RAG health: {report['status']} ({report['storage_dir']})")
        if report["reasons"]:
            typer.echo("Reasons: " + ", ".join(report["reasons"]))
    if not report["ready"]:
        raise typer.Exit(1)


@rag_app.command("eval")
def rag_eval_cmd(
    ctx: typer.Context,
    split: str = typer.Option("dev", "--split", help="Golden split: dev|val|test"),
    metrics: str = typer.Option(
        "retriever", "--metrics", help="Metrics to run: retriever|ragas|semantic|qagen|all"
    ),
    k: str = typer.Option("5,10", "--k", help="Comma-separated top-k values (retriever)"),
    n: int = typer.Option(10, "--n", help="Max golden queries for the ragas/semantic eval"),
    n_nodes: int = typer.Option(25, "--n-nodes", help="Nodes to generate QA pairs from (qagen)"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
    out: str = typer.Option(
        "docs/llamaindex-eval-baseline.md",
        "--out",
        help="Baseline markdown report path (timestamped section is appended)",
    ),
    no_write_report: bool = typer.Option(
        False,
        "--no-write-report",
        help="Print evaluation results without modifying the markdown baseline report",
    ),
):
    """Run golden-set evaluation (retriever HitRate/MRR and/or RAGAS-style).

    Retrieves each golden query of ``--split`` through the T4 fusion retriever
    (``--metrics retriever``, HitRate@K/MRR@K at paper and node level) and/or
    synthesizes answers via the T5 query engine and scores them with the
    self-written RAGAS-style 4 metrics (``--metrics ragas``: faithfulness,
    answer_relevancy, context_precision, answer_correctness). Results are
    printed and appended as a timestamped section to ``--out``
    (``docs/llamaindex-eval-baseline.md`` by default).
    """
    cfg = ctx.obj["config"]

    llama_available = True
    format_eval_report: Any = None
    run_ragas_eval: Any = None
    run_retriever_eval: Any = None
    run_semantic_eval: Any = None
    run_qagen: Any = None
    try:
        from drbrain.rag import eval as _eval_mod
        from drbrain.rag.config import get_llamaindex_config

        llama_available = _eval_mod._LLAMA_INDEX_AVAILABLE
        format_eval_report = _eval_mod.format_eval_report
        run_ragas_eval = _eval_mod.run_ragas_eval
        run_retriever_eval = _eval_mod.run_retriever_eval
        run_semantic_eval = _eval_mod.run_semantic_eval
        run_qagen = _eval_mod.run_qagen
    except ImportError:  # pragma: no cover - defensive
        llama_available = False

    if not llama_available or any(fn is None for fn in (run_retriever_eval, run_ragas_eval)):
        typer.echo(
            "llama-index is not installed. Run: uv add llama-index-core llama-index-retrievers-bm25",
            err=True,
        )
        raise typer.Exit(1)

    if metrics not in ("retriever", "ragas", "semantic", "qagen", "all"):
        typer.echo(
            f"--metrics must be one of retriever|ragas|semantic|qagen|all, got {metrics!r}",
            err=True,
        )
        raise typer.Exit(2)

    ks: list[int] = []
    for part in k.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ks.append(int(part))
        except ValueError:
            typer.echo(f"--k must be a comma-separated list of positive ints, got {k!r}", err=True)
            raise typer.Exit(2)
    if not ks or any(kv <= 0 for kv in ks):
        typer.echo(f"--k must be a comma-separated list of positive ints, got {k!r}", err=True)
        raise typer.Exit(2)

    li = get_llamaindex_config(cfg)
    valid_splits = [str(s) for s in (li.eval.split or ["dev", "val", "test"])]
    if split not in valid_splits:
        typer.echo(
            f"--split must be one of {valid_splits}, got {split!r} (golden file: {li.eval.golden_set})",
            err=True,
        )
        raise typer.Exit(2)

    with open_db(cfg) as db:
        retriever_results = None
        ragas_results = None
        semantic_results = None
        qagen_results = None
        if metrics in ("retriever", "all"):
            retriever_results = run_retriever_eval(cfg, db, split=split, ks=ks)
        if metrics in ("ragas", "all"):
            ragas_results = run_ragas_eval(cfg, db, split=split, n=n)
        if metrics in ("semantic", "all"):
            semantic_results = run_semantic_eval(cfg, db, split=split, n=n)
        if metrics in ("qagen", "all"):
            qagen_results = run_qagen(cfg, n_nodes=n_nodes)

    if json_output:
        payload: dict[str, Any] = {"split": split, "metrics": metrics}
        if retriever_results is not None:
            payload["retriever"] = retriever_results
        if ragas_results is not None:
            payload["ragas"] = ragas_results
        if semantic_results is not None:
            payload["semantic"] = semantic_results
        if qagen_results is not None:
            payload["qagen"] = qagen_results
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        for label, results in (
            ("Retriever", retriever_results),
            ("RAGAS-style", ragas_results),
            ("Semantic", semantic_results),
            ("QAgen", qagen_results),
        ):
            if results is None:
                continue
            status = results.get("status")
            typer.echo(
                f"[{label}] status={status} split={results.get('split')} queries={results.get('queries', 0)}"
            )
            if status == "ok" and "hit_rate" in results:
                for level in ("paper", "node"):
                    typer.echo(
                        f"  {level}: HitRate="
                        + ", ".join(
                            f"@{k}={results['hit_rate'][level][str(k)]}" for k in results["ks"]
                        )
                        + " MRR="
                        + ", ".join(f"@{k}={results['mrr'][level][str(k)]}" for k in results["ks"])
                    )
            elif status == "ok" and "metrics" in results:
                for key, info in results["metrics"].items():
                    typer.echo(f"  {key}: mean={info['mean']} (missing={info['missing']})")
            elif status == "ok" and "mean_similarity" in results:
                typer.echo(
                    f"  mean_similarity={results['mean_similarity']} "
                    f"pass_rate={results['pass_rate']} (threshold={results['threshold']}, "
                    f"scored={results['scored']}, missing={results['missing']})"
                )
            elif status == "ok" and "generated" in results:
                typer.echo(
                    f"  generated={results['generated']} QA pairs from {results['nodes_used']} "
                    f"nodes -> {results['golden_set']}"
                )
                typer.echo(f"  note: {results['note']}")
            if results.get("reason"):
                typer.echo(f"  reason: {results['reason']}")

    report = format_eval_report(cfg, retriever_results, ragas_results)
    out_path = Path(out)
    # Direct Python callers historically invoke Typer command functions too;
    # an omitted option is then an OptionInfo object rather than its bool
    # default. CLI invocation itself always supplies a bool.
    skip_report = (
        no_write_report
        if isinstance(no_write_report, bool)
        else bool(getattr(no_write_report, "default", False))
    )
    if not skip_report:
        from drbrain.rag.eval import _write_text_atomically

        existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        content = existing.rstrip() + "\n\n---\n\n" + report if existing else report
        _write_text_atomically(out_path, content)
    if not json_output:
        if skip_report:
            typer.echo("Baseline report not written (--no-write-report)")
        else:
            typer.echo(f"Baseline appended to {out_path}")
