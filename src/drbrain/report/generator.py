"""Single-paper JSON report with citation coverage stats."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from drbrain.runtime import RuntimeContext
from drbrain.storage.paths import paper_fs_key, writable_artifact_path


def _runtime_selected() -> bool:
    """Return whether this process explicitly selected a runtime root."""
    return "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ


def _default_reports_dir() -> Path:
    """Resolve implicit report output beneath the active runtime root."""
    if _runtime_selected():
        # ``runtime_root`` validates an explicitly empty selector instead of
        # silently falling back to the process working directory.
        from drbrain.runtime import runtime_root

        return runtime_root() / "data" / "reports"
    return Path("data/reports")


def _ensure_output_dir(path: str | Path, *, runtime: RuntimeContext | None = None) -> Path:
    """Create a report directory without following a symlink component."""
    if isinstance(path, str) and (not path or "\x00" in path):
        raise ValueError("report output directory must be a non-empty local path")
    out = Path(path).expanduser()
    if runtime is not None:
        out = runtime.assert_within_root(out, label="report output directory")

    # ``mkdir(parents=True, exist_ok=True)`` follows an intermediate alias.
    # Walk the lexical path before creating missing components, then verify it
    # again after creation so a stale link cannot redirect the report.
    for ancestor in (out, *out.parents):
        if ancestor.is_symlink():
            raise ValueError(f"report output directory contains a symlink: {ancestor}")
    current = out
    missing: list[Path] = []
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    if current.is_symlink() or not current.is_dir():
        raise ValueError(f"report output directory is not a real directory: {current}")
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"report output directory is not a real directory: {directory}")
    if out.is_symlink() or not out.is_dir():
        raise ValueError(f"report output directory is not a real directory: {out}")
    return out


@dataclass
class RefEntry:
    """A reference or citation entry."""

    title: str
    year: int | None
    ids: dict = field(default_factory=dict)
    in_graph: bool = False
    local_id: str | None = None


@dataclass
class PaperReport:
    """Complete report for a single paper."""

    local_id: str
    title: str
    year: int | None
    ids: dict = field(default_factory=dict)
    status: str = "uploaded"
    concepts: dict = field(default_factory=dict)
    arguments: list[dict] = field(default_factory=list)
    references: list[RefEntry] = field(default_factory=list)
    citations: list[RefEntry] = field(default_factory=list)
    validation: dict = field(default_factory=dict)

    @property
    def summary(self) -> dict:
        total_refs = len(self.references)
        total_cits = len(self.citations)
        refs_in = sum(1 for r in self.references if r.in_graph)
        cits_in = sum(1 for r in self.citations if r.in_graph)
        total = total_refs + total_cits
        coverage = (refs_in + cits_in) / total if total > 0 else 0.0
        return {
            "refs_in_graph": refs_in,
            "cits_in_graph": cits_in,
            "total_refs": total_refs,
            "total_cits": total_cits,
            "graph_coverage": round(coverage, 3),
        }

    @property
    def boundary_alert(self) -> dict:
        s = self.summary
        missing = [r for r in self.references if not r.in_graph and r.title]
        alert = {
            "missing_core_refs": len(missing) > 5,
            "isolated_subgraph": s["graph_coverage"] < 0.3 and total_refs_and_citations(self) > 10,
        }
        if self.validation.get("items_rejected", 0) > 0:
            alert["validation_failures"] = True
        return alert

    def to_dict(self) -> dict:
        return {
            "paper": {
                "local_id": self.local_id,
                "title": self.title,
                "year": self.year,
                "ids": self.ids,
                "status": self.status,
            },
            "concepts": self.concepts,
            "arguments": self.arguments,
            "references": [
                {
                    "title": r.title,
                    "year": r.year,
                    "ids": r.ids,
                    "in_graph": r.in_graph,
                    "local_id": r.local_id,
                }
                for r in self.references
            ],
            "citations": [
                {
                    "title": r.title,
                    "year": r.year,
                    "ids": r.ids,
                    "in_graph": r.in_graph,
                    "local_id": r.local_id,
                }
                for r in self.citations
            ],
            "summary": self.summary,
            "boundary_alert": self.boundary_alert,
            "validation": self.validation,
        }

    def save(self, output_dir: str | Path | None = None) -> Path:
        runtime = None
        if _runtime_selected():
            from drbrain.runtime import RuntimeContext

            runtime = RuntimeContext.create()
        out = _ensure_output_dir(
            output_dir if output_dir is not None else _default_reports_dir(),
            runtime=runtime,
        )
        # A database ID may be a DOI containing slashes; reports are always a
        # single safe filesystem component while the original ID remains in
        # the JSON payload.
        path = writable_artifact_path(out, f"{paper_fs_key(self.local_id)}.json")
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(out), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        return path


def total_refs_and_citations(report: PaperReport) -> int:
    return len(report.references) + len(report.citations)
