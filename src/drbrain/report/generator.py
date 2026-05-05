"""Single-paper JSON report with citation coverage stats."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


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

    def save(self, output_dir: str | Path = "data/reports") -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.local_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def total_refs_and_citations(report: PaperReport) -> int:
    return len(report.references) + len(report.citations)
