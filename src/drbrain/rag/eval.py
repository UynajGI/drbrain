"""Evaluation layer: golden set, retriever metrics, RAGAS-style metrics.

Ticket: T7 (评估体系). Depends on T3 (index layer) / T4 (fusion) / T5 (query
engine). Implements the design-doc §4.5 evaluation loop:

* :func:`build_golden_set` — semi-automated golden set construction from the
  ``test-run/papers`` corpus (query = title/abstract-derived question, relevant
  papers = source paper + same-topic papers, relevant nodes derived from
  ``tree.json`` structure). Idempotent: an existing file is left untouched
  unless ``force=True``.
* :func:`load_golden` — read the JSONL golden set, filtered by split.
* :func:`run_retriever_eval` — HitRate@K / MRR@K over the T4 fusion retriever
  (paper-level and node-level relevance), aggregated per split.
* :func:`run_ragas_eval` — self-written 4-metric prompt evaluation of the T5
  ``ask_llamaindex`` output (faithfulness / answer_relevancy /
  context_precision / answer_correctness), scored through the DrbrainLLM
  bridge (the drbrain fallback chain).
* :func:`format_eval_report` — markdown baseline report for
  ``docs/llamaindex-eval-baseline.md``.

Design decisions (llama-index-core 0.14.23):

* ``RetrieverEvaluator`` *is* importable in 0.14.23, but its ``evaluate``
  expects a single flat list of ``expected_ids`` per query (node-level only).
  Our golden set carries both paper-level and node-level relevance, and the
  framework's hit_rate/mrr semantics are awkward to bend for a fused
  multi-leg retriever — so the ticket's fallback clause is used: hit_rate/mrr
  are computed by hand (the math is a one-liner; the framework adds no value
  here).
* RAGAS is not installed (heavy dependency, optional extra per design §5) —
  the 4 metrics are self-written prompt evaluations through
  ``DrbrainLLM.complete`` (``call_text_with_fallback``, drbrain fallback chain
  intact). See :mod:`drbrain.rag.llm`.
* ``answer_correctness`` compares the generated answer against a
  ``reference_answer`` stored in the golden set — the abstract of the primary
  relevant paper (cheap, non-LLM ground truth; no golden answers were
  hand-written, keeping annotation cost low per ticket guidance).
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from drbrain.config import Config
from drbrain.rag.config import get_llamaindex_config

try:
    from llama_index.core.schema import NodeWithScore

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - envs without llama-index
    NodeWithScore = None  # type: ignore[assignment,misc]
    _LLAMA_INDEX_AVAILABLE = False

log = logging.getLogger(__name__)

__all__ = [
    "_LLAMA_INDEX_AVAILABLE",
    "build_golden_set",
    "format_eval_report",
    "load_golden",
    "run_ragas_eval",
    "run_retriever_eval",
]

#: Default dev/val/test split ratio for the golden set (design §4.5, 60/20/20).
DEFAULT_SPLIT_RATIO = (0.6, 0.2, 0.2)
#: Fixed shuffle seed so split assignment is deterministic across runs
#: (idempotent regeneration and reproducible baselines).
_SPLIT_SEED = 20260812
#: Cap on reference answers (abstracts) — enough text for a correctness check.
_REFERENCE_MAX_CHARS = 800
#: Cap on a single context chunk handed to the scoring LLM.
_CONTEXT_CHUNK_MAX_CHARS = 1500
#: Titles treated as "content" nodes when deriving relevant_nodes.
_CONTENT_TITLE_PREFIXES = (
    "abstract",
    "summary",
    "overview",
    "introduction",
    "results",
    "discussion",
    "conclusion",
    "experimental",
    "methods",
    "materials",
    "section ",
)
#: Titles preferred as the ``reference_answer`` source (abstract first).
_ABSTRACT_TITLE_PREFIXES = ("abstract", "summary")


# ── golden set ───────────────────────────────────────────────────────────────


def load_golden(cfg: Config | dict[str, Any] | None = None, split: str | None = None) -> list[dict]:
    """Load the golden set (JSONL), optionally filtered by split.

    Each line is ``{"query", "relevant_papers", "relevant_nodes", "split",
    "reference_answer"?}``. ``split=None`` returns every entry; a missing or
    unreadable golden file returns ``[]`` (never raises).
    """
    li = get_llamaindex_config(cfg)
    golden_path = Path(li.eval.golden_set)
    if not golden_path.exists():
        log.warning("[rag] golden set not found at %s", golden_path)
        return []
    out: list[dict[str, Any]] = []
    try:
        with golden_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    log.warning("[rag] skipping malformed golden line: %.80s", line)
                    continue
                if split is not None and item.get("split") != split:
                    continue
                out.append(item)
    except OSError as exc:  # pragma: no cover - defensive
        log.warning("[rag] cannot read golden set %s: %s", golden_path, exc)
        return []
    return out


#: Curated golden queries (T7). Each entry is ``{id, q, papers}`` where
#: ``papers`` lists relevant ``test-run/papers`` directory names (source paper
#: first, then same-topic papers). Queries are title/abstract-derived
#: questions covering common materials-science topics; relevance was curated
#: by hand (source paper + same-topic papers), not LLM-generated, to keep
#: annotation cost low and labels verifiable (ticket guidance).
_GOLDEN_QUERIES: list[dict[str, Any]] = [
    # ── perovskite / solar cells ─────────────────────────────────────────
    {
        "id": "pv-tin-surface-reconstruction",
        "q": "How does the multifunctional molecule phDMADBr form a protective surface layer on quasi-2D tin-based perovskite films to improve the stability of lead-free solar cells?",
        "papers": ["10.1002_adma.202308655"],
    },
    {
        "id": "pv-lead-free-challenges",
        "q": "What are the main challenges for lead-free perovskite solar cells based on tin halide perovskites, and how does Sn(II) oxidation affect device performance?",
        "papers": ["10.1002_adma.202308655"],
    },
    {
        "id": "pv-nanocrystal-emission-tuning",
        "q": "How is the green-to-blue emission of halide perovskite nanocrystals precisely controlled using terbium chloride as a chlorine source?",
        "papers": ["10.3390_nano11092390"],
    },
    {
        "id": "pv-ws2-invisible-solar-cell",
        "q": "How is a near-invisible solar cell fabricated using a monolayer of WS2?",
        "papers": ["10.1038_s41598-022-15352-x"],
    },
    {
        "id": "pv-si-potential-induced-degradation",
        "q": "What causes potential-induced degradation in encapsulant-less p-type crystalline silicon photovoltaic modules?",
        "papers": ["10.35848_1347-4065_acc9ce"],
    },
    # ── batteries ────────────────────────────────────────────────────────
    {
        "id": "bat-zn-anodes",
        "q": "What are the recent developments in three-dimensional Zn metal anodes for battery applications?",
        "papers": ["10.1002_inf2.12485"],
    },
    {
        "id": "bat-pre-metallization",
        "q": "What emerging pre-metallization technologies exist for rechargeable metal-ion batteries?",
        "papers": ["10.1002_smll.202306262"],
    },
    {
        "id": "bat-zn-anode-and-metal-ion",
        "q": "What recent strategies improve metal anodes for rechargeable batteries, including three-dimensional zinc anodes and pre-metallization approaches?",
        "papers": ["10.1002_inf2.12485", "10.1002_smll.202306262"],
    },
    {
        "id": "bat-go-pva-pb-composite",
        "q": "What is the structure and electrochemical performance of a graphene oxide/polyvinyl alcohol-formaldehyde composite loaded with Pb ions?",
        "papers": ["10.3390_polym14112303"],
    },
    # ── photocatalysis / heterogeneous catalysis ─────────────────────────
    {
        "id": "cat-h2-furfural",
        "q": "How is photocatalytic and photoelectrocatalytic H2 evolution combined with valuable furfural production?",
        "papers": ["10.1016_j.apcata.2022.118987"],
    },
    {
        "id": "cat-dry-reforming-methane",
        "q": "How does CeO2 incorporation affect the efficient photothermochemical dry reforming of methane over Ni supported on ZrO2?",
        "papers": ["10.1016_j.cattod.2022.05.014"],
    },
    {
        "id": "cat-methylene-blue-papaya",
        "q": "Which carbon material derived from Carica papaya fruit juice shows high photocatalytic activity for the degradation of methylene blue in aqueous solution?",
        "papers": ["10.3390_catal13050886"],
    },
    {
        "id": "cat-water-gas-shift",
        "q": "How does the synergistic function of CeO2-x/CoO1-x/Co dual interfacial sites boost the reactivity of the water-gas shift reaction?",
        "papers": ["10.1038_s41467-023-42577-9"],
    },
    {
        "id": "cat-pdcu-formic-acid",
        "q": "How are branched PdCu nanoalloys synthesized in a bidirectionally controlled way for efficient and robust formic acid oxidation electrocatalysis?",
        "papers": ["10.1016_j.jcis.2021.05.018"],
    },
    # ── 2D materials / condensed matter ──────────────────────────────────
    {
        "id": "2d-graphene-edge-pinning",
        "q": "What causes the edge-pinning effect of graphene nanoflakes sliding atop graphene?",
        "papers": ["10.48550_arxiv.2311.12853"],
    },
    {
        "id": "2d-mxene-polarons",
        "q": "How do large Frohlich polarons contribute to band transport in MXenes?",
        "papers": ["10.1038_s41567-022-01541-y"],
    },
    {
        "id": "2d-tmdc-ultrashort-pulse",
        "q": "How do transition metal dichalcogenide monolayers respond to an ultrashort optical pulse, producing femtosecond currents and anisotropic electron dynamics?",
        "papers": ["10.1103_physrevb.103.155416"],
    },
    {
        "id": "2d-bafe2as2-mn",
        "q": "What incoherent electronic band states are observed in Mn-substituted BaFe2As2?",
        "papers": ["10.1103_physrevb.108.245124"],
    },
    {
        "id": "2d-irO2-spin-hall",
        "q": "What is the role of Dirac nodal lines and strain on the high spin Hall conductivity of epitaxial IrO2 thin films?",
        "papers": ["10.48550_arxiv.2006.04365"],
    },
    # ── nanoparticles / quantum dots / drug delivery ─────────────────────
    {
        "id": "nano-bodipy",
        "q": "What are the photophysical properties of halogenated tetraphenyl BODIPY dyes, computed from first principles?",
        "papers": ["10.1021_acs.jpcc.0c01742.s001"],
    },
    {
        "id": "nano-nanoplatelets-decay",
        "q": "How do the excitonic and biexcitonic decay rates in colloidal nanoplatelets depend on temperature?",
        "papers": ["10.1021_acs.jpclett.0c01628.s001"],
    },
    {
        "id": "nano-cdse-zno-qd",
        "q": "How can CdSe-ZnO core-shell quantum dots serve as a sensing platform for protein detection?",
        "papers": ["10.3390_nanomanufacturing1010002"],
    },
    {
        "id": "nano-gqd-drug-delivery",
        "q": "How are graphene quantum dots prepared by ball milling and applied for enhanced anti-cancer drug delivery?",
        "papers": ["10.1016_j.onano.2022.100072"],
    },
    {
        "id": "nano-magnetic-nanospheres",
        "q": "How do APTES monolayer coated self-assembled magnetic nanospheres enable controlled release of the anticancer drug Nintedanib?",
        "papers": ["10.1038_s41598-021-84770-0"],
    },
    {
        "id": "nano-nanoparticle-drug-delivery",
        "q": "How are engineered nanoparticles, such as graphene quantum dots and functionalized magnetic nanospheres, applied for anticancer drug delivery?",
        "papers": [
            "10.1016_j.onano.2022.100072",
            "10.1038_s41598-021-84770-0",
            "10.1021_acsomega.3c02260",
        ],
    },
    {
        "id": "nano-silver-alkanethiolate",
        "q": "How are microcrystalline silver n-alkanethiolates characterized by X-ray free electron laser serial femtosecond crystallography?",
        "papers": ["10.1021_jacs.3c02183"],
    },
    # ── metals / alloys ──────────────────────────────────────────────────
    {
        "id": "metal-fecral-annealing",
        "q": "How do recrystallization and texture evolve in a warm-pilgered FeCrAl alloy tube during annealing at 850 C?",
        "papers": ["10.1016_j.jnucmat.2022.153575"],
    },
    {
        "id": "metal-al-cu-mg-aging",
        "q": "How does aging treatment affect the evolution of the S-prime phase in a rapid cold punched Al-Cu-Mg alloy?",
        "papers": ["10.1016_s1003-6326_21_65627-3"],
    },
    {
        "id": "metal-hydrogen-pipeline-weld",
        "q": "What determines the hydrogen-assisted fracture resistance of pipeline welds in gaseous hydrogen?",
        "papers": ["10.1016_j.ijhydene.2020.11.239"],
    },
    {
        "id": "metal-nonwoven-mechanical",
        "q": "What is known about the mechanical behavior of nonwoven fabrics?",
        "papers": ["10.1177_1558925020970197"],
    },
    {
        "id": "metal-volcanic-sand-impact",
        "q": "What is the mechanical response of wet volcanic sand to impact loading, and how do water content and initial compaction affect it?",
        "papers": ["10.1007_s40870-020-00257-5"],
    },
    {
        "id": "metal-stainless-corrosion",
        "q": "How does stainless steel corrode under anoxic, highly saline and elevated temperature conditions?",
        "papers": ["10.5194_sand-2-39-2023"],
    },
    # ── polymers / composites / corrosion coatings ───────────────────────
    {
        "id": "pol-flame-retardant-pva",
        "q": "How do polyphosphazene hybridized perovskite copper hydroxystannate microspheres improve the flame retardant and mechanical properties of poly(vinyl alcohol) composites?",
        "papers": ["10.1002_vnl.22022"],
    },
    {
        "id": "pol-food-packaging-phenolic",
        "q": "What biodegradable active materials containing phenolic acids are used for food packaging applications?",
        "papers": ["10.1111_1541-4337.13011"],
    },
    {
        "id": "pol-pbs-biomaster",
        "q": "How do Biomaster-silver incorporated PBS and PBS/TPS films perform in terms of morphology, thermal properties, permeability and antimicrobial activity?",
        "papers": ["10.3390_polym13030391"],
    },
    {
        "id": "pol-pha-cell-free",
        "q": "Is cell-free synthesis a feasible platform for polyhydroxyalkanoate (PHA) production?",
        "papers": ["10.3390_polym15102333"],
    },
    {
        "id": "corrosion-sio2-go-coating",
        "q": "How do SiO2-GO nanofillers enhance the corrosion resistance of waterborne polyurethane acrylic coatings?",
        "papers": ["10.1177_2633366x20941524"],
    },
    {
        "id": "corrosion-composite-overview",
        "q": "What strategies improve the corrosion resistance of metal and coated surfaces, including nanocomposite coatings and stainless steel in aggressive environments?",
        "papers": ["10.1177_2633366x20941524", "10.5194_sand-2-39-2023"],
    },
    # ── optics / metasurfaces / photonics ────────────────────────────────
    {
        "id": "opt-phase-change-metasurface",
        "q": "What progress has been made in metasurfaces based on Ge-Sb-Te phase-change materials?",
        "papers": ["10.1063_5.0023925"],
    },
    {
        "id": "opt-chiroptical-metasurface",
        "q": "What are the principles, classifications and applications of chiroptical metasurfaces?",
        "papers": ["10.3390_s21134381"],
    },
    {
        "id": "opt-moire-bic",
        "q": "How are optical moire bound states in the continuum realized in one-dimensional photonic crystal slabs?",
        "papers": ["10.1038_s41467-024-53433-9"],
    },
    {
        "id": "opt-thz-graphene-plasmonics",
        "q": "How can a tunable terahertz photodetector be built using ferroelectric-integrated graphene plasmonics for a portable spectrometer?",
        "papers": ["10.48550_arxiv.2401.05780"],
    },
    {
        "id": "opt-nbn-snspd",
        "q": "What is the role of sputtered NbN films in ultrahigh performance superconducting nanowire single-photon detectors?",
        "papers": ["10.48550_arxiv.2311.17000"],
    },
    # ── electronic / semiconductor materials ─────────────────────────────
    {
        "id": "el-znO-defects-conductivity",
        "q": "How do paramagnetic donor-like defects contribute to the high n-type conductivity of hydrogenated ZnO microparticles?",
        "papers": ["10.1038_s41598-020-74449-3"],
    },
    {
        "id": "el-sno2-nanowire-transistor",
        "q": "How does xenon flash light irradiation control the threshold voltage of polyvinylpyrrolidone-coated SnO2 nanowire transistors?",
        "papers": ["10.1063_1.5139668"],
    },
    {
        "id": "el-bifeO3-magnetoelectric",
        "q": "How does nano-size affect the magnetostriction of BiFeO3 and the magnetoelectric coupling of BiFeO3-P(VDF-TrFE) composites?",
        "papers": ["10.48550_arxiv.2211.00952"],
    },
    {
        "id": "el-gan-multichannel",
        "q": "What are the prospects of multi-channel technology for the next generation of GaN power devices?",
        "papers": ["10.1063_5.0086978"],
    },
    # ── others ───────────────────────────────────────────────────────────
    {
        "id": "oth-thermal-conductive-film",
        "q": "How is a thermally conductive film fabricated using a perforated graphite sheet and UV-curable pressure-sensitive adhesive?",
        "papers": ["10.3390_nano11010093"],
    },
    {
        "id": "oth-microgroove-condensation",
        "q": "How does water condense on microgrooved silicon surfaces with hydrophilic, hydrophobic and biphilic coatings?",
        "papers": ["10.1021_acs.langmuir.3c02433"],
    },
    {
        "id": "oth-thermoelectric-solar",
        "q": "What is the performance of a hybrid thermoelectric generator and flat plate solar collector system in a semi-arid climate?",
        "papers": ["10.1016_j.csite.2023.102842"],
    },
]


def _paper_nodes(paper_dir: Path) -> list[dict[str, str]]:
    """Flatten a paper's tree.json into ``[{node_id, title, text}]``.

    Node text is resolved with the T3 indexer logic (:func:`collect_tree_nodes`,
    raw.md line ranges) when llama-index is available; otherwise titles only.
    """
    if not paper_dir.is_dir():
        return []
    if _LLAMA_INDEX_AVAILABLE:
        try:
            from drbrain.rag.indexer import collect_tree_nodes

            docs = collect_tree_nodes(paper_dir)
            return [
                {
                    "node_id": str(doc.metadata.get("node_id") or ""),
                    "title": str(doc.metadata.get("title") or ""),
                    "text": doc.text or "",
                }
                for doc in docs
            ]
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[rag] collect_tree_nodes failed for %s: %s", paper_dir, exc)
    # Fallback: flatten tree.json titles only.
    out: list[dict[str, str]] = []
    tree_path = paper_dir / "tree.json"
    if not tree_path.exists():
        return out
    try:
        tree = json.loads(tree_path.read_text(encoding="utf-8"))

        def _flatten(nodes: list[dict]) -> None:
            for node in nodes:
                out.append(
                    {
                        "node_id": str(node.get("node_id") or ""),
                        "title": str(node.get("title") or ""),
                        "text": "",
                    }
                )
                if isinstance(node.get("nodes"), list) and node["nodes"]:
                    _flatten(node["nodes"])

        _flatten(tree.get("structure", []))
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        log.warning("[rag] cannot parse tree.json at %s: %s", tree_path, exc)
    return out


#: First-section heading pattern: numbered ("1. Introduction"), roman-numeral
#: ("I. INTRODUCTION"), or a bare word heading (INTRODUCTION/ABSTRACT/SUMMARY).
#: The abstract text sits before it (title/authors → abstract → first section).
_HEADING_RE = re.compile(
    r"^\s*(?:\d+[\.\)]\s*\S|[IVX]+\.\s*\S|(?:INTRODUCTION|ABSTRACT|SUMMARY)\b)",
    re.IGNORECASE,
)
#: Signs that a paragraph is an author/affiliation block rather than prose.
_AUTHORISH_SIGNS = (
    "electronic mail",
    "received:",
    "dated:",
    "submitted:",
    "accepted:",
    "@",
    "\\*",
)


def _is_authorish(block: str) -> bool:
    b = block.strip().lower()
    return any(s in b for s in _AUTHORISH_SIGNS)


def _reference_paragraph(raw_text: str, min_chars: int = 120) -> str:
    """Best-effort abstract extraction from ``raw.md``.

    When a section heading exists, returns the longest prose paragraph before
    it (the abstract sits between the title/author block and the first
    section); otherwise returns the first long prose paragraph that is not an
    author/affiliation block (papers without any section heading, e.g. some
    arXiv manuscripts). Degrades to the longest paragraph overall.
    """
    long_blocks = [b.strip() for b in re.split(r"\n\s*\n", raw_text) if len(b.strip()) >= min_chars]
    if not long_blocks:
        return ""
    lines = raw_text.split("\n")
    heading_at = next((i for i, line in enumerate(lines) if _HEADING_RE.match(line)), None)
    if heading_at is not None:
        pre_blocks = [
            b.strip()
            for b in re.split(r"\n\s*\n", "\n".join(lines[:heading_at]))
            if len(b.strip()) >= min_chars
        ]
        if pre_blocks:
            return max(pre_blocks, key=len)
    for block in long_blocks:
        if not _is_authorish(block):
            return block
    return max(long_blocks, key=len)


def _is_content_title(title: str) -> bool:
    t = (title or "").strip().lower()
    return any(t.startswith(p) for p in _CONTENT_TITLE_PREFIXES)


def _relevant_nodes_for(papers_dir: Path, paper_id: str) -> tuple[list[dict[str, str]], str | None]:
    """Derive relevant nodes + a reference answer for one paper.

    Content nodes (titles matching :data:`_CONTENT_TITLE_PREFIXES`) are the
    semantically relevant ones; when a paper's tree carries no content node
    (back-matter only), *all* its nodes are treated as relevant (lenient
    fallback — node-level then degrades to paper-level, documented leniency).
    The reference answer is the abstract/summary node text (raw.md heuristic
    fallback), truncated.
    """
    nodes = _paper_nodes(papers_dir / paper_id)
    if not nodes:
        return [], None
    content = [n for n in nodes if _is_content_title(n["title"])]
    selected = content if content else nodes
    reference = ""
    for prefix in _ABSTRACT_TITLE_PREFIXES:
        for n in content:
            if n["title"].strip().lower().startswith(prefix):
                reference = n["text"]
                break
        if reference:
            break
    if not reference:
        raw_path = papers_dir / paper_id / "raw.md"
        if raw_path.exists():
            reference = _reference_paragraph(raw_path.read_text(encoding="utf-8"))
    reference = reference.strip()
    if reference:
        reference = reference[:_REFERENCE_MAX_CHARS].rstrip() + (
            "…" if len(reference) > _REFERENCE_MAX_CHARS else ""
        )
    return selected, reference or None


def _assign_splits(
    entries: list[dict[str, Any]], ratio: tuple[float, float, float] = DEFAULT_SPLIT_RATIO
) -> dict[str, str]:
    """Deterministic 60/20/20 dev/val/test assignment (seeded shuffle).

    A fixed seed keeps regeneration idempotent and avoids clustering related
    topics into a single split.
    """
    rng = random.Random(_SPLIT_SEED)
    order = list(range(len(entries)))
    rng.shuffle(order)
    n = len(order)
    n_dev = round(n * ratio[0])
    n_val = round(n * ratio[1])
    split_of: dict[int, str] = {}
    for idx, pos in enumerate(order):
        if pos < n_dev:
            split_of[idx] = "dev"
        elif pos < n_dev + n_val:
            split_of[idx] = "val"
        else:
            split_of[idx] = "test"
    return {entry["id"]: split_of[i] for i, entry in enumerate(entries)}


def build_golden_set(
    cfg: Config | dict[str, Any] | None = None,
    papers_dir: str | Path | None = None,
    force: bool = False,
    out_path: str | Path | None = None,
    query_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build (or verify) the golden set and write it to ``out_path``.

    Args:
        cfg: Config (``llamaindex.eval.golden_set`` decides the output path).
        papers_dir: Corpus root containing one directory per paper (defaults
            to ``cfg.dirs.papers``).
        force: Rebuild even when the golden file already exists.
        out_path: Override the output JSONL path.
        query_ids: Optional subset of curated query ids (integration tests use
            a small subset to keep LLM/index costs down).

    Idempotent: when the output file already exists and ``force`` is false,
    returns ``{"status": "exists", ...}`` without touching it. Writes
    atomically (tmp + replace). Relevant nodes are derived from each paper's
    tree.json (content nodes, all-nodes fallback); ``reference_answer`` is the
    abstract (see :func:`_relevant_nodes_for`).
    """
    li = get_llamaindex_config(cfg)
    if out_path is None:
        out_path = Path(li.eval.golden_set)
    out_path = Path(out_path)
    if not force and out_path.exists():
        return {"status": "exists", "path": str(out_path), "force": False}

    if papers_dir is None:
        dirs = getattr(cfg, "dirs", None)
        papers_dir = Path(getattr(dirs, "papers", ".") if dirs is not None else ".")
    papers_dir = Path(papers_dir)

    entries = list(_GOLDEN_QUERIES)
    if query_ids:
        wanted = set(query_ids)
        entries = [e for e in entries if e["id"] in wanted]
    if not entries:
        return {"status": "empty", "path": str(out_path), "queries": 0}

    split_of = _assign_splits(entries)
    missing: list[str] = []
    seen_papers: set[str] = set()
    lines: list[str] = []
    for entry in entries:
        qid, query = entry["id"], entry["q"]
        existing = [(papers_dir / p).is_dir() for p in entry["papers"]]
        relevant_papers = [p for p, ok in zip(entry["papers"], existing) if ok]
        missing.extend(p for p, ok in zip(entry["papers"], existing) if not ok)
        seen_papers.update(relevant_papers)
        relevant_nodes: list[dict[str, str]] = []
        reference = None
        for pid in relevant_papers:
            nodes, ref = _relevant_nodes_for(papers_dir, pid)
            relevant_nodes.extend({"paper_id": pid, "node_id": n["node_id"]} for n in nodes)
            if reference is None and ref:
                reference = ref
        record: dict[str, Any] = {
            "query": query,
            "relevant_papers": relevant_papers,
            "relevant_nodes": relevant_nodes,
            "split": split_of[qid],
        }
        if reference:
            record["reference_answer"] = reference
        lines.append(json.dumps(record, ensure_ascii=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(out_path)

    counts: dict[str, int] = {}
    for line in lines:
        split = json.loads(line).get("split", "")
        counts[split] = counts.get(split, 0) + 1
    return {
        "status": "ok",
        "path": str(out_path),
        "queries": len(lines),
        "papers": sorted(seen_papers),
        "splits": counts,
        "missing_papers": sorted(set(missing)),
        "force": force,
    }


# ── retriever evaluation (hit_rate / MRR, hand-computed) ────────────────────


def _node_identity(nws: Any) -> tuple[str, str]:
    """Return ``(paper_id, node_id)`` from a ``NodeWithScore`` metadata."""
    node = getattr(nws, "node", None)
    meta = dict(getattr(node, "metadata", None) or {}) if node is not None else {}
    pid = str(meta.get("paper_id") or "")
    nid = str(meta.get("node_id") or getattr(node, "node_id", None) or "")
    return pid, nid


def _rank_metrics(nodes: Sequence[Any], item: dict[str, Any], ks: Sequence[int]) -> dict[str, Any]:
    """Paper-level + node-level hit/mrr ranks for one golden query.

    Returns the first relevant rank per level (``None`` when nothing matched)
    plus per-``k`` hit/MRR values:

    * ``hit_rate@k`` (paper) = first relevant *paper* appears within top-k;
    * ``hit_rate@k`` (node) = first relevant *(paper_id, node_id)* within top-k;
    * ``mrr@k`` = 1/first-relevant-rank when the rank is within top-k, else 0.
    """
    rel_papers = {str(p) for p in item.get("relevant_papers") or []}
    rel_nodes = {
        (str(r.get("paper_id") or ""), str(r.get("node_id") or ""))
        for r in item.get("relevant_nodes") or []
    }
    paper_rank: int | None = None
    node_rank: int | None = None
    for i, nws in enumerate(nodes, start=1):
        pid, nid = _node_identity(nws)
        if paper_rank is None and pid in rel_papers:
            paper_rank = i
        if node_rank is None and rel_nodes and (pid, nid) in rel_nodes:
            node_rank = i
        if paper_rank is not None and (node_rank is not None or not rel_nodes):
            break
    ks_sorted = sorted(int(k) for k in ks)

    def _levels(first_rank: int | None) -> dict[str, Any]:
        hits: dict[str, bool] = {}
        mrrs: dict[str, float] = {}
        for k in ks_sorted:
            hits[str(k)] = first_rank is not None and first_rank <= k
            mrrs[str(k)] = (
                round(1.0 / first_rank, 6) if first_rank is not None and first_rank <= k else 0.0
            )
        return {"hit_rate": hits, "mrr": mrrs, "first_rank": first_rank}

    return {
        "query": item.get("query", ""),
        "paper": _levels(paper_rank),
        "node": _levels(node_rank),
    }


def _aggregate_rank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean hit_rate/MRR over per-query rows, for paper and node levels."""
    ks = sorted({int(k) for row in rows for k in row["paper"]["hit_rate"]})
    out: dict[str, Any] = {"queries": len(rows), "ks": ks, "hit_rate": {}, "mrr": {}}
    for level in ("paper", "node"):
        out["hit_rate"][level] = {
            str(k): round(sum(1 for r in rows if r[level]["hit_rate"].get(str(k))) / len(rows), 4)
            for k in ks
        }
        out["mrr"][level] = {
            str(k): round(sum(r[level]["mrr"].get(str(k), 0.0) for r in rows) / len(rows), 4)
            for k in ks
        }
    return out


def _coerce_cfg(cfg: Config | dict[str, Any]) -> Config:
    """Normalize a dict config to a typed :class:`Config`.

    Mirrors T6's dict/Config dual-form support: CLI tests pass plain dicts,
    while the CLI itself always passes a real :class:`Config`. Only the
    sections the RAG layer touches are mapped.
    """
    if isinstance(cfg, Config) or not isinstance(cfg, dict):
        return cfg
    from drbrain.config import (
        ApiConfig,
        DBConfig,
        DirsConfig,
        EmbedConfig,
        LlamaIndexConfig,
        LLMConfig,
    )

    return Config(
        llm=LLMConfig(**cfg.get("llm", {})),
        db=DBConfig(**cfg.get("db", {})),
        dirs=DirsConfig(**cfg.get("dirs", {})),
        api=ApiConfig(**cfg.get("api", {})),
        embed=EmbedConfig(**cfg.get("embed", {})),
        llamaindex=LlamaIndexConfig.from_dict(cfg.get("llamaindex", {})),
    )


def run_retriever_eval(
    cfg: Config | dict[str, Any],
    db: Any,
    split: str = "dev",
    ks: Sequence[int] = (5, 10),
    top_k: int | None = None,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """HitRate@K / MRR@K of the T4 fusion retriever over the golden ``split``.

    Retrieves each golden query once through ``build_hybrid_retriever``
    (BM25 + vector + configured custom legs) with ``top_k=max(ks)`` and scores
    paper-level and node-level relevance. Returns per-query rows plus the
    aggregated means. Status ``unavailable`` when no fusion retriever can be
    built (no index / llamaindex disabled); ``empty`` when the split has no
    golden queries.
    """
    cfg = _coerce_cfg(cfg)
    golden = load_golden(cfg, split=split)
    if not golden:
        return {"status": "empty", "split": split, "queries": 0}
    if max_queries:
        golden = golden[: int(max_queries)]

    from drbrain.rag.engine import build_hybrid_retriever

    k_max = max(int(k) for k in ks)
    fusion = build_hybrid_retriever(cfg, db, top_k=k_max)
    if fusion is None:
        return {
            "status": "unavailable",
            "split": split,
            "reason": "no fusion retriever (no index built or llamaindex disabled)",
        }

    rows: list[dict[str, Any]] = []
    for item in golden:
        try:
            nodes = fusion.retrieve(item["query"])[:k_max]
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[rag] retriever eval failed for query %.60r: %s", item["query"], exc)
            nodes = []
        rows.append(_rank_metrics(nodes, item, ks))

    agg = _aggregate_rank(rows)
    agg.update({"status": "ok", "split": split, "per_query": rows})
    return agg


# ── RAGAS-style generation metrics (self-written prompts) ───────────────────


def _prompt_faithfulness(question: str, answer: str, context: str) -> str:
    return (
        "You are an evaluation judge for a retrieval-augmented question answering system.\n"
        "Score the FAITHFULNESS of the Answer with respect to the Context.\n"
        "Faithfulness measures whether every factual claim in the answer is supported by "
        "the provided context (0.0 = fully unsupported/hallucinated, 1.0 = fully supported).\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Context:\n{context}\n\n"
        "Reply with exactly one line: SCORE: <number between 0 and 1>"
    )


def _prompt_answer_relevancy(question: str, answer: str) -> str:
    return (
        "You are an evaluation judge for a question answering system.\n"
        "Score the ANSWER RELEVANCY: how well the answer directly addresses the question, "
        "without being evasive or off-topic (0.0 = completely irrelevant, 1.0 = perfectly on-topic).\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Reply with exactly one line: SCORE: <number between 0 and 1>"
    )


def _prompt_context_precision(question: str, context: str) -> str:
    return (
        "You are an evaluation judge for a retrieval-augmented question answering system.\n"
        "Score the CONTEXT PRECISION: what fraction of the retrieved context is relevant to "
        "answering the question (0.0 = none of it is relevant, 1.0 = all of it is relevant).\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Reply with exactly one line: SCORE: <number between 0 and 1>"
    )


def _prompt_answer_correctness(question: str, answer: str, reference: str) -> str:
    return (
        "You are an evaluation judge for a question answering system.\n"
        "Score the ANSWER CORRECTNESS: how factually consistent the answer is with the "
        "reference answer from the source paper (0.0 = contradicts/ignores the reference, "
        "1.0 = fully consistent).\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Reference answer (source paper abstract):\n{reference}\n\n"
        "Reply with exactly one line: SCORE: <number between 0 and 1>"
    )


_SCORE_RE = re.compile(r"\bSCORE\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _parse_score(text: str | None) -> float | None:
    """Parse ``SCORE: 0.75`` (or a bare number) out of an LLM verdict."""
    if not text:
        return None
    m = _SCORE_RE.search(text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:  # pragma: no cover - defensive
            return None
    # Tolerate a bare numeric reply (some models skip the prefix).
    for token in text.split():
        try:
            value = float(token.strip(".,:[]()"))
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            continue
    return None


def _score_metric(llm: Any, prompt: str) -> float | None:
    """Run one scoring prompt through the DrbrainLLM bridge."""
    try:
        response = llm.complete(prompt, max_tokens=128)
        return _parse_score(getattr(response, "text", None))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("[rag] metric scoring call failed: %s", exc)
        return None


def _context_for(nodes: Sequence[Any], limit: int = 3) -> str:
    """Join top retrieved node texts for the context-based metrics."""
    chunks: list[str] = []
    for nws in (nodes or [])[:limit]:
        node = getattr(nws, "node", None)
        text = (getattr(node, "text", None) or "").strip()
        if not text:
            continue
        if len(text) > _CONTEXT_CHUNK_MAX_CHARS:
            text = text[:_CONTEXT_CHUNK_MAX_CHARS].rstrip() + "…"
        chunks.append(text)
    return "\n\n".join(chunks)


def _coerce_llm_cfg(cfg: Config | dict[str, Any]) -> Any:
    """Minimal object bearing ``llm.models`` for ``DrbrainLLM`` from a dict.

    Mirrors T6's dict/Config dual-form support: CLI tests pass plain dicts,
    while the CLI itself always passes a real :class:`Config`.
    """
    if not isinstance(cfg, dict):
        return cfg
    from types import SimpleNamespace

    return SimpleNamespace(
        llm=SimpleNamespace(models=list((cfg.get("llm") or {}).get("models") or [])),
        api=SimpleNamespace(cache_ttl=(cfg.get("api") or {}).get("cache_ttl") or 0),
        dirs=SimpleNamespace(cache=(cfg.get("dirs") or {}).get("cache", "data/cache")),
    )


def run_ragas_eval(
    cfg: Config | dict[str, Any],
    db: Any,
    split: str = "val",
    n: int = 10,
    top_k: int = 5,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """RAGAS-style 4-metric evaluation of the T5 ``ask_llamaindex`` output.

    For each of the first ``n`` golden queries of ``split`` (deterministic
    order), synthesizes an answer with
    :func:`~drbrain.rag.engine.ask_llamaindex` and scores it with four
    self-written prompt metrics through the ``DrbrainLLM`` bridge (drbrain
    fallback chain intact):

    * ``faithfulness`` — answer claims vs retrieved context;
    * ``answer_relevancy`` — answer vs question;
    * ``context_precision`` — retrieved context vs question;
    * ``answer_correctness`` — answer vs golden ``reference_answer`` (omitted
      when the golden entry carries none).

    Status ``unavailable`` when llamaindex cannot be used; ``empty`` when the
    split has no golden queries.
    """
    cfg = _coerce_cfg(cfg)
    golden = load_golden(cfg, split=split)
    if not golden:
        return {"status": "empty", "split": split, "queries": 0}
    if not _LLAMA_INDEX_AVAILABLE:
        return {
            "status": "unavailable",
            "split": split,
            "reason": "llama-index not installed",
        }
    if max_queries:
        golden = golden[: int(max_queries)]
    sample = golden[: int(n)]

    from drbrain.rag.engine import ask_llamaindex, build_hybrid_retriever
    from drbrain.rag.llm import DrbrainLLM

    llm = DrbrainLLM(_coerce_llm_cfg(cfg))
    fusion = build_hybrid_retriever(cfg, db, top_k=top_k)
    if fusion is None:
        return {
            "status": "unavailable",
            "split": split,
            "reason": "no fusion retriever (no index built or llamaindex disabled)",
        }

    rows: list[dict[str, Any]] = []
    for item in sample:
        question = item.get("query", "")
        try:
            result = ask_llamaindex(cfg, db, question, top_k=top_k, streaming=False)
            answer = str(result.get("answer") or "")
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[rag] ask_llamaindex failed for %.60r: %s", question, exc)
            answer = ""
        try:
            nodes = fusion.retrieve(question)[:top_k]
        except Exception:  # pragma: no cover - defensive
            nodes = []
        context = _context_for(nodes)
        reference = item.get("reference_answer") or ""
        row: dict[str, Any] = {
            "query": question,
            "answer_len": len(answer),
            "context_nodes": len(nodes),
            "faithfulness": _score_metric(llm, _prompt_faithfulness(question, answer, context)),
            "answer_relevancy": _score_metric(llm, _prompt_answer_relevancy(question, answer)),
            "context_precision": _score_metric(llm, _prompt_context_precision(question, context)),
            "answer_correctness": (
                _score_metric(llm, _prompt_answer_correctness(question, answer, reference))
                if reference
                else None
            ),
        }
        rows.append(row)

    metrics: dict[str, dict[str, Any]] = {}
    for key in ("faithfulness", "answer_relevancy", "context_precision", "answer_correctness"):
        values = [r[key] for r in rows if r[key] is not None]
        metrics[key] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "missing": sum(1 for r in rows if r[key] is None),
        }
    return {
        "status": "ok",
        "split": split,
        "queries": len(rows),
        "metrics": metrics,
        "per_query": rows,
    }


# ── baseline report ──────────────────────────────────────────────────────────


def format_eval_report(
    cfg: Config | dict[str, Any],
    retriever: dict[str, Any] | None = None,
    ragas: dict[str, Any] | None = None,
) -> str:
    """Render a markdown baseline report (timestamped) for the eval doc."""
    li = get_llamaindex_config(cfg)
    lines = [
        f"## LlamaIndex RAG 评估基线 — {datetime.now().isoformat(timespec='seconds')}",
        "",
        "### 配置",
        f"- golden_set: `{li.eval.golden_set}`;split 选项: {li.eval.split}",
        f"- enabled={li.enabled} · retrievers={li.retrievers} · fusion_mode={li.fusion_mode}"
        f" · rerank={li.rerank} · similarity_cutoff={li.similarity_cutoff}",
        f"- embed_model: `{getattr(getattr(cfg, 'embed', None), 'model', 'n/a')}`",
        "",
    ]

    if retriever is not None:
        lines.append("### Retriever eval(HitRate@K / MRR@K)")
        lines.append("")
        lines.append(
            f"- status: `{retriever.get('status')}`;split: `{retriever.get('split')}`"
            f";queries: {retriever.get('queries', 0)}"
        )
        if retriever.get("status") == "ok":
            lines.append("")
            lines.append(
                "| level | metric | " + " | ".join(f"K={k}" for k in retriever["ks"]) + " |"
            )
            lines.append("| --- | --- | " + " | ".join("---" for _ in retriever["ks"]) + " |")
            for level in ("paper", "node"):
                for metric in ("hit_rate", "mrr"):
                    vals = retriever[metric][level]
                    lines.append(
                        f"| {level} | {metric} | "
                        + " | ".join(str(vals[str(k)]) for k in retriever["ks"])
                        + " |"
                    )
        if retriever.get("reason"):
            lines.append("")
            lines.append(f"- reason: {retriever['reason']}")
        lines.append("")

    if ragas is not None:
        lines.append("### RAGAS-style eval(自写 4 指标 prompt)")
        lines.append("")
        lines.append(
            f"- status: `{ragas.get('status')}`;split: `{ragas.get('split')}`"
            f";queries: {ragas.get('queries', 0)}"
        )
        if ragas.get("status") == "ok":
            lines.append("")
            lines.append("| metric | mean | missing |")
            lines.append("| --- | --- | --- |")
            for key, info in ragas["metrics"].items():
                mean = info["mean"] if info["mean"] is not None else "n/a"
                lines.append(f"| {key} | {mean} | {info['missing']} |")
        if ragas.get("reason"):
            lines.append("")
            lines.append(f"- reason: {ragas['reason']}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Semantic similarity eval (zero-LLM, embedding cosine) ────────────────────


def run_semantic_eval(
    cfg: Config | dict[str, Any],
    db: Any,
    split: str = "val",
    n: int = 30,
    top_k: int = 5,
) -> dict[str, Any]:
    """Embedding-cosine similarity of synthesized answers vs golden references.

    Zero-LLM (per LlamaIndex ``SemanticSimilarityEvaluator`` semantics): the
    answer and the golden ``reference_answer`` are embedded through the
    configured drbrain embed provider (persistent 0.6B service when
    ``embed.provider=openai-compat``) and scored by cosine similarity. Cheap
    enough to run as a regression gate after every library merge.

    Requires golden entries with a ``reference_answer``; entries without one
    are skipped (counted as ``missing``).
    """
    cfg = _coerce_cfg(cfg)
    golden = load_golden(cfg, split=split)
    if not golden:
        return {"status": "empty", "split": split, "queries": 0}
    golden = [g for g in golden if g.get("reference_answer")][: int(n)]
    if not golden:
        return {"status": "empty", "split": split, "reason": "no reference_answer in golden"}

    from drbrain.rag.engine import ask_llamaindex
    from drbrain.services.embedding import _embed_batch

    embed_cfg = getattr(cfg, "embed", None)
    scores: list[float] = []
    missing = 0
    for item in golden:
        try:
            answer = ask_llamaindex(item["query"], cfg, db, top_k=top_k)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[rag] semantic eval ask failed for %.60r: %s", item["query"], exc)
            answer = None
        text = (answer or "").strip() if isinstance(answer, str) else ""
        if not text:
            missing += 1
            continue
        try:
            vecs = _embed_batch([text, str(item["reference_answer"])], embed_cfg)
            if not vecs or len(vecs) != 2:
                missing += 1
                continue
            import numpy as np

            a, b = np.asarray(vecs[0], dtype="float32"), np.asarray(vecs[1], dtype="float32")
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            scores.append(cos)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[rag] semantic eval embed failed: %s", exc)
            missing += 1

    if not scores:
        return {"status": "empty", "split": split, "reason": "no scorable answers", "missing": missing}
    mean = sum(scores) / len(scores)
    return {
        "status": "ok",
        "split": split,
        "queries": len(golden),
        "scored": len(scores),
        "missing": missing,
        "mean_similarity": round(mean, 4),
        "pass_rate": round(sum(1 for s in scores if s >= 0.8) / len(scores), 4),
        "threshold": 0.8,
    }


# ── QA pair generation (one-off LLM cost, reusable golden expansion) ─────────


def run_qagen(
    cfg: Config | dict[str, Any],
    n_nodes: int = 25,
    num_questions_per_chunk: int = 2,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Generate retrieval QA pairs from indexed nodes via LlamaIndex
    ``generate_question_context_pairs`` and merge them into the golden set.

    One-off LLM cost (tokenrouter/fallback chain); the generated pairs are
    appended to the golden JSONL as split ``generated`` so later retriever
    evals can use ``--split generated`` for a statistically thicker test.
    """
    cfg = _coerce_cfg(cfg)
    if not _LLAMA_INDEX_AVAILABLE:
        return {"status": "unavailable", "reason": "llama-index not installed"}

    from drbrain.rag.indexer import load_index

    index, _bm25 = load_index(cfg)
    if index is None:
        return {"status": "unavailable", "reason": "no vector index (run: drbrain rag index)"}

    nodes = index.docstore.docs.values() if hasattr(index.docstore, "docs") else []
    nodes = list(nodes)[: int(n_nodes)]
    if not nodes:
        return {"status": "empty", "reason": "no nodes in index docstore"}

    from llama_index.core.evaluation import generate_question_context_pairs

    models = list(getattr(cfg.llm, "models", []) or [])
    llm = None
    if models:
        from llama_index.llms.openai import OpenAI as LIOpenAI

        m = models[0]
        llm = LIOpenAI(
            model=str(m.get("model", "gpt-4o-mini")),
            api_key=str(m.get("api_key") or m.get("api_keys", ["sk-none"])[0] if isinstance(m.get("api_keys"), list) else m.get("api_key") or "sk-none"),
            api_base=str(m.get("base_url") or "https://api.openai.com/v1").replace("/v1", ""),
            temperature=0.1,
        )
    try:
        qa = generate_question_context_pairs(
            nodes, llm=llm, num_questions_per_chunk=int(num_questions_per_chunk)
        )
    except Exception as exc:
        return {"status": "error", "reason": f"generation failed: {exc}"}

    li = get_llamaindex_config(cfg)
    golden_path = Path(li.eval.golden_set)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with open(golden_path, "a", encoding="utf-8") as f:
        for q, ctx_ids in zip(qa.queries.values(), qa.relevant_docs.values()):
            if not str(q).strip():
                continue
            f.write(
                json.dumps(
                    {
                        "query": str(q).strip(),
                        "relevant_papers": [],
                        "relevant_nodes": [str(i) for i in ctx_ids],
                        "reference_answer": "",
                        "split": "generated",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            appended += 1
    return {
        "status": "ok",
        "generated": appended,
        "nodes_used": len(nodes),
        "golden_set": str(golden_path),
        "note": "eval with: drbrain rag eval --metrics retriever --split generated",
    }
