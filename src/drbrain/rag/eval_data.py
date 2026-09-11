"""Golden-set construction, loading and atomic evaluation artifacts."""
from __future__ import annotations
import json
import logging
import os
import random
import re
import tempfile
from collections.abc import Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from drbrain.config import Config
from drbrain.rag.config import get_llamaindex_config
from drbrain.security import redact_sensitive_text
from drbrain.storage.paths import raw_md_path, resolve_paper_dir, tree_json_path, writable_artifact_path
log = logging.getLogger(__name__)
_LLAMA_INDEX_AVAILABLE = find_spec("llama_index") is not None
DEFAULT_SPLIT_RATIO = (0.6, 0.2, 0.2)
_SPLIT_SEED = 20260812
_REFERENCE_MAX_CHARS = 800
_CONTENT_TITLE_PREFIXES = ("abstract", "summary", "overview", "introduction", "results",
                          "discussion", "conclusion", "experimental", "methods", "materials", "section ")
_ABSTRACT_TITLE_PREFIXES = ("abstract", "summary")


def _runtime_selected() -> bool:
    """Return whether an invocation explicitly selected a runtime root."""
    return "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ

def _ensure_eval_parent(path: Path) -> Path:
    """Create an evaluation-output parent without following symlinks."""
    parent = path.parent
    for ancestor in (parent, *parent.parents):
        if ancestor.is_symlink():
            raise ValueError(f"evaluation output directory contains a symlink: {ancestor}")
    current = parent
    missing: list[Path] = []
    while not current.exists():
        missing.append(current)
        next_parent = current.parent
        if next_parent == current:
            break
        current = next_parent
    if current.is_symlink() or not current.is_dir():
        raise ValueError(f"evaluation output directory is not a real directory: {current}")
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"evaluation output directory is not a real directory: {directory}")
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"evaluation output directory is not a real directory: {parent}")
    return parent

def _safe_eval_output(path: str | Path) -> Path:
    """Resolve an evaluation output under the selected runtime, if any."""
    candidate = Path(path).expanduser()
    if _runtime_selected():
        from drbrain.runtime import RuntimeContext

        candidate = RuntimeContext.create().assert_within_root(candidate, label="evaluation output")
    if not candidate.name or candidate.name in {".", ".."}:
        raise ValueError("evaluation output must be a regular file path")
    if candidate.exists() and not candidate.is_file():
        raise ValueError(f"evaluation output is not a regular file: {candidate}")
    parent = _ensure_eval_parent(candidate)
    return writable_artifact_path(parent, candidate.name)

def _write_text_atomically(path: Path, content: str) -> None:
    """Atomically replace ``path`` through same-directory staging.

    ``os.replace`` is atomic on the local filesystem. Keeping the temporary
    file beside its destination avoids cross-volume moves, so readers never
    observe a partially rewritten baseline or golden set.
    """
    path = _safe_eval_output(path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            fd = -1  # ownership moved to the context manager
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)
        raise

def _append_text_atomically(path: Path, content: str) -> None:
    """Append text through an atomic replace, retaining the old file on error."""
    path = _safe_eval_output(path)
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    _write_text_atomically(path, previous + content)


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
    try:
        tree_path = tree_json_path(paper_dir)
    except (OSError, TypeError, ValueError) as exc:
        log.warning("[rag] unsafe tree.json path at %s: %s", paper_dir, exc)
        return out
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
    paper_path = resolve_paper_dir(papers_dir, paper_id)
    if paper_path is None:
        return [], None
    nodes = _paper_nodes(paper_path)
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
        try:
            raw_path = raw_md_path(paper_path)
        except (OSError, TypeError, ValueError) as exc:
            log.warning("[rag] unsafe raw.md path at %s: %s", paper_path, exc)
            raw_path = None
        if raw_path is not None and raw_path.is_file():
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
        existing = [resolve_paper_dir(papers_dir, p) is not None for p in entry["papers"]]
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
