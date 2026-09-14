# Vendored upstream sources (fixed revisions)

Two upstream research repositories are vendored as git submodules so the
unified tree RAG (`docs/unified-tree-rag-design.md`) can reuse reviewed code
at *fixed* revisions.  They are **not** dependencies of the DrBrain package:
nothing imports them at runtime except the narrow adapters in
`src/drbrain/tree/upstream.py`, and neither upstream `__init__` (with its
heavy optional imports) is executed.

| submodule | fixed commit | version marker | license |
| --- | --- | --- | --- |
| `vendor/pageindex` | `bfbd4b305cd3f0f39a5094779627a2c93634ad79` | v0.2.17 | MIT (`vendor/pageindex/LICENSE`) |
| `vendor/raptor` | `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767` | master | MIT (`vendor/raptor/LICENSE.txt`) |

Both revisions match the source audits in `docs/research/`; the submodule
gitlinks in the index pin them, and `tests/tree/test_vendor.py` verifies the
checked-out commits, the licenses, and the presence of the exact functions the
adapters load.  A checkout missing a submodule (or at a different revision)
fails with an actionable error instead of silently using another version.

Local clone note: on this machine HTTPS clones are slow, so the checked-out
`.git/config` overrides the submodule URL to SSH while `.gitmodules` keeps the
canonical HTTPS URL for portable checkouts.  That local override changes
neither the recorded commit nor the committed URL.

## Reuse boundary

Directly reused (narrow, verified by tests):

* `pageindex/page_index_md.py: md_to_tree` – Markdown structure extraction
  (transient structure hints; no persisted SDK document store).
* `pageindex/page_index_classic.py: page_index_main(doc, opt, page_list=...)`
  – PDF structure extraction over an injected real page list.
* `pageindex/utils.py` – text/JSON helpers, token counting, heading checks.
* `pageindex/tree_optimize.py: merge_tree` – cost-based structure optimisation
  (its continuous-page cost model is *not* reused as-is; the unified cost
  protocol lives in `docs/unified-tree-algorithms.md`).
* `raptor/cluster_utils.py: GMM_cluster`, `UMAP_cluster`,
  `RAPTOR_Clustering` – global/local UMAP + GMM posterior fitting.
* `raptor/tree_structures.py: Node/Tree` – plain data containers used by the
  adapters.

Adapted by DrBrain (new code, upstream semantics preserved and tested):

* Re-weighting of the *raw* two-stage posterior with structural affinity and
  per-stage thresholds (`src/drbrain/tree/posteriors.py`), which upstream does
  not expose.
* Parent-acceptance cost gate, summary cache keys, node identity/revisions,
  reachability and stop conditions (`src/drbrain/tree/cost.py`,
  `contracts.py`) – upstream has no equivalents.
* Narrow loaders that avoid executing the upstream package `__init__` and
  give actionable errors when optional dependencies are missing
  (`src/drbrain/tree/upstream.py`).

Deliberately **not** reused:

* `LocalAPI` / `DocStore` / SDK document directories and `tree.json` /
  `pages.json` persistence; the unified store keeps one canonical copy in
  `drbrain.db`.
* RAPTOR's tree pickling (`RetrievalAugmentation.save`) and its OpenAI
  wrappers; model access goes through the DrBrain `index_model` role.
* Upstream chunkers (RAPTOR `utils.split_text`, PageIndex flash layout
  heuristics) for canonical text: blocks come from DrBrain parsers with exact
  source ranges.
* The fixed native chat/QA loops; tree navigation is DrBrain's own
  retrieval-only state machine.
