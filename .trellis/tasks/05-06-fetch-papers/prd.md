# drbrain fetch — PDF Acquisition from Open Access Sources

## What

Two new capabilities:

### `drbrain fetch <doi|title|arxiv_id>`
Single-paper fetch. Resolve identifier → find PDF → download → ingest.

### `drbrain citations <id> --fetch-interested`
Select interesting placeholder papers from citation results → batch fetch.

## PDF Acquisition Strategy (Multi-stage Fallback)

### Stage 1: OpenAlex OA (free, no auth needed)
- Query `work.best_oa_location` — if `is_oa: true` and `pdf_url` present, download directly
- OpenAlex has good OA coverage (~50% of recent papers)

### Stage 2: arXiv (free, no auth needed)
- If paper has `arxiv_id`, construct: `https://arxiv.org/pdf/{arxiv_id}.pdf`
- arXiv mirrors available as fallback

### Stage 3: Unpaywall (free, email-based auth)
- Query `https://api.unpaywall.org/v2/{doi}?email={config.email}`
- Returns `best_oa_location.url_for_pdf` for legal OA versions from institutional repositories, preprints, etc.

### Stage 4: Institutional Proxy (requires config)
User configures in `config.local.yaml`:
```yaml
fetch:
  institutional_proxy: "http://proxy.lib.university.edu:8080"
  proxy_type: "ezproxy"  # or "url_prefix"
```
For EZproxy: transforms `https://doi.org/10.xxx` → `https://doi-org.proxy.lib.university.edu/10.xxx`
For URL prefix: prepends proxy to all publisher URLs

### Stage 5: Direct DOI (free, no auth)
- Try `https://doi.org/{doi}` with `Accept: application/pdf` header
- Some publishers serve OA PDFs directly

## Configuration

```yaml
# config.yaml (checked in, defaults)
fetch:
  max_concurrent: 3
  timeout_per_fetch: 60
  user_agent: "DrBrain/0.1 (mailto:config.email)"
  fallback_order: ["openalex", "arxiv", "unpaywall", "doi_direct"]

# config.local.yaml (gitignored, secrets)
fetch:
  unpaywall_email: "user@example.com"
  institutional_proxy: ""  # optional
  proxy_type: ""           # ezproxy | url_prefix
```

## CLI Design

```bash
# Single fetch
drbrain fetch 10.1234/example.doi
drbrain fetch "Attention Is All You Need"
drbrain fetch --arxiv 1706.03762

# Batch from citations
drbrain citations p3f8a2 --type refs    # shows list
# interactive: select papers to fetch
drbrain citations p3f8a2 --fetch-interested
```

## Implementation

### New module: `src/drbrain/services/fetch.py`
- `fetch_paper(doi=None, title=None, arxiv_id=None) -> str | None` — returns local_id if success
- `resolve_pdf_url(doi, title, arxiv_id) -> str | None` — walk fallback stages
- `download_pdf(url, paper_dir) -> Path` — stream download with progress
- `fetch_from_citations(db, local_id, selected_refs) -> list[str]` — batch

### Integration
- `commands.py`: new `fetch_cmd`
- `commands.py`: add `--fetch-interested` to `citations_cmd`
- `config.py`: add `FetchConfig` dataclass

## Files
- `src/drbrain/services/fetch.py` — core fetch logic
- `src/drbrain/cli/commands.py` — `fetch_cmd`, `--fetch-interested` on citations
- `src/drbrain/config.py` — `FetchConfig`
- `config.yaml` — defaults
- `tests/test_fetch.py` — unit tests

## Acceptance
- `drbrain fetch <doi>` downloads PDF → ingest → returns paper_id
- `drbrain citations <id> --fetch-interested` interactive batch
- OpenAlex, arXiv, Unpaywall, direct DOI fallbacks all work
- Institutional proxy configurable
- No illegal sources (no Sci-Hub)
