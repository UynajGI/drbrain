# PyPI Publish Setup

## What
- Verify pyproject.toml build metadata
- Add `MANIFEST.in` to include templates in sdist
- Add `.github/workflows/publish.yml` — auto-publish to PyPI on version tag push
- Update README Quick Start to show `pip install drbrain` as primary

## Files
- `pyproject.toml` — add classifiers, urls, keywords
- `MANIFEST.in` — include `src/drbrain/templates/`
- `.github/workflows/publish.yml` — trusted publishing via GitHub Actions
- `README.md` — pip install first, git clone second

## Acceptance
- `uv build` produces a valid wheel
- README shows pip install as primary install method
