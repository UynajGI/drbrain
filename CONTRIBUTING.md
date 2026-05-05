# Contributing to DrBrain

Thanks for your interest in contributing! This document explains how to get involved.

## Development Setup

```bash
# Clone and install
git clone https://github.com/UynajGI/DrBrain.git
cd DrBrain
uv sync
uv pip install -e .

# Install pre-commit hooks
pre-commit install

# Run fast tests (skip integration)
uv run pytest -m "not integration"

# Run all tests
uv run pytest
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Make your changes
3. Ensure all checks pass:
   ```bash
   uv run ruff check .                  # lint
   uv run ruff format --check .         # format check
   uv run pytest -m "not integration"   # fast tests
   uv run pytest                        # full test suite
   ```
4. Submit a PR with a clear description

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `test:` — adding or updating tests
- `chore:` — maintenance (CI, deps, config)

### Test Guidelines

- Test **behavior contracts**, not implementation details
- A refactor should not break tests — if it does, the test was too coupled
- Use pytest fixtures for isolation
- Mark slow tests (network, LLM) with `@pytest.mark.integration`

## Code Style

- **Linter/formatter**: ruff (configured in `pyproject.toml`)
- **Type hints**: encouraged
- **Docstrings**: Google-style for public API functions
- **Code comments**: English, only when logic isn't self-evident

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `drbrain/` | Python package (library + CLI) |
| `tests/` | Test suite |
| `data/` | User paper library (not tracked) |
| `workspace/` | User workspace outputs (not tracked) |
| `docs/` | Documentation |

## Reporting Issues

- **Bugs**: open a GitHub issue with steps to reproduce
- **Features**: open a GitHub issue describing the use case
- **Security**: see [SECURITY.md](SECURITY.md) — do **not** open a public issue

## Questions?

Open a [discussion](https://github.com/UynajGI/DrBrain/discussions) or file an issue.
