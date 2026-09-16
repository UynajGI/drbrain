.DEFAULT_GOAL := help

UV ?= uv
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff
MYPY ?= $(UV) run mypy
PRE_COMMIT ?= $(UV) run pre-commit

.PHONY: help sync install-hooks format format-check lint typecheck test test-unit security check ci

help: ## Show available developer commands
	@awk 'BEGIN {FS = ":.*##"; print "Usage: make <target>\n"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install locked dependencies and the editable package
	$(UV) sync
	$(UV) pip install -e .

install-hooks: ## Install Lefthook and pre-commit hooks for this worktree
	@command -v lefthook >/dev/null 2>&1 || { echo "lefthook is required; install it with your package manager." >&2; exit 1; }
	lefthook install
	$(PRE_COMMIT) install

format: ## Format Python sources and tests in place
	$(RUFF) format src/ tests/

format-check: ## Check formatting without modifying files
	$(RUFF) format --check src/ tests/

lint: ## Run Ruff lint checks
	$(RUFF) check src/ tests/

typecheck: ## Run mypy over the application package
	$(MYPY) src/drbrain

# Tests are hermetic: a DRBRAIN_ROOT inherited from the shell would point the
# suite at an unrelated runtime root (mass failures).  The Makefile unsets it
# for every pytest invocation, matching the documented test recipe.
test: ## Run the complete test suite
	env -u DRBRAIN_ROOT -u DRBRAIN_RUNTIME_ROOT $(PYTEST) -q

test-unit: ## Run tests excluding external integrations
	env -u DRBRAIN_ROOT -u DRBRAIN_RUNTIME_ROOT $(PYTEST) -m "not integration" --timeout=30 -q

security: ## Run dependency and repository secret checks
	$(UV) run pip-audit --desc
	gitleaks detect --source . --config .gitleaks.toml --redact

check: format-check lint typecheck test-unit ## Run the local pre-PR verification suite

ci: check ## Alias for the complete local CI-equivalent check
