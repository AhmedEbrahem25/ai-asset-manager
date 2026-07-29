# Development tasks.
#
# Every recipe clears PYTHONPATH first. A system-wide PYTHONPATH pointing at another
# interpreter's site-packages silently shadows the virtual environment, which presents as
# imports resolving to the wrong versions of everything; clearing it per-recipe is the
# reliable fix and costs nothing on machines where it was never set.

PYTHON ?= .venv/Scripts/python.exe
export PYTHONPATH :=

.DEFAULT_GOAL := help
.PHONY: help install test test-all lint typecheck check exe exe-onefile dist run clean

help:  ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the project and its dev dependencies
	uv sync --extra dev

test:  ## Run the test suite (skips the opt-in real-cache tests)
	$(PYTHON) -m pytest -q -m "not real_cache"

test-all:  ## Run everything, including tests against the real HuggingFace cache
	$(PYTHON) -m pytest -q

lint:  ## Check formatting and lint rules
	$(PYTHON) -m ruff check .

typecheck:  ## Run mypy in strict mode
	$(PYTHON) -m mypy ai_asset_manager

check: lint typecheck test  ## Everything CI would run

exe:  ## Build the standalone executable into dist/aam/
	$(PYTHON) scripts/build_exe.py --clean

exe-onefile:  ## Build a single-file executable (starts ~1s slower)
	$(PYTHON) scripts/build_exe.py --clean --onefile

dist: ## Build the executable and a distributable zip
	$(PYTHON) scripts/build_exe.py --clean --zip

run:  ## Scan the caches on this machine, then show the inventory
	$(PYTHON) -m ai_asset_manager.cli scan --auto
	$(PYTHON) -m ai_asset_manager.cli inventory

clean:  ## Remove build artefacts and caches
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
