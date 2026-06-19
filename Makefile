# Hide & Seek 2.0 — developer task runner.
#
# Usage:  make <target>
# Run from the repository root. Targets set PYTHONPATH=. so that absolute
# imports (e.g. `from config import Config`) resolve.
#
# NOTE: on Windows, run these under Git Bash / WSL (GNU make + a POSIX shell),
# or invoke the underlying commands directly in PowerShell, e.g.
#   $env:PYTHONPATH="."; python examples/quickstart.py

PYTHON  ?= python
PIP     ?= pip
export PYTHONPATH := .

.DEFAULT_GOAL := help
.PHONY: help install test smoke train lint format demo viz

help: ## Show this help.
	@echo "Hide & Seek 2.0 — available targets:"
	@echo "  install   Install runtime + dev dependencies"
	@echo "  test      Run the pytest suite"
	@echo "  smoke     Run the quickstart demo (examples/quickstart.py)"
	@echo "  train     Launch training (train.py)"
	@echo "  lint      Lint with ruff"
	@echo "  format    Auto-format with black + ruff --fix"
	@echo "  demo      Train the self-play AI + export learned trajectories"
	@echo "  viz       Serve the 3D viewer at http://localhost:8000 (runs demo first)"

install: ## Install runtime and dev dependencies.
	# requirements.txt exists for CI/Docker reproducibility (pinned runtime set);
	# for a dev checkout, the editable package + [dev] extra is the single source.
	$(PIP) install -e ".[dev]"

test: ## Run the pytest suite.
	$(PYTHON) -m pytest

smoke: ## Quick end-to-end sanity check via the quickstart demo.
	$(PYTHON) examples/quickstart.py

train: ## Launch training with the default config.
	$(PYTHON) train.py

lint: ## Static lint check (no changes written).
	$(PYTHON) -m ruff check .

format: ## Auto-format the codebase.
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check --fix .

# --- 3D replay viewer ------------------------------------------------------
# `demo` trains the CPU self-play learner (needs numpy, no GPU) and exports the
# LEARNED behaviour + measured curve into the viewer; `viz` serves viz/web with
# the stdlib http.server. PowerShell equivalents:
#   demo:  python -m learn.export_viewer
#   viz:   python -m learn.export_viewer ; python -m viz.serve
# The viewer must be opened over http:// (NOT file://) — it uses ES modules.

demo: ## Train the self-play AI + export learned trajectories and the measured curve.
	$(PYTHON) -m learn.export_viewer

viz: demo ## Serve the 3D viewer (viz/web) at http://localhost:8000 — runs `demo` first.
	$(PYTHON) -m viz.serve
