# Makefile for local-rag-notebook
# Usage examples:
#  - make env
#  - make fmt
#  - make lint
#  - make test
#  - make run ARGS='query "hello" --synthesize --backend ollama --model llama3.1:8b --endpoint http://localhost:11435'

SHELL := /bin/sh
VENV := .venv

# OS-aware Python path inside venv
ifeq ($(OS),Windows_NT)
PY := $(VENV)/Scripts/python.exe
PIP := $(PY) -m pip
ACTIVATE := .venv\Scripts\Activate.ps1
NULL := nul
else
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
ACTIVATE := . .venv/bin/activate
NULL := /dev/null
endif

# Default target
.PHONY: all
all: env

# Create virtual env and install deps (runtime + dev)
.PHONY: env
env:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	@if [ -f requirements.txt ]; then \
		echo "Installing runtime requirements..."; \
		$(PIP) install -r requirements.txt; \
	else \
		echo "requirements.txt not found; skipping runtime deps"; \
	fi
	@echo "Installing dev requirements..."
	$(PIP) install -r requirements-dev.txt

# Run the CLI with ARGS='...'
.PHONY: run
run:
	@echo "Running: cli.py $(ARGS)"
	$(PY) cli.py $(ARGS)

# Code quality
.PHONY: fmt
fmt:
	$(PY) -m isort .
	$(PY) -m black .

.PHONY: lint
lint:
	$(PY) -m ruff check .

.PHONY: test
test:
	$(PY) -m pytest -q

# Remove caches (keeps venv)
.PHONY: clean
clean:
	@echo "Cleaning caches..."
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>$(NULL) || true
	@rm -rf .pytest_cache 2>$(NULL) || true
	@rm -rf .ruff_cache 2>$(NULL) || true
