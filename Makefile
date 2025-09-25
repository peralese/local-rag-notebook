.PHONY: env fmt lint test run clean

PY = python
PIP = python -m pip

env:
	$(PY) -m venv .venv
	. .venv/Scripts/activate || . .venv/bin/activate; \
	$(PIP) install --upgrade pip && \
	$(PIP) install -r requirements.txt && \
	$(PIP) install -r requirements-dev.txt

fmt:
	ruff check --fix .
	isort .
	black .

lint:
	ruff check .

test:
	pytest -q

run:
	$(PY) cli.py query "hello world" --synthesize --backend ollama --model llama3.1:8b --endpoint http://localhost:11435 --show-contexts

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info

