.PHONY: install format format-check lint typecheck test test-cov run-api run-ui clean

install:
	poetry install --with dev --with vectorstores --with evaluation --with ingestion --with embeddings --with retrieval --with reranking --with llm --with monitoring

format:
	poetry run black --line-length 100 src api frontend tests scripts
	poetry run isort --profile black --line-length 100 src api frontend tests scripts

format-check:
	poetry run black --check --line-length 100 src api frontend tests scripts
	poetry run isort --check-only --profile black --line-length 100 src api frontend tests scripts

lint: format-check
	poetry run ruff check src api frontend tests scripts --select E,F,I,UP,B,SIM

typecheck:
	poetry run mypy src api frontend

test:
	poetry run pytest

test-cov:
	poetry run pytest --cov=src --cov-report=html

generate-samples:
	poetry run python scripts/generate_sample_documents.py

# Usage: make ingest DOMAIN=hr INPUT=documents/hr
ingest:
	poetry run python -m src.ingestion.cli --input $(INPUT) --domain $(DOMAIN)

compare-chunking:
	poetry run python -m scripts.compare_chunking_strategies

build-index:
	poetry run python -m scripts.build_index_and_compare_retrieval

compare-reranking:
	poetry run python -m scripts.compare_reranking

run-conversational-demo:
	poetry run python -m scripts.run_conversational_pipeline_demo

run-evaluation:
	poetry run python -m scripts.run_evaluation_framework

run-api:
	poetry run uvicorn api.main:app --reload --port 8000

run-ui:
	poetry run streamlit run frontend/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
