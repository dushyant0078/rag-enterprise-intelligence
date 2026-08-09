#!/usr/bin/env bash
# Runs on every container start (Render free tier wipes disk between restarts/deploys,
# so this makes the demo self-sufficient rather than assuming persisted data exists).
set -euo pipefail

echo "==> Generating sample documents (if not already present)..."
python -m scripts.generate_sample_documents || true

echo "==> Ingesting sample documents into every domain..."
for domain_dir in documents/*/; do
  domain=$(basename "$domain_dir")
  python -m src.ingestion.cli --input "$domain_dir" --domain "$domain" || true
done

echo "==> Starting API server on port ${PORT:-8000}..."
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"