# syntax=docker/dockerfile:1
# FastAPI backend image (Milestone 9's api/, orchestrating Milestones 3-8's pipeline).
#
# Scope note: installs core deps + the "embeddings" group (scikit-learn, for the default
# TfidfEmbedding backend actually exercised by every test in this repo). The vectorstores/
# retrieval/reranking/llm groups (Qdrant/Chroma/Weaviate/Pinecone clients, BM25, Cohere,
# Claude/OpenAI/Gemini SDKs) are correct, real code but need external services or API keys
# this default image doesn't assume are present -- add them with EXTRA_POETRY_GROUPS at
# build time (see docker-compose.yml) if your deployment's config selects those backends.

FROM python:3.12-slim AS builder

ENV POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
# hadolint DL3008 (pin apt versions) is deliberately not addressed: this sandbox has no live
# apt access to Debian's repos to look up real, currently-available version strings, and a
# guessed/invented pin would fail the build outright the first time it drifted from what's
# actually in the repo -- worse than an unpinned build-time-only dependency (discarded when
# the runtime stage below doesn't inherit from this builder stage).

RUN curl -sSL https://install.python-poetry.org -o /tmp/install-poetry.py \
    && python3 /tmp/install-poetry.py \
    && rm /tmp/install-poetry.py
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR /build
COPY pyproject.toml poetry.lock* ./

ARG EXTRA_POETRY_GROUPS=""
RUN poetry install --only main,embeddings${EXTRA_POETRY_GROUPS:+,${EXTRA_POETRY_GROUPS}} \
    --no-root --no-directory


FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app
ENV PATH="/build/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /build/.venv /build/.venv
COPY api/ ./api/
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
RUN mkdir -p documents

# Runtime data directories -- see api/state.py's RAG_*_DIR env vars (Milestone 9/11). Owned by
# the non-root user since the app writes to them (ingested docs, the FAISS index, feedback).
RUN mkdir -p data/processed data/raw data/embeddings data/metadata data/feedback \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]

COPY startup.sh ./
RUN chmod +x startup.sh
CMD ["./startup.sh"]