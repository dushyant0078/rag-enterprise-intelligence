# syntax=docker/dockerfile:1
# Streamlit dashboard image (Milestone 10's frontend/). Deliberately minimal: the frontend
# is a pure HTTP client of the API image (frontend/api_client.py) -- it needs streamlit and
# requests, nothing from src/ or the API's heavier ML dependencies at all.

FROM python:3.14-slim AS builder

ENV POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*
# hadolint DL3008 not addressed here either -- see docker/api.Dockerfile's comment on the
# same trade-off (no live apt access in this build environment to pin real version strings).

RUN curl -sSL https://install.python-poetry.org -o /tmp/install-poetry.py \
    && python3 /tmp/install-poetry.py \
    && rm /tmp/install-poetry.py
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR /build
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root --no-directory


FROM python:3.14-slim AS runtime

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app
ENV PATH="/build/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /build/.venv /build/.venv
COPY frontend/ ./frontend/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
