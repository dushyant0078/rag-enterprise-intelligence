# RAG Enterprise Intelligence

<!-- Replace OWNER with the real GitHub org/user this repo is pushed to -- these badge URLs
     are correct GitHub Actions badge syntax but non-functional until then, same as
     k8s/ingress.yaml's placeholder domain and .github/CODEOWNERS' placeholder owner. -->
[![CI](https://github.com/OWNER/RAG-Enterprise-Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/RAG-Enterprise-Intelligence/actions/workflows/ci.yml)
[![CodeQL](https://github.com/OWNER/RAG-Enterprise-Intelligence/actions/workflows/codeql.yml/badge.svg)](https://github.com/OWNER/RAG-Enterprise-Intelligence/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LLM-powered RAG Enterprise Document Intelligence System** — a citation-aware,
evaluation-driven Retrieval-Augmented Generation platform for regulated,
document-heavy enterprises (banking, insurance, legal, HR, technical ops).

> **Status:** All 14 milestones complete — business understanding,
> enterprise architecture, repository scaffold, document ingestion, chunking
> + embeddings, vector database + retrieval, reranking + prompt engineering,
> a full conversational RAG pipeline, a comprehensive evaluation framework, a
> production FastAPI backend, a 10-page Streamlit dashboard, real
> monitoring & observability, a Docker/Kubernetes deployment set, a full
> GitHub CI/CD setup, and a final benchmark report. Built
> milestone-by-milestone with real execution and honest verification at
> every step — see [`reports/final_benchmark_report.md`](reports/final_benchmark_report.md)
> for the numbers and [`docs/`](docs/) for each milestone's full writeup.

## What this is

Every enterprise sits on document collections too large to read manually and
too high-stakes for an LLM that hallucinates. This platform answers questions
over multi-format enterprise document collections (PDF, DOCX, PPTX, HTML,
Markdown, TXT, CSV, scanned/OCR) with:

- Hybrid dense + sparse retrieval, reranked before it ever reaches the LLM
- A dedicated Response Validator stage that checks generated claims against
  retrieved passages and returns "insufficient evidence" rather than guessing
- Citations down to document, page, and chunk ID with a confidence score
- RBAC-aware retrieval so no query surfaces a document its user isn't
  entitled to
- A swappable-by-config backend for embeddings, vector store, reranker, and
  LLM provider — no code changes to move from FAISS to Qdrant, or GPT to
  Claude

Full framing — personas, business problem, security/compliance/hallucination
risk analysis, and retrieval challenges — is in
[`docs/01_business_understanding.md`](docs/01_business_understanding.md).

## Quickstart

```bash
git clone <this-repo> && cd RAG-Enterprise-Intelligence
make install                                       # poetry install, all optional groups
make generate-samples                               # sample documents into documents/
make ingest DOMAIN=hr INPUT=documents/hr            # repeat per domain
make run-api                                        # terminal 1 — http://localhost:8000/docs
make run-ui                                         # terminal 2 — http://localhost:8501
```

Works out of the box with no API keys: the default configuration is
TF-IDF embeddings + an embedded FAISS index + a fully offline LLM stand-in
(`DeterministicTestLLM`) — the same stack every test in this repo runs
against. Point it at a real embedding model, vector store, or LLM by
setting the matching environment variables (see `.env.example`) — no code
changes required. Prefer containers? `docker compose up --build` does the
same two services (see `docs/12_docker_kubernetes_deployment.md`).

## Benchmark highlights

Full numbers, methodology, and an honest reading of what they do and don't
show: [`reports/final_benchmark_report.md`](reports/final_benchmark_report.md).

- **480 tests passing, 86% coverage** — including 24 real end-to-end
  frontend tests against a live backend and 63 real HTTP API tests
- **MRR 1.0** on the evaluation framework's labeled benchmark — the
  correct document was the top retrieval hit on every query
- **112.8 req/s** measured throughput, **4.9ms** mean single-request
  latency — a real load test against a genuinely running server, run
  fresh for the final report
- A real token-bucket rate limiter verified under a real burst: **exactly
  60 of 70** rapid requests succeed, `429` begins precisely at request #61
- **18 real bugs found and fixed** across development, every one caught
  by actually running code against real data rather than by inspection,
  each with a named regression test — see the report's full retrospective

## Architecture

Full architecture, component responsibilities, sequence/deployment/data-flow
diagrams, and key design trade-offs are in
[`docs/02_architecture.md`](docs/02_architecture.md).

```
User → Chat UI → FastAPI Gateway → Auth → Conversation Manager → Query Rewriter
     → Embedding Service → Vector DB → Retriever → Reranker → Context Builder
     → LLM Inference Engine → Citation Generator → Response Validator → User
```
(Every arrow above is traced end-to-end through real OpenTelemetry spans
(Milestone 11) — see the diagrams for the full picture, including the
deployment topology.)

## Repository structure

```
RAG-Enterprise-Intelligence/
├── data/                 # raw / processed / embeddings / metadata (gitignored contents)
├── documents/             # sample source documents for ingestion demos
├── notebooks/             # exploratory only — nothing here is a src/ dependency
├── src/
│   ├── ingestion/          parsing/           chunking/
│   ├── embeddings/         retrieval/         reranking/
│   ├── prompting/          llm/               memory/
│   ├── citations/          evaluation/        monitoring/
│   ├── security/           utils/
├── api/                   # FastAPI gateway (Milestone 9) ✅
├── frontend/               # Streamlit UI (Milestone 10) ✅
├── configs/                # reference config YAML per swappable backend (design docs, not
│                            # live-loaded — see "Configuration" below)
├── tests/                  # mirrors src/ 1:1
├── scripts/                # operational scripts, added per milestone
├── docker/                 # Dockerfiles + compose-spec schema (Milestone 12) ✅
├── k8s/                    # Kubernetes manifests, kustomize-ready (Milestone 12) ✅
├── docs/                   # design docs, one set per milestone
├── reports/                # benchmark & evaluation output (Milestone 8, 14)
└── .github/workflows/       # CI/CD (Milestone 13)
```

Every `src/` subpackage carries its own responsibility docstring in its
`__init__.py`; `tests/` mirrors that structure 1:1 so retrieval tests live
next to `src/retrieval`, etc.

## Configuration

`configs/` holds one reference YAML file per swappable backend (embedding
model, vector store, retrieval strategy, reranker, LLM provider) —
documentation of what's available and its parameters, written in
Milestone 2. Swapping backends in the actually-built code happens two
ways, not through a live config-loading system reading these files
directly: pass the desired class as a constructor argument (see
`src/memory/pipeline_factory.py`, or any of the `scripts/*.py` demos), or,
for the API (Milestone 9), through environment variables — see
`.env.example` and `docs/09_fastapi_backend.md`. Secrets are never
committed — copy `.env.example` to `.env` locally.

## Development

```bash
make install       # poetry install, all optional dependency groups
make format         # black + isort (mutates files)
make lint           # format-check (non-mutating) + ruff -- what CI actually runs
make typecheck      # mypy across src/, api/, frontend/
make test           # pytest, full suite with coverage
make run-api         # http://localhost:8000/docs
make run-ui           # http://localhost:8501
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full development workflow,
including a real, honestly-documented limitation around `poetry install`'s
completion time in constrained sandboxes and a `pip`-based fallback.

### Ingestion (Milestone 3)

```bash
make generate-samples              # writes sample docs into documents/ (see docs/03_ingestion_pipeline.md)
make ingest DOMAIN=hr INPUT=documents/hr
# equivalently:
poetry run python -m src.ingestion.cli --input documents/hr --domain hr
```

Supports PDF (native + scanned/OCR), DOCX, PPTX, HTML, Markdown, CSV, and TXT.
Re-running is incremental (unchanged files are skipped), duplicate content is
detected across paths, and edited files get a new tracked version — see
`docs/03_ingestion_pipeline.md` for what was actually verified end-to-end.

### Chunking & embeddings (Milestone 4)

```bash
poetry run python -m scripts.compare_chunking_strategies
```

7 chunking strategies (`fixed_size`, `recursive`, `sentence_aware`,
`markdown_aware`, `table_aware`, `semantic`, `hybrid` — `src/chunking/`) and
an embedding layer (`src/embeddings/`) supporting 5 production models (BGE
Large, E5 Large, MiniLM, Instructor XL, GTE Large, via
`SentenceTransformerEmbedding`) plus an offline TF-IDF backend for air-gapped
deployments. Content-hash caching, and both sync and async batch embedding.
See `docs/04_chunking_and_embeddings.md` for the real chunking-strategy
comparison this produces against the sample corpus.

### Vector store & retrieval (Milestone 5)

```bash
poetry run python -m scripts.build_index_and_compare_retrieval
```

3 vector stores run fully locally with no server or network access (FAISS,
ChromaDB, Qdrant — `src/retrieval/vectorstore/`), plus correct, complete
client code for Weaviate and Pinecone (need a reachable instance / cloud
account respectively). All 10 retrieval strategies from the spec live in
`src/retrieval/`: dense, BM25 sparse, hybrid, ensemble (real RRF), multi-query,
self-query, parent-child, contextual compression, time-aware, and
metadata-aware (RBAC enforcement point). See
`docs/05_vectorstore_and_retrieval.md` for what's real vs. untestable here,
and the retrieval-strategy comparison this produces.

### Reranking & prompting (Milestone 6)

```bash
poetry run python -m scripts.compare_reranking
```

Reranking (`src/reranking/`): `LexicalOverlapReranker` runs fully offline
and is what's tested end-to-end here; `CrossEncoderReranker` (BGE reranker /
MS-MARCO) and `CohereReranker` are correct, complete code needing Hugging
Face Hub / Cohere API access respectively. Prompting (`src/prompting/`): a
`PromptTemplate` schema, a token-budgeted `ContextBuilder`, reusable
guardrail snippets tied to Milestone 1's named risks, and all 8 required
templates (QA, summarization, contract analysis, compliance checking, risk
extraction, financial analysis, technical troubleshooting, research
assistance). See `docs/06_reranking_and_prompting.md` for a real diagnosed
example of what reranking fixed on the sample corpus, and why.

### Conversational RAG pipeline (Milestone 7)

```bash
poetry run python -m scripts.run_conversational_pipeline_demo
```

`ConversationalRAGPipeline` (`src/memory/rag_pipeline.py`) runs the full
stage sequence end to end: Conversation Manager → Query Rewriter →
Retriever → Reranker → Context Builder → LLM → Citation Generator →
Response Validator. LLM backends (`src/llm/`) cover Claude (verified
against the *live* Anthropic API — genuinely reachable from this sandbox),
GPT/Gemini/Llama3/Mistral/Qwen/DeepSeek (correct, network-blocked here),
and a fully offline `DeterministicTestLLM` for real pipeline testing.
Memory (`src/memory/`) covers session state with enforced user isolation,
context-window management, summarization, long-term memory, and query
rewriting. Citations (`src/citations/`) enforce grounding for real — a
fabricated citation or unsupported claim gets the answer replaced with the
insufficient-evidence fallback, not just flagged. See
`docs/07_conversational_rag_pipeline.md` for four real bugs this milestone
found and fixed while building the test harness.

### Evaluation framework (Milestone 8)

```bash
poetry run python -m scripts.run_evaluation_framework
```

`RAGEvaluator` (`src/evaluation/`) runs the labeled 7-query benchmark
through the real pipeline, scoring every metric in the spec: Precision@K,
Recall@K, MRR, nDCG (exact IR formulas, no LLM), Citation Accuracy (exact),
and lexical-proxy Hallucination Rate/Faithfulness/Answer Correctness
(honest stand-ins for what RAGAS/DeepEval normally compute via an LLM
judge). DeepEval's `LLMTestCase` construction is verified working;
RAGAS hits a real, documented upstream dependency conflict in this
sandbox — see `docs/08_evaluation_framework.md` for the full diagnosis, a
real nDCG bug this milestone caught and fixed, and the complete results.

### FastAPI backend (Milestone 9)

```bash
make run-api        # or: poetry run uvicorn api.main:app --reload --port 8000
```

Every endpoint from the spec (`/`, `/health`, `/metrics`, `/model-info`,
`/chat`, `/query`, `/search`, `/summarize`, `/upload`, `/reindex`,
`/evaluate`), all wired to the real pipeline — not stubs. Real API-key auth
+ RBAC (`X-API-Key` header; see `src/security/`), a real token-bucket rate
limiter, prompt-injection detection, and PII-masked audit logging. Full
interactive docs at `/docs`. See `docs/09_fastapi_backend.md` for demo
credentials, the RBAC model, and three real bugs found by actually running
the API end-to-end (not just unit-testing it).

### Streamlit dashboard (Milestone 10)

```bash
make run-api   # in one terminal
make run-ui    # in another — opens http://localhost:8501
```

All 10 spec pages (Chat Assistant, Document Upload, Semantic Search,
Summarization, Compliance Analyzer, Retrieval Explorer, Source Viewer,
Conversation History, Evaluation Dashboard, Admin Monitoring), every one a
real client of the FastAPI backend over HTTP — never a shortcut around it.
Tested end-to-end against a real background `uvicorn` server using
Streamlit's `AppTest`: real widget interactions, real HTTP calls, real
pipeline responses. See `docs/10_streamlit_dashboard.md` for the API
extensions this milestone required (template selection, a rerank toggle,
two new endpoints) and what was verified.

### Monitoring & observability (Milestone 11)

Real OpenTelemetry spans for every pipeline stage (query rewrite,
retrieval, rerank, context build, generation, validation) plus FastAPI
HTTP-level auto-instrumentation — verified with an isolated
`InMemorySpanExporter`, not assumed from the SDK's docs. Real per-stage
latency now recorded (`ValidatedAnswer.metadata["stage_latencies_ms"]`),
closing a gap Milestones 8-9 explicitly flagged as missing. Token usage,
cost estimation, live retrieval health, rolling hallucination-rate
alerting, and a real `scipy.stats.ks_2samp` drift detector, all exposed via
`GET /monitoring/summary`; user feedback (`POST /feedback`) flows from real
thumbs up/down buttons on the Chat Assistant page through to the Admin
Monitoring page's satisfaction rate. See `docs/11_monitoring_and_observability.md`
for the full writeup, including a real test-isolation bug this milestone
caught and fixed.

### Docker & Kubernetes (Milestone 12)

```bash
docker compose up --build   # local: API on :8000, dashboard on :8501
kubectl apply -k k8s/       # or a real cluster
```

No Docker daemon or cluster is available in the sandbox this was built in
— verified instead with real, industry-standard static-binary tools:
`hadolint` on both Dockerfiles, `kubeconform` validating all 11 Kubernetes
resources against the real Kubernetes 1.30 OpenAPI schema (both as
authored and as `kustomize build` actually resolves them), and the
official Compose Specification JSON schema for `docker-compose.yml`. Two
real bugs were caught doing this work: the frontend was silently ignoring
its `RAG_API_BASE_URL` configuration (would have broken the real
deployment), and the validation script's own error handling around
pre-accepted lint warnings. See `docs/12_docker_kubernetes_deployment.md`
for both, plus an honest discussion of where this deployment's actual
architecture (built in Milestones 3-11) diverges from Milestone 1's
original conceptual sketch, and a real, stated limitation around
multi-replica API scaling.

### GitHub setup & CI/CD (Milestone 13)

`.github/workflows/ci.yml` runs on every push/PR to `main`: lint
(`black`/`isort`/`ruff`), typecheck (`mypy`), the full test suite with
coverage, Milestone 12's deployment-artifact validation
(`hadolint`/`kubeconform`/`kustomize`), and a real Docker build of both
images — GitHub Actions runners have a Docker daemon, closing the one gap
Milestone 12 couldn't verify in its own development sandbox.
`.github/workflows/codeql.yml` adds GitHub's native security scanning.
Neither workflow could be run on a real GitHub Actions runner from this
project's own development sandbox either, so both were validated instead
with `actionlint` and `shellcheck` — real, static analysis of the workflow
YAML and every embedded shell step, both clean. Repo hygiene — `LICENSE`
(MIT), `CONTRIBUTING.md`, `SECURITY.md`, issue/PR templates, Dependabot —
rounds out the milestone. See `docs/13_github_cicd.md` for three real
findings caught along the way, including a genuine Poetry dependency
conflict `poetry lock`'s solver surfaced and an honestly-documented limit
on how far Poetry itself could be verified to work in this sandbox.

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 1 | Business understanding & enterprise architecture | ✅ |
| 2 | Repository & folder structure | ✅ |
| 3 | Document ingestion pipeline | ✅ |
| 4 | Chunking & embedding system | ✅ |
| 5 | Vector database & retrieval engine | ✅ |
| 6 | Reranking & prompt engineering | ✅ |
| 7 | Conversational RAG pipeline | ✅ |
| 8 | Evaluation framework | ✅ |
| 9 | FastAPI backend | ✅ |
| 10 | Streamlit enterprise dashboard | ✅ |
| 11 | Monitoring & observability | ✅ |
| 12 | Docker, Kubernetes, deployment | ✅ |
| 13 | GitHub repository setup & CI/CD | ✅ |
| 14 | Final documentation & benchmarking report | ✅ |

## License

[MIT](LICENSE). See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to
propose a change, [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community
standards, and [`SECURITY.md`](SECURITY.md) for reporting a vulnerability
privately rather than as a public issue.
