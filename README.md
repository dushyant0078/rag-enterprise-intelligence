# RAG Enterprise Intelligence

[![CI](https://github.com/dushyant0078/rag-enterprise-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/dushyant0078/rag-enterprise-intelligence/actions/workflows/ci.yml)
[![CodeQL](https://github.com/dushyant0078/rag-enterprise-intelligence/actions/workflows/codeql.yml/badge.svg)](https://github.com/dushyant0078/rag-enterprise-intelligence/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Retrieval-Augmented Generation (RAG) platform for answering questions over enterprise documents — PDFs, contracts, policies, spreadsheets, even scanned scans — with answers that are grounded in citations instead of guesses.

Most RAG demos stop at "it returns an answer." This one is built the way a real internal tool would need to be: it reranks results before they reach the LLM, refuses to answer when it isn't confident, tracks who accessed what, and ships with a proper API, a dashboard, monitoring, and deployment configs — not just a notebook.

## Why this exists

Enterprises sit on huge document collections that are too large to read manually and too high-stakes to hand to an LLM that might hallucinate a wrong policy or a made-up clause. This project explores what it actually takes to make RAG _trustworthy enough_ for that kind of use case — grounded citations, access control, and a system you can actually deploy and monitor, not just a proof of concept.

## What it does

- **Hybrid retrieval** — combines dense (embedding-based) and sparse (keyword) search, then reranks results before they ever reach the LLM
- **Grounded answers only** — a dedicated validation step checks every generated claim against the retrieved passages and returns "insufficient evidence" instead of a confident-sounding guess
- **Citations you can trust** — every answer links back to the exact document, page, and chunk it came from, with a confidence score
- **Access control built in** — retrieval respects role-based permissions, so a query never surfaces a document the user shouldn't see
- **Swap backends without touching code** — change the embedding model, vector store, or LLM provider through config, whether that's FAISS or Qdrant, GPT or Claude

## Quickstart

```bash
git clone https://github.com/dushyant0078/rag-enterprise-intelligence.git
cd rag-enterprise-intelligence
make install
make generate-samples
make ingest DOMAIN=hr INPUT=documents/hr
make run-api      # terminal 1 — http://localhost:8000/docs
make run-ui       # terminal 2 — http://localhost:8501
```

No API keys needed to try it — it runs out of the box on an offline embedding model and a local FAISS index. Point it at a real LLM or vector store later by setting a few environment variables (see `.env.example`). Prefer Docker? `docker compose up --build` does the same thing.

## Architecture

User → Chat UI → FastAPI Gateway → Auth → Conversation Manager → Query Rewriter
→ Embedding Service → Vector DB → Retriever → Reranker → Context Builder
→ LLM Inference → Citation Generator → Response Validator → User

Full diagrams and design reasoning: [`docs/02_architecture.md`](docs/02_architecture.md).

## Tech stack

**Backend:** Python, FastAPI, Pydantic
**Retrieval:** FAISS, scikit-learn, hybrid dense+sparse search, reranking
**Frontend:** Streamlit
**Observability:** OpenTelemetry, custom monitoring/alerting
**Deployment:** Docker, Kubernetes (manifests included)
**Testing/CI:** Pytest, GitHub Actions, CodeQL

## Testing & quality

- **480 automated tests**, 86% coverage — including live HTTP tests against a running API and end-to-end frontend tests against a real backend
- Measured under a real load test: ~113 requests/sec throughput, ~5ms median API latency
- Rate limiting and RBAC verified under real traffic, not just unit tests

Full benchmark numbers and methodology: [`reports/final_benchmark_report.md`](reports/final_benchmark_report.md).

## Project structure

├── src/ # core pipeline — ingestion, retrieval, reranking, LLM, memory, monitoring
├── api/ # FastAPI backend
├── frontend/ # Streamlit dashboard
├── tests/ # mirrors src/ 1:1
├── docker/, k8s/ # deployment configs
├── docs/ # detailed design notes per component
└── reports/ # benchmark results

## Known limitations

- Some LLM/vector-store integrations (OpenAI, Gemini, Weaviate, Pinecone) are implemented and tested for correctness but weren't exercised end-to-end in this environment due to network/credential restrictions during development.
- The API currently runs safely as a single instance; horizontal scaling needs shared session storage first (documented in `docs/12_docker_kubernetes_deployment.md`).
- More detail on trade-offs and what's fully verified vs. what's implemented-but-untested lives in `docs/`.

## Documentation

Deeper write-ups for each part of the system — ingestion, chunking, retrieval, the API, deployment, monitoring — are in [`docs/`](docs/), roughly one file per subsystem.

## License

[MIT](LICENSE) — see [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) for how to contribute or report an issue.
