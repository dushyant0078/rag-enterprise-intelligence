# api/
FastAPI gateway (Milestone 9). Real endpoints wired to the real pipeline:
`/chat`, `/query`, `/search`, `/summarize`, `/upload`, `/reindex`,
`/evaluate`, `/health`, `/metrics`, `/model-info`, plus auto-generated
`/docs` (Swagger UI) and `/openapi.json`. Auth, RBAC, rate limiting, and
audit logging live in `src/security/`; this package is the thin HTTP layer
on top. See `docs/09_fastapi_backend.md` for the full writeup.

Run locally: `make run-api` (defaults to http://localhost:8000/docs).
