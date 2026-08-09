# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security vulnerability.
Instead, use GitHub's private vulnerability reporting (Security tab →
"Report a vulnerability") on this repository, or email the maintainer
directly if that isn't available. Include what you found, how to
reproduce it, and its potential impact; you should get an acknowledgment
within a few days.

## What's actually built here

This project implements several real, tested security controls — worth
knowing both for reporting issues accurately and for understanding what's
already covered:

| Control | Where | Notes |
|---|---|---|
| API key authentication + RBAC | `src/security/api_auth.py` | In-memory registry — a real deployment needs a real secrets/identity backend behind it (see `docs/09_fastapi_backend.md`) |
| Rate limiting | `src/security/rate_limiter.py` | Real token-bucket algorithm |
| PII masking (audit logs only) | `src/security/pii_masking.py` | Regex-based; real but not comprehensive — see its module docstring |
| Prompt injection detection | `src/security/prompt_injection.py` | A heuristic first layer, explicitly documented as a floor not a ceiling |
| Audit logging | `src/security/audit_log.py` | Structured, PII-masked |
| Response grounding / hallucination rejection | `src/citations/response_validator.py` | Enforced, not just requested — see `docs/07_conversational_rag_pipeline.md` |

None of these claim to be a complete security program on their own. If
you're evaluating this project for a real deployment, read
`docs/09_fastapi_backend.md`'s "Known limitations" section and
`docs/12_docker_kubernetes_deployment.md`'s deployment-topology notes
before treating any of the above as sufficient in isolation.

## Supported versions

This is a milestone-built demonstration project (see `README.md`'s
roadmap), not a versioned package with a support matrix — security fixes
land on the `main` branch as they're found, not backported to tagged
releases.

## Secrets

Never commit real API keys, credentials, or `.env` files — `.env.example`
is the template; copy it to `.env` locally and keep that out of version
control (already gitignored). `k8s/secret.yaml` is a placeholder template
for the same reason — see its own header comment.
