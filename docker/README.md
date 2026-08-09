# docker/

Dockerfiles for the two real services this project built (Milestone 12):
`api.Dockerfile` (FastAPI backend, Milestone 9) and `frontend.Dockerfile`
(Streamlit dashboard, Milestone 10). `docker-compose.yml` (repo root) runs
both locally; `k8s/` (repo root) has the Kubernetes manifests.

`compose-spec-schema.json` is a vendored copy of the official Compose
Specification schema, used by `scripts/validate_deployment_artifacts.sh`
to validate `docker-compose.yml` without needing network access at
validation time — see `compose-spec-schema.NOTE.md` for its source.

No Docker daemon is available in the sandbox this was built in, so these
were validated with real tooling (hadolint, kubeconform) instead of an
actual build — see `docs/12_docker_kubernetes_deployment.md` for exactly
what that means and doesn't mean.

Quickstart: `docker compose up --build` from the repo root.
