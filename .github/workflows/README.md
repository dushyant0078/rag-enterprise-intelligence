# .github/workflows/

- **`ci.yml`** — on every push/PR to `main`: lint (black/isort/ruff),
  typecheck (mypy), the full pytest suite with coverage, deployment-artifact
  validation (hadolint/kubeconform/kustomize — the same real tools used in
  Milestone 12, no Docker daemon needed), and a real Docker build of both
  images (GitHub Actions runners have a Docker daemon, closing the one gap
  Milestone 12 couldn't verify in its own development sandbox).
- **`codeql.yml`** — GitHub's native static security analysis for Python,
  on every push/PR plus a weekly scheduled scan.

Both validated with `actionlint` (+ `shellcheck` for embedded shell steps)
before being committed — see `docs/13_github_cicd.md`.
