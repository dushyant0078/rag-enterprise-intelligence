# Contributing

This project was built milestone-by-milestone (see the roadmap in
`README.md`); `docs/` has a full write-up of what each milestone verified
and how. That history is worth reading before a substantial change —
several design decisions (why the pipeline doesn't use LangChain, why the
API HPA is capped at 1 replica, why certain backends are "correct but
untestable here") are explained there, not just in code comments.

## Development setup

```bash
git clone <this-repo>
cd RAG-Enterprise-Intelligence
make install     # poetry install --with every optional group
cp .env.example .env   # fill in real credentials only for backends you'll exercise
```

`make install` pulls in some heavy transitive dependencies (torch via
`sentence-transformers`, several vector-store SDKs, three LLM provider
SDKs) across all the optional groups together — full resolution can
genuinely take several minutes on a cold cache, more if `poetry.lock`
needs to be generated fresh (this repo doesn't commit one — see
`docs/13_github_cicd.md` for why). If `poetry install` is impractical in
your environment, every command in this project also works against a
plain `pip install --break-system-packages -e .` plus whichever optional
groups' packages you need individually (`pyproject.toml`'s
`[tool.poetry.group.*.dependencies]` sections list them); that's how this
project's own test suite was actually run and verified throughout
development.

## Before opening a PR

```bash
make lint        # black --check, isort --check, ruff
make typecheck    # mypy across src/, api/, frontend/
make test         # the full pytest suite
```

All three run in CI (`.github/workflows/ci.yml`) on every PR; running them
locally first saves a round trip. If you touched `docker/` or `k8s/`, also
run `bash scripts/validate_deployment_artifacts.sh` (needs `hadolint`,
`kubeconform`, and `kustomize` on `PATH` — see that script's header for
where to get them; no Docker daemon or live cluster required).

## Code conventions

- **Real over aspirational.** If a backend or integration can't actually be
  exercised in your environment (no credentials, no network path), say so
  in a comment and in the PR description, and add whatever *can* be
  verified — an import check, a construction check, a schema check —
  rather than leaving it silently untested. Grep `docs/` for "untestable"
  or "correct-but-" for the established pattern.
- **New backends follow the registry pattern** already used throughout
  `src/` (`src/llm/registry.py`, `src/retrieval/vectorstore/registry.py`,
  etc.) — a `BaseX` ABC, a concrete class, and a name → class mapping.
- **Tests mirror `src/` 1:1.** A new module gets a same-named test file
  under the matching `tests/` subdirectory.
- **Formatting is enforced, not debated** — `black`/`isort` with
  `--line-length 100`, applied via `make lint`. Don't hand-format around it.
- **Type hints are required** on new public functions/methods; `mypy`
  runs in strict-ish mode (see `pyproject.toml`'s `[tool.mypy]`) and CI
  fails on new errors.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/` — they ask for
the specific context (what you ran, what backend/config, what you
expected) that makes a real RAG pipeline bug reproducible. Security
issues follow a different path — see `SECURITY.md`, not a public issue.
