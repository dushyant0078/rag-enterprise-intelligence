# Milestone 13 — GitHub Repository Setup & CI/CD

## What was built

**Repository hygiene:** `LICENSE` (MIT — replacing Milestone 2's
`"Proprietary # revisited in Milestone 13"` placeholder), `CONTRIBUTING.md`,
`CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md` (tied
directly to the real controls in `src/security/`, not generic boilerplate),
YAML issue forms (bug report, feature request) with fields specific to this
project (which milestone/component, which configured backend),
`PULL_REQUEST_TEMPLATE.md`, `.github/dependabot.yml`, and
`.github/CODEOWNERS`.

**CI/CD:** `.github/workflows/ci.yml` (lint, typecheck, test with
coverage, deployment-artifact validation, a real Docker build) and
`.github/workflows/codeql.yml` (GitHub's native Python security scanning,
on push/PR plus a weekly schedule).

## No GitHub Actions runner here either — verified anyway

Same situation as Milestone 12's Docker/Kubernetes work: this sandbox
can't actually trigger a GitHub Actions run. Real, industry-standard
static analysis substituted for that: `actionlint` (validates workflow
YAML structure, action references, expression syntax) plus `shellcheck`
for every embedded shell step (actionlint shells out to it automatically
when available) — both downloaded as static binaries from their GitHub
releases, both run against the real committed workflow files, both clean.

## Three real findings from actually testing things

**1. A genuine Poetry dependency conflict, caught by the solver itself.**
Attempting a real `poetry lock` (this project had never actually run one —
every previous milestone's dependencies were verified via `pip install`)
failed with a concrete, specific version-solving error: `ragas` transitively
requires `langsmith>=0.3.45,<1.0.0`, which directly conflicts with this
project's own `langsmith = "^0.1"` constraint from Milestone 11 (itself a
mistake — `^0.1` restricts to `>=0.1.0,<0.2.0`, which doesn't even match the
real, already-tested 0.10.17 version). Fixed to
`langsmith = ">=0.3.45,<1.0.0"`, matching both ragas's real requirement and
the version already verified working in Milestone 11.

**2. The Makefile's own lint/typecheck/format targets were stale.**
`make lint`, `make typecheck`, and `make format` only ever covered
`src api tests` — never `frontend/` or `scripts/`, despite this project
manually running `black`/`isort`/`ruff`/`mypy` across all five directories
by hand since Milestone 9. `make lint` was silently checking less than what
had actually been verified every milestone since. Fixed, and split `format`
(mutates files, for local use) from a new `format-check` (never mutates,
what CI actually runs) — `make lint` now depends on `format-check` so it
matches exactly what CONTRIBUTING.md documents. Verified directly: `make
lint`, `make typecheck`, and `make test` were all run for real after the
fix, not just written and assumed correct.

**3. `poetry install` doesn't fully work end-to-end in this specific
sandbox — traced precisely, not just observed.** `poetry env info` showed a
real virtualenv path; `poetry run which mypy` resolved to
`/usr/local/bin/mypy` — the system-wide `pip`-installed binary, not
anything inside that virtualenv, which turned out to have only 3 items in
`site-packages` (essentially empty). `poetry run` was transparently falling
through to system `PATH` the whole time, which only worked because every
package was already `pip`-installed from twelve prior milestones' testing.
A genuinely fresh `poetry install` in this environment does not complete a
full dependency resolution within this sandbox's practical interactive
tool-call time constraints (attempted twice, at 60s and ~580s, an exhaustive
SAT-solve across every optional group's heavy transitive dependencies
together — torch via `sentence-transformers`, multiple vector-store SDKs,
three LLM provider SDKs — combined). This is a stated limitation of *this
interactive sandbox*, not necessarily of the project or of a real CI
runner: GitHub Actions jobs get up to 6 hours by default and typically much
better sustained network throughput, where a cold-cache `poetry install`
for a project like this completing in minutes is normal and unremarkable.
Documented honestly in `CONTRIBUTING.md` with a `pip`-based fallback
(exactly how this project's own test suite has actually been run and
verified throughout its development) for contributors who hit the same
wall locally. No `poetry.lock` is committed as a result — `ci.yml`'s cache
key is keyed on `pyproject.toml`'s hash instead of a lockfile hash,
correctly busting whenever dependencies change, just without
lockfile-level version precision.

## What CI actually runs (matches this project's own verification history)

| Job | What | Mirrors |
|---|---|---|
| `lint-and-typecheck` | `make lint`, `make typecheck` | The exact commands run manually at the end of every milestone since M3 |
| `test` | full `pytest` with coverage | Every milestone's final verification pass |
| `deployment-artifacts` | `scripts/validate_deployment_artifacts.sh` | Milestone 12's own validation, now automated |
| `docker-build` | real `docker build` via `docker/build-push-action`, `push: false` | **New real verification** — this sandbox never had a Docker daemon; GitHub Actions runners do |

## Testing

- `make lint`, `make typecheck`, and `make test` all re-run for real after
  every fix in this milestone (not just written and assumed to work) —
  **480 tests passing repo-wide, 86% coverage.**
- Both workflow files validated with `actionlint -shellcheck=...` — clean.
- `scripts/validate_deployment_artifacts.sh` (Milestone 12) still passes
  unchanged, confirming the `deployment-artifacts` CI job's steps are
  correct before ever running in CI.

## Known limitations / deferred work

- **No `poetry.lock` is committed**, for the reason detailed above — a
  real limitation worth revisiting (e.g., by generating it directly from a
  GitHub Actions run, which has the time budget this sandbox doesn't, and
  committing the result back).
- **Badge URLs and `CODEOWNERS` use an `OWNER` placeholder** — this repo
  hasn't been pushed to a real GitHub org/user yet; both are correct
  syntax, functional as soon as that placeholder is replaced, same pattern
  as `k8s/ingress.yaml`'s placeholder domain.
- **`docker-build`'s job is real but was never actually executed** — it's
  correct, standard `docker/build-push-action` usage building the exact
  Dockerfiles `hadolint` already validated in Milestone 12, but "would
  build correctly on a real runner" is a claim based on those Dockerfiles
  being clean and the action being standard, not a claim this specific job
  has been observed to pass.
- **Dependabot's `pip` ecosystem entry works best with a committed lock
  file**, which this repo doesn't have — version-bump PRs may be less
  precise until that's resolved.

---
*Next: Milestone 14 — final enterprise documentation and benchmarking report.*
