## What does this change?

<!-- What, and why. Link an issue if there is one. -->

## Which milestone/component does this touch?

<!-- e.g. "Milestone 7 pipeline", "API auth", "Streamlit Chat Assistant page" -->

## How was this verified?

<!--
Be specific about what actually ran, matching this project's own convention
(see CONTRIBUTING.md and any docs/NN_*.md for examples): which tests, against
which backend configuration, and what the real output was — not just "should
work". If something can't be verified in your environment (no credentials,
no network path to a service), say so explicitly rather than leaving it
implied.
-->

- [ ] `make lint` passes
- [ ] `make typecheck` passes
- [ ] `make test` passes
- [ ] New code has new tests (or an explanation of why it's untestable here)
- [ ] Touched `docker/` or `k8s/`? `bash scripts/validate_deployment_artifacts.sh` passes
- [ ] Docs updated if this changes documented behavior

## Anything reviewers should scrutinize specifically?

<!-- Optional but encouraged — e.g. "not confident about the RBAC filter
intersection logic" or "didn't test the multi-replica case" -->
