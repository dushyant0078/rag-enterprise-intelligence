#!/usr/bin/env bash
# Validates every Docker/Kubernetes artifact from Milestone 12 without needing a Docker
# daemon or a live cluster -- hadolint lints the Dockerfiles, kubeconform validates every
# Kubernetes manifest (both as authored and as `kustomize build` actually resolves them)
# against the real Kubernetes OpenAPI schema, and docker-compose.yml is checked against the
# official Compose Specification JSON schema. Requires hadolint, kubeconform, and kustomize
# on PATH (each a static binary from their GitHub releases -- no package manager needed) plus
# the `jsonschema` and `pyyaml` Python packages.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== hadolint: Dockerfiles ==="
# DL3008 (pin apt package versions) and DL3066 (numeric vs named USER) are deliberately
# accepted -- see the matching comments in docker/*.Dockerfile for why. Ignored explicitly
# here (not just piped past) so this script still fails on any *other*, unreviewed finding.
hadolint --ignore DL3008 --ignore DL3066 docker/api.Dockerfile docker/frontend.Dockerfile

echo "=== docker-compose.yml vs the Compose Specification schema ==="
python3 - <<'PYEOF'
import json
import sys

import yaml
from jsonschema import Draft202012Validator

with open("docker-compose.yml") as f:
    doc = yaml.safe_load(f)

# Vendored (docker/compose-spec-schema.json) rather than fetched over the network each run --
# see docker/compose-spec-schema.NOTE.md for its source and how to refresh it. The schema
# itself declares "$schema": draft 2020-12 -- validate with the matching validator, not an
# older draft's, since keyword semantics genuinely differ between drafts.
with open("docker/compose-spec-schema.json") as f:
    schema = json.load(f)

errors = list(Draft202012Validator(schema).iter_errors(doc))
if errors:
    for e in errors:
        print("ERROR:", list(e.path), e.message)
    sys.exit(1)
print("valid")
PYEOF

echo "=== kubeconform: individual manifests ==="
kubeconform -summary -kubernetes-version 1.30.0 \
    k8s/namespace.yaml k8s/configmap.yaml k8s/secret.yaml k8s/api-pvc.yaml \
    k8s/api-deployment.yaml k8s/api-service.yaml k8s/api-hpa.yaml \
    k8s/frontend-deployment.yaml k8s/frontend-service.yaml k8s/ingress.yaml

echo "=== kubeconform: kustomize build output ==="
kustomize build k8s/ | kubeconform -summary -kubernetes-version 1.30.0 -

echo "=== all deployment artifacts valid ==="
