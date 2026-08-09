from __future__ import annotations


def test_root_returns_api_info(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "RAG Enterprise Intelligence API"
    assert body["docs_url"] == "/docs"


def test_health_ok_when_corpus_indexed(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "vector_store" in r.json()["components"]


def test_health_requires_no_auth(client) -> None:
    r = client.get("/health")  # no X-API-Key header
    assert r.status_code == 200


def test_metrics_returns_prometheus_text_format(client) -> None:
    client.get("/health")  # generate at least one recorded request
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "rag_api_requests_total" in r.text
    assert 'path="/health"' in r.text


def test_metrics_requires_no_auth(client) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200


def test_model_info_requires_auth(client) -> None:
    r = client.get("/model-info")
    assert r.status_code == 401


def test_model_info_returns_configured_components(client, admin_key) -> None:
    r = client.get("/model-info", headers={"X-API-Key": admin_key})
    assert r.status_code == 200
    body = r.json()
    assert body["vector_store"] == "faiss"
    assert body["reranker"] == "lexical_overlap"
    assert body["chunking_strategy"] == "hybrid"
    assert body["llm_model"] == "deterministic-test-llm"


def test_docs_endpoint_available(client) -> None:
    r = client.get("/docs")
    assert r.status_code == 200


def test_openapi_schema_available(client) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "/chat" in schema["paths"]
    assert "/query" in schema["paths"]
    assert "/search" in schema["paths"]
    assert "/upload" in schema["paths"]
    assert "/reindex" in schema["paths"]
    assert "/summarize" in schema["paths"]
    assert "/evaluate" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/metrics" in schema["paths"]
    assert "/model-info" in schema["paths"]
