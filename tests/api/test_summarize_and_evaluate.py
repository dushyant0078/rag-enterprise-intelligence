from __future__ import annotations


def test_summarize_raw_text(client, admin_key) -> None:
    r = client.post(
        "/summarize",
        json={"text": "Wire transfers over $10,000 require enhanced due diligence review."},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200
    assert r.json()["source"] == "text"
    assert "Summarize the above" not in r.json()["summary"]  # no template artifacts leaking through
    assert len(r.json()["summary"]) > 0


def test_summarize_by_doc_id(client, admin_key) -> None:
    r = client.post("/summarize", json={"doc_id": "hr__hr-doc"}, headers={"X-API-Key": admin_key})
    assert r.status_code == 200
    assert r.json()["source"] == "doc_id:hr__hr-doc"


def test_summarize_unknown_doc_id_returns_404(client, admin_key) -> None:
    r = client.post(
        "/summarize", json={"doc_id": "does-not-exist"}, headers={"X-API-Key": admin_key}
    )
    assert r.status_code == 404


def test_summarize_requires_text_or_doc_id(client, admin_key) -> None:
    r = client.post("/summarize", json={}, headers={"X-API-Key": admin_key})
    assert r.status_code == 400


def test_evaluate_runs_full_benchmark(client, admin_key) -> None:
    r = client.post("/evaluate", json={"k": 3}, headers={"X-API-Key": admin_key})
    assert r.status_code == 200
    body = r.json()
    assert body["benchmark_size"] == 7
    assert body["k"] == 3
    assert 0.0 <= body["mean_precision_at_k"] <= 1.0
    assert 0.0 <= body["grounded_rate"] <= 1.0
    assert body["latency_p50_ms"] >= 0


def test_evaluate_requires_auth(client) -> None:
    r = client.post("/evaluate", json={})
    assert r.status_code == 401


def test_evaluate_validates_k_range(client, admin_key) -> None:
    r = client.post("/evaluate", json={"k": 0}, headers={"X-API-Key": admin_key})
    assert r.status_code == 422
    r = client.post("/evaluate", json={"k": 100}, headers={"X-API-Key": admin_key})
    assert r.status_code == 422
