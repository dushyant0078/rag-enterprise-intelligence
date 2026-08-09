from __future__ import annotations


def test_evaluation_dashboard_runs_real_benchmark(app_test) -> None:
    at = app_test("pages/9_Evaluation_Dashboard.py")
    at.run(timeout=15)
    at.button[0].click().run(timeout=30)
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert "Retrieval metrics" in subheaders
    assert "Generation metrics" in subheaders
    assert "Response latency" in subheaders
    assert any("Benchmark size: 7 queries" in c.value for c in at.caption)


def test_admin_monitoring_shows_health_and_metrics(app_test) -> None:
    at = app_test("pages/10_Admin_Monitoring.py")
    at.run(timeout=15)
    assert not at.exception
    assert len(at.success) >= 1  # health status
    # Configured components rendered as metrics.
    metric_labels = [m.label for m in at.metric]
    assert "Vector store" in metric_labels
    assert "Llm" in metric_labels or "LLM" in metric_labels
    # Real Prometheus text made it into a code block.
    assert any("rag_api_requests_total" in c.value for c in at.code)


def test_admin_monitoring_shows_monitoring_summary(app_test) -> None:
    at = app_test("pages/10_Admin_Monitoring.py")
    at.run(timeout=15)
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert "Monitoring summary" in subheaders
    metric_labels = [m.label for m in at.metric]
    assert "Estimated cost" in metric_labels
    assert "Grounded rate" in metric_labels
    assert "Feedback satisfaction" in metric_labels


def test_admin_monitoring_reflects_real_chat_and_feedback(
    app_test, live_api_server, admin_key
) -> None:
    import requests

    base_url, _ = live_api_server
    chat_response = requests.post(
        f"{base_url}/chat", json={"query": "What is the enhanced due diligence threshold?"},
        headers={"X-API-Key": admin_key},
    ).json()  # fmt: skip
    requests.post(
        f"{base_url}/feedback",
        json={"session_id": chat_response["session_id"], "rating": "up"},
        headers={"X-API-Key": admin_key},
    )

    at = app_test("pages/10_Admin_Monitoring.py")
    at.run(timeout=15)
    assert not at.exception
    metric_values = {m.label: m.value for m in at.metric}
    assert metric_values["Feedback satisfaction"] == "100%"
