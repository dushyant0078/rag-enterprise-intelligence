from __future__ import annotations


def test_home_page_connects_and_shows_health(app_test) -> None:
    at = app_test("app.py")
    at.run(timeout=15)
    assert not at.exception
    assert any("RAG Enterprise Intelligence" in t.value for t in at.title)
    assert any("Connected" in s.value for s in at.success)


def test_home_page_shows_error_for_unreachable_api(app_test) -> None:
    at = app_test("app.py")
    at.session_state["api_base_url"] = "http://127.0.0.1:1"  # nothing listens here
    at.run(timeout=15)
    assert not at.exception
    assert len(at.error) >= 1


def test_home_page_lists_all_ten_pages(app_test) -> None:
    at = app_test("app.py")
    at.run(timeout=15)
    body = " ".join(m.value for m in at.markdown)
    for label in (
        "Chat Assistant",
        "Document Upload",
        "Semantic Search",
        "Summarization",
        "Compliance Analyzer",
        "Retrieval Explorer",
        "Source Viewer",
        "Conversation History",
        "Evaluation Dashboard",
        "Admin Monitoring",
    ):
        assert label in body
