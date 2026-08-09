from __future__ import annotations


def test_retrieval_explorer_compares_dense_and_reranked(app_test) -> None:
    at = app_test("pages/6_Retrieval_Explorer.py")
    at.run(timeout=15)
    at.text_input[0].input("enhanced due diligence wire transfer").run(timeout=1)
    at.button[0].click().run(timeout=20)
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert "Dense only" in subheaders
    assert "Reranked" in subheaders


def test_source_viewer_loads_real_document(app_test) -> None:
    at = app_test("pages/7_Source_Viewer.py")
    at.run(timeout=15)
    at.text_input[0].input("hr__leave-doc").run(timeout=1)
    at.button[0].click().run(timeout=15)
    assert not at.exception
    assert any("parental leave" in ta.value.lower() for ta in at.text_area)


def test_source_viewer_unknown_doc_shows_error(app_test) -> None:
    at = app_test("pages/7_Source_Viewer.py")
    at.run(timeout=15)
    at.text_input[0].input("does-not-exist").run(timeout=1)
    at.button[0].click().run(timeout=15)
    assert not at.exception
    assert len(at.error) >= 1


def test_conversation_history_shows_real_chat_turns(app_test, live_api_server, admin_key) -> None:
    import requests

    base_url, _ = live_api_server
    chat_response = requests.post(
        f"{base_url}/chat",
        json={"query": "What is the enhanced due diligence threshold?"},
        headers={"X-API-Key": admin_key},
    ).json()
    session_id = chat_response["session_id"]

    at = app_test("pages/8_Conversation_History.py")
    at.run(timeout=15)
    at.text_input[0].input(session_id).run(timeout=1)
    at.button[0].click().run(timeout=15)
    assert not at.exception
    assert len(at.chat_message) == 2
