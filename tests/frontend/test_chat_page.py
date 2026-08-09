from __future__ import annotations


def test_chat_page_loads(app_test) -> None:
    at = app_test("pages/1_Chat_Assistant.py")
    at.run(timeout=15)
    assert not at.exception


def test_chat_page_asks_and_gets_grounded_answer(app_test) -> None:
    at = app_test("pages/1_Chat_Assistant.py")
    at.run(timeout=15)
    at.chat_input[0].set_value("What is the enhanced due diligence threshold?").run(timeout=15)
    assert not at.exception
    assert at.session_state["chat_session_id"] is not None
    assert len(at.session_state["chat_messages"]) == 2


def test_chat_page_submit_thumbs_up_feedback(app_test) -> None:
    at = app_test("pages/1_Chat_Assistant.py")
    at.run(timeout=15)
    at.chat_input[0].set_value("What is the enhanced due diligence threshold?").run(timeout=15)
    assert not at.exception

    # Find and click the thumbs-up button for the assistant's message (index 1).
    up_button = next(b for b in at.button if b.key == "feedback_up_1")
    up_button.click().run(timeout=15)
    assert not at.exception
    assert at.session_state["chat_feedback_given"].get(1) == "up"
    assert any("Feedback recorded" in c.value for c in at.caption)
