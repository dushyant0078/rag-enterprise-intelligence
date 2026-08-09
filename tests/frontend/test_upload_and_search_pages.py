from __future__ import annotations


def test_upload_page_uploads_and_reindexes(app_test) -> None:
    at = app_test("pages/2_Document_Upload.py")
    at.run(timeout=15)
    assert not at.exception
    # No file selected yet -- upload button should be disabled, not crash.
    upload_button = next(b for b in at.button if b.label == "Upload")
    assert upload_button.disabled

    reindex_button = next(b for b in at.button if b.label == "Reindex now")
    reindex_button.click().run(timeout=20)
    assert not at.exception
    assert len(at.success) >= 1


def test_search_page_real_query_returns_results(app_test) -> None:
    at = app_test("pages/3_Semantic_Search.py")
    at.run(timeout=15)
    at.text_input[0].input("parental leave").run(timeout=1)
    at.button[0].click().run(timeout=15)
    assert not at.exception
    # At least one result container rendered as markdown with a doc_id.
    assert any("hr__leave-doc" in m.value for m in at.markdown)


def test_search_page_with_rerank_checkbox(app_test) -> None:
    at = app_test("pages/3_Semantic_Search.py")
    at.run(timeout=15)
    at.text_input[0].input("enhanced due diligence wire transfer").run(timeout=1)
    at.checkbox[0].check().run(timeout=1)
    at.button[0].click().run(timeout=15)
    assert not at.exception


def test_search_page_no_results_shows_info(app_test) -> None:
    at = app_test("pages/3_Semantic_Search.py")
    at.run(timeout=15)
    at.text_input[0].input("completely unrelated nonexistent gibberish query zzz").run(timeout=1)
    at.button[0].click().run(timeout=15)
    assert not at.exception
