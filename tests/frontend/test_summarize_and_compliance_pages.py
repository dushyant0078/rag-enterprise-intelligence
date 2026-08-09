from __future__ import annotations


def test_summarize_page_pasted_text(app_test) -> None:
    at = app_test("pages/4_Summarization.py")
    at.run(timeout=15)
    at.text_area[0].input(
        "Wire transfers over $10,000 require enhanced due diligence review by compliance."
    ).run(timeout=1)
    at.button[0].click().run(timeout=15)
    assert not at.exception
    assert any("Source: text" in c.value for c in at.caption)


def test_summarize_page_by_doc_id(app_test) -> None:
    at = app_test("pages/4_Summarization.py")
    at.run(timeout=15)
    at.radio[0].set_value("Ingested document (by ID)").run(timeout=1)
    at.text_input[0].input("hr__leave-doc").run(timeout=1)
    at.button[0].click().run(timeout=15)
    assert not at.exception
    assert any("hr__leave-doc" in c.value for c in at.caption)


def test_compliance_analyzer_real_check(app_test) -> None:
    at = app_test("pages/5_Compliance_Analyzer.py")
    at.run(timeout=15)
    at.text_area[0].input(
        "A $12,000 domestic wire transfer was processed without an enhanced due diligence review."
    ).run(timeout=1)
    at.button[0].click().run(timeout=15)
    assert not at.exception
    # Grounded or not, the page must render without error and show a confidence caption.
    assert any("confidence" in c.value for c in at.caption)
