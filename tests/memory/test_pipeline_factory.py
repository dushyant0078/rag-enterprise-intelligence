from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.pipeline_factory import build_default_pipeline, build_index


def _write_sample_doc(processed_dir: Path, domain: str, doc_id: str, text: str) -> None:
    from src.parsing.schemas import PageContent, ParsedDocument

    # doc_id must carry the "domain__..." prefix the real ingestion pipeline always
    # produces (src/ingestion/pipeline.py's _make_doc_id) -- domain filtering derives the
    # domain from that prefix, not from the folder the file is saved under (see the same
    # fix, with the bug it caught, in tests/api/conftest.py's _write_doc).
    full_doc_id = f"{domain}__{doc_id}"
    doc = ParsedDocument(
        source_path=f"{full_doc_id}.txt",
        format="txt",
        pages=[PageContent(page_number=1, text=text)],
        full_text=text,
        page_count=1,
    )
    out_dir = processed_dir / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{full_doc_id}.json").write_text(doc.model_dump_json(), encoding="utf-8")


def test_build_index_raises_on_empty_corpus(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_index(processed_dir=tmp_path / "empty", index_dir=tmp_path / "index")


def test_build_index_produces_searchable_store(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    _write_sample_doc(
        processed_dir, "hr", "doc1", "Employees get twenty six weeks of paid parental leave."
    )

    store, model, count = build_index(processed_dir=processed_dir, index_dir=tmp_path / "index")
    assert count > 0

    query_vector = model.embed(["parental leave weeks"])[0]
    results = store.search(query_vector.tolist(), top_k=1, namespace="all")
    assert len(results) == 1


def test_domain_is_derived_from_doc_id_prefix_not_folder_placement(tmp_path: Path) -> None:
    """Regression test for a real bug caught via an end-to-end API test (not a
    unit test with a hand-picked correct fixture): domain filtering keys off
    the "domain__..." prefix in the doc_id itself (matching the real
    ingestion pipeline's naming, src/ingestion/pipeline.py's _make_doc_id),
    not which folder the processed JSON happens to be saved under. A test
    fixture that saved a plainly-named doc (e.g. "hr-doc") into an "hr/"
    folder silently produced zero domain-filtered results rather than
    wrong ones, which is a worse failure mode to debug -- caught only by an
    API test that asserted on actual grounding, not by checking file placement.
    """
    from src.memory.pipeline_factory import load_processed_documents

    processed_dir = tmp_path / "processed"
    _write_sample_doc(processed_dir, "hr", "doc1", "Some HR content.")

    docs = load_processed_documents(processed_dir)
    assert len(docs) == 1
    doc_id, domain, _ = docs[0]
    assert domain == "hr"
    assert doc_id.startswith("hr__")


def test_rebuilding_index_with_different_corpus_does_not_crash_on_dimension_mismatch(
    tmp_path: Path,
) -> None:
    """Regression test for a real bug caught via the API's /reindex endpoint:
    calling build_index() a second time against a *changed* corpus (a
    realistic reindex-after-upload scenario) previously tried to upsert
    freshly-fit-TF-IDF vectors of a new dimensionality into a FAISS store
    that had reloaded the *previous* run's persisted, differently-dimensioned
    index from disk, raising a FAISS assertion. Fixed by clearing the
    persisted index directory on every rebuild, since TF-IDF's fitted
    dimensionality is a function of the corpus and isn't stable across
    corpus changes the way a pretrained embedding model's would be.
    """
    processed_dir = tmp_path / "processed"
    index_dir = tmp_path / "index"

    _write_sample_doc(
        processed_dir, "hr", "doc1", "Employees get twenty six weeks of paid parental leave."
    )
    build_index(processed_dir=processed_dir, index_dir=index_dir)

    # Add a second, different document -- changes the fitted TF-IDF vocabulary/dimensionality.
    _write_sample_doc(
        processed_dir, "banking", "doc2", "Wire transfers over ten thousand dollars require review."
    )
    store, model, count = build_index(
        processed_dir=processed_dir, index_dir=index_dir
    )  # must not raise
    assert count > 1
    assert store.count("all") == count


def test_build_default_pipeline_answers_from_real_corpus(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    _write_sample_doc(
        processed_dir,
        "hr",
        "doc1",
        "Employees in their first year get twelve weeks of parental leave.",
    )

    pipeline = build_default_pipeline(processed_dir=processed_dir, index_dir=tmp_path / "index")
    session_id = pipeline.start_session(user_id="test")
    result = pipeline.ask(session_id, "How many weeks of parental leave in the first year?")
    assert result.is_grounded
