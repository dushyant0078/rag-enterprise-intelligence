from __future__ import annotations

from pathlib import Path

from src.evaluation.benchmark_dataset import BenchmarkDataset, default_sample_corpus_benchmark
from src.evaluation.latency import percentile, summarize_latencies

# -- latency -----------------------------------------------------------


def test_percentile_p50_of_sorted_list() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0


def test_percentile_p95_close_to_max() -> None:
    values = list(range(1, 101))  # 1..100
    result = percentile([float(v) for v in values], 95)
    assert result >= 90.0


def test_percentile_empty_list() -> None:
    assert percentile([], 50) == 0.0


def test_summarize_latencies_structure() -> None:
    summary = summarize_latencies([10.0, 20.0, 30.0])
    assert set(summary.keys()) == {"p50", "p95", "mean"}
    assert summary["mean"] == 20.0


def test_summarize_latencies_empty() -> None:
    summary = summarize_latencies([])
    assert summary == {"p50": 0.0, "p95": 0.0, "mean": 0.0}


# -- benchmark dataset -----------------------------------------------------


def test_default_sample_corpus_benchmark_has_seven_queries() -> None:
    queries = default_sample_corpus_benchmark()
    assert len(queries) == 7
    assert len({q.query_id for q in queries}) == 7  # all unique ids


def test_every_default_query_has_expected_answer_and_relevant_domain() -> None:
    for q in default_sample_corpus_benchmark():
        assert q.expected_answer
        assert q.relevant_doc_ids


def test_benchmark_dataset_iterable_and_len() -> None:
    dataset = BenchmarkDataset.default_sample_corpus()
    assert len(dataset) == 7
    assert len(list(dataset)) == 7


def test_benchmark_dataset_save_and_load_roundtrip(tmp_path: Path) -> None:
    dataset = BenchmarkDataset.default_sample_corpus()
    path = tmp_path / "benchmark.json"
    dataset.save(path)

    reloaded = BenchmarkDataset.load(path)
    assert len(reloaded) == len(dataset)
    assert reloaded.queries[0].query_id == dataset.queries[0].query_id
    assert reloaded.queries[0].expected_answer == dataset.queries[0].expected_answer
