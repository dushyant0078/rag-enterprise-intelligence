"""Custom benchmark dataset (the spec's "Custom benchmark datasets" item):
formalizes the 7 hand-written queries used informally across Milestones
4-7's comparison scripts into a real, labeled, persistable benchmark --
each query now has ground-truth relevant_doc_ids (for retrieval metrics)
and a verified expected_answer (for the answer-correctness proxy), instead
of just a pass/fail domain check.

Expected answers were written by reading the actual sample documents (see
scripts/generate_sample_documents.py), not guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.schemas import BenchmarkQuery


def default_sample_corpus_benchmark() -> list[BenchmarkQuery]:
    return [
        BenchmarkQuery(
            query_id="banking-edd-threshold",
            query="enhanced due diligence threshold for international wire transfers",
            expected_answer=(
                "International wire transfers over $3,000 require enhanced due diligence; "
                "domestic wire transfers require it above $10,000."
            ),
            relevant_doc_ids=["banking"],
            domain="banking",
        ),
        BenchmarkQuery(
            query_id="hr-parental-leave-first-year",
            query="how many weeks of paid parental leave in the first year",
            expected_answer=(
                "Employees in their first year of employment are eligible for 12 weeks of "
                "paid parental leave."
            ),
            relevant_doc_ids=["hr"],
            domain="hr",
        ),
        BenchmarkQuery(
            query_id="insurance-water-damage-vacancy",
            query="water damage coverage if the property was vacant",
            expected_answer=(
                "Water damage is covered, but the vacancy exclusion applies if the property "
                "was vacant or unoccupied for more than 60 consecutive days before the loss, "
                "unless the Vacancy Permit endorsement was purchased."
            ),
            relevant_doc_ids=["insurance"],
            domain="insurance",
        ),
        BenchmarkQuery(
            query_id="legal-termination-notice",
            query="notice period to terminate the vendor agreement for convenience",
            expected_answer=(
                "Either party may terminate the agreement for convenience upon sixty (60) "
                "days' prior written notice to the other party."
            ),
            relevant_doc_ids=["legal"],
            domain="legal",
        ),
        BenchmarkQuery(
            query_id="technical-lockout-tagout",
            query="lockout tagout procedure before servicing the equipment",
            expected_answer=(
                "Notify affected personnel, shut down the unit using the normal stop "
                "procedure, isolate all energy sources at the primary disconnect, apply lock "
                "and tag to the primary disconnect, and verify zero energy state before "
                "beginning work."
            ),
            relevant_doc_ids=["technical"],
            domain="technical",
        ),
        BenchmarkQuery(
            query_id="research-imbalanced-metric",
            query="which metric besides accuracy should be reported for imbalanced classification",
            expected_answer=(
                "F1, PR-AUC, and Matthews correlation coefficient should be reported rather "
                "than accuracy alone under severe class imbalance."
            ),
            relevant_doc_ids=["research"],
            domain="research",
        ),
        BenchmarkQuery(
            query_id="financial-top-risk-factor",
            query="what was the top risk factor reported last quarter",
            expected_answer=(
                "Foreign currency exposure was the top risk factor reported in 2025-Q3."
            ),
            relevant_doc_ids=["financial"],
            domain="financial",
        ),
    ]


class BenchmarkDataset:
    def __init__(self, queries: list[BenchmarkQuery]) -> None:
        self.queries = queries

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([json.loads(q.model_dump_json()) for q in self.queries], indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkDataset:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([BenchmarkQuery(**q) for q in raw])

    @classmethod
    def default_sample_corpus(cls) -> BenchmarkDataset:
        return cls(default_sample_corpus_benchmark())
