from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_request_id, get_state, require_api_key
from api.schemas import EvaluateRequest, EvaluateResponse
from api.state import AppState
from src.evaluation.benchmark_dataset import BenchmarkDataset
from src.evaluation.evaluator import RAGEvaluator
from src.security.api_auth import ApiKeyRecord
from src.security.audit_log import log_request

router = APIRouter(tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(
    body: EvaluateRequest,
    state: AppState = Depends(get_state),
    auth: ApiKeyRecord = Depends(require_api_key),
    request_id: str = Depends(get_request_id),
) -> EvaluateResponse:
    """Runs the Milestone 8 evaluation framework against the built-in
    labeled benchmark, on demand. Real work (a full benchmark pass through
    the pipeline), not a canned response -- expect this to take a few
    seconds even with the offline backends.
    """
    evaluator = RAGEvaluator(state.rag_pipeline, k=body.k)
    report = evaluator.evaluate(BenchmarkDataset.default_sample_corpus())

    log_request(
        request_id=request_id, user_id=auth.user_id, endpoint="/evaluate",
        extra={"benchmark_size": report.benchmark_size, "grounded_rate": report.grounded_rate},
    )  # fmt: skip

    return EvaluateResponse(
        benchmark_size=report.benchmark_size,
        k=report.k,
        mean_precision_at_k=report.mean_precision_at_k,
        mean_recall_at_k=report.mean_recall_at_k,
        mean_mrr=report.mean_mrr,
        mean_ndcg_at_k=report.mean_ndcg_at_k,
        mean_citation_accuracy=report.mean_citation_accuracy,
        mean_faithfulness=report.mean_faithfulness,
        mean_answer_correctness=report.mean_answer_correctness,
        grounded_rate=report.grounded_rate,
        latency_p50_ms=report.latency_p50_ms,
        latency_p95_ms=report.latency_p95_ms,
    )
