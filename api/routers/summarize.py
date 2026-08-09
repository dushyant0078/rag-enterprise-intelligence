from __future__ import annotations

import glob
import json

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_request_id, get_state, require_api_key
from api.schemas import SummarizeRequest, SummarizeResponse
from api.state import AppState
from src.parsing.schemas import ParsedDocument
from src.prompting.templates.summarization import SUMMARIZATION_TEMPLATE
from src.security.api_auth import ApiKeyRecord
from src.security.audit_log import log_request

router = APIRouter(tags=["summarize"])


def _load_document_text(processed_dir, doc_id: str) -> str | None:
    matches = glob.glob(str(processed_dir / "**" / f"{doc_id}.json"), recursive=True)
    if not matches:
        return None
    with open(matches[0]) as f:
        data = json.loads(f.read())
    return ParsedDocument(**data).full_text


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(
    body: SummarizeRequest,
    state: AppState = Depends(get_state),
    auth: ApiKeyRecord = Depends(require_api_key),
    request_id: str = Depends(get_request_id),
) -> SummarizeResponse:
    if not body.text and not body.doc_id:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'doc_id'")

    content: str
    source: str
    if body.text:
        content, source = body.text, "text"
    else:
        assert body.doc_id is not None  # guaranteed by the check above -- narrows for mypy too
        loaded = _load_document_text(state.processed_dir, body.doc_id)
        if loaded is None:
            raise HTTPException(
                status_code=404, detail=f"No ingested document with id '{body.doc_id}'"
            )
        content, source = loaded, f"doc_id:{body.doc_id}"

    messages = SUMMARIZATION_TEMPLATE.render(
        context=f"[Source: {source}]\n{content}", length_hint=body.length_hint
    )
    response = state.llm.generate(messages, max_tokens=400)

    log_request(
        request_id=request_id, user_id=auth.user_id, endpoint="/summarize", extra={"source": source}
    )

    return SummarizeResponse(summary=response.content, source=source)
