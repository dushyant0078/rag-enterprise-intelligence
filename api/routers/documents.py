from __future__ import annotations

import glob
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import get_request_id, get_state, require_api_key
from api.schemas import (
    DocumentDetailResponse,
    IngestionResultOut,
    ReindexRequest,
    ReindexResponse,
)
from api.state import AppState
from src.parsing.registry import UnsupportedFormatError, supported_extensions
from src.parsing.schemas import ParsedDocument
from src.security.api_auth import ApiKeyRecord
from src.security.audit_log import log_request

router = APIRouter(tags=["documents"])

_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB -- generous for a policy/contract PDF, not unbounded


@router.post("/upload", response_model=IngestionResultOut)
async def upload(
    file: UploadFile = File(...),
    domain: str = Form(...),
    state: AppState = Depends(get_state),
    auth: ApiKeyRecord = Depends(require_api_key),
    request_id: str = Depends(get_request_id),
) -> IngestionResultOut:
    """Ingests (parses + hashes + version-tracks) the uploaded file -- does
    NOT chunk/embed/index it. Call /reindex afterward to make it searchable;
    kept as two explicit steps since re-embedding the whole corpus per
    upload would be wasteful at any real scale (batch uploads, then one
    reindex, is the intended flow).
    """
    if auth.allowed_domains is not None and domain not in auth.allowed_domains:
        raise HTTPException(status_code=403, detail=f"API key not authorized for domain '{domain}'")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
    filename = file.filename  # narrowed to `str` here, unlike file.filename itself

    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in supported_extensions():
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {supported_extensions()}",
        )

    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit"
        )

    dest_dir = state.raw_upload_dir / domain
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_bytes(contents)

    try:
        result = state.ingestion_pipeline.ingest_file(dest_path, domain)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    log_request(
        request_id=request_id, user_id=auth.user_id, endpoint="/upload",
        extra={"doc_id": result.doc_id, "status": result.status.value, "domain": domain},
    )  # fmt: skip

    return IngestionResultOut(
        doc_id=result.doc_id, status=result.status.value, domain=domain, message=result.message
    )


@router.post("/reindex", response_model=ReindexResponse)
def reindex(
    body: ReindexRequest,
    state: AppState = Depends(get_state),
    auth: ApiKeyRecord = Depends(require_api_key),
    request_id: str = Depends(get_request_id),
) -> ReindexResponse:
    if auth.allowed_domains is not None:
        raise HTTPException(
            status_code=403, detail="Reindexing requires an unrestricted (admin) API key"
        )

    try:
        chunk_count = state.reindex(body.domain)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    log_request(
        request_id=request_id, user_id=auth.user_id, endpoint="/reindex",
        extra={"chunks_indexed": chunk_count},
    )  # fmt: skip

    return ReindexResponse(status="ok", chunks_indexed=chunk_count, domains=sorted(state.domains))


@router.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
def get_document(
    doc_id: str,
    state: AppState = Depends(get_state),
    auth: ApiKeyRecord = Depends(require_api_key),
) -> DocumentDetailResponse:
    """Full parsed content + metadata for one ingested document -- the
    Source Viewer page's data source (Milestone 10). RBAC-checked against
    the document's domain, not just its existence, so a scoped key can't
    read a document outside its allow-list even by guessing a real doc_id.
    """
    processed_matches = glob.glob(
        str(state.processed_dir / "**" / f"{doc_id}.json"), recursive=True
    )
    if not processed_matches:
        raise HTTPException(status_code=404, detail=f"No ingested document with id '{doc_id}'")

    domain = doc_id.split("__")[0]
    if auth.allowed_domains is not None and domain not in auth.allowed_domains:
        raise HTTPException(status_code=403, detail=f"API key not authorized for domain '{domain}'")

    with open(processed_matches[0]) as f:
        parsed = ParsedDocument(**json.loads(f.read()))

    metadata_matches = glob.glob(str(state.metadata_dir / "**" / f"{doc_id}.json"), recursive=True)
    version, language, used_ocr = 1, None, False
    if metadata_matches:
        with open(metadata_matches[0]) as f:
            meta = json.loads(f.read())
        version = meta.get("version", 1)
        language = meta.get("language")
        used_ocr = meta.get("used_ocr", False)

    return DocumentDetailResponse(
        doc_id=doc_id,
        domain=domain,
        format=parsed.format,
        page_count=parsed.page_count,
        language=language,
        used_ocr=used_ocr,
        version=version,
        full_text=parsed.full_text,
        tables=[t.model_dump() for t in parsed.tables],
    )
