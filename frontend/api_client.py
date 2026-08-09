"""HTTP client for the FastAPI backend (Milestone 9). Every Streamlit page
goes through this rather than calling the pipeline directly -- matching the
architecture diagram (User -> Streamlit UI -> FastAPI Gateway -> ...), and
keeping the frontend a real client of the same API any other integration
would use, not a shortcut around it.
"""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 60  # /evaluate runs a full benchmark pass -- generous on purpose


def resolve_default_base_url() -> str:
    """The base URL to use before the user has typed anything into the
    sidebar -- RAG_API_BASE_URL if set (how docker-compose.yml and
    k8s/frontend-deployment.yaml point the frontend at the API service by
    name rather than localhost), else DEFAULT_BASE_URL. Callers that build
    session_state's initial value (frontend/app.py, frontend/state.py) use
    this instead of the bare DEFAULT_BASE_URL constant -- passing that
    constant straight to RAGAPIClient(base_url=...) short-circuits the
    env-var fallback inside __init__ below, since a non-None base_url
    always wins there. Caught by reasoning through the real
    docker-compose.yml deployment path, not by a unit test (every existing
    frontend test explicitly sets session_state, bypassing this default
    entirely).
    """
    return os.environ.get("RAG_API_BASE_URL", DEFAULT_BASE_URL)


class APIClientError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API error {status_code}: {detail}")


def _extract_detail(response: requests.Response) -> str:
    try:
        body = response.json()
        return str(body.get("detail", response.text))
    except ValueError:
        return response.text


class RAGAPIClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("RAG_API_BASE_URL", DEFAULT_BASE_URL)).rstrip(
            "/"
        )
        self.api_key = api_key or ""
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self._session.request(
            method, url, headers=self._headers(), timeout=DEFAULT_TIMEOUT_SECONDS, **kwargs
        )
        if response.status_code >= 400:
            raise APIClientError(response.status_code, _extract_detail(response))
        return response.json()  # type: ignore[no-any-return]

    # -- system -------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/model-info")

    def metrics_text(self) -> str:
        response = self._session.get(f"{self.base_url}/metrics", timeout=DEFAULT_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text

    # -- RAG ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        session_id: str | None = None,
        domain: str | None = None,
        template: str = "qa",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/chat",
            json={"query": query, "session_id": session_id, "domain": domain, "template": template},
        )

    def query(
        self, query: str, domain: str | None = None, top_k: int = 5, template: str = "qa"
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/query",
            json={"query": query, "domain": domain, "top_k": top_k, "template": template},
        )

    def search(
        self, query: str, domain: str | None = None, top_k: int = 10, rerank: bool = False
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/search",
            json={"query": query, "domain": domain, "top_k": top_k, "rerank": rerank},
        )

    def summarize(
        self, text: str | None = None, doc_id: str | None = None, length_hint: str = "3-4 sentences"
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/summarize", json={"text": text, "doc_id": doc_id, "length_hint": length_hint}
        )

    # -- documents --------------------------------------------------------

    def upload(self, filename: str, file_bytes: bytes, domain: str) -> dict[str, Any]:
        url = f"{self.base_url}/upload"
        response = self._session.post(
            url,
            headers=self._headers(),
            files={"file": (filename, file_bytes)},
            data={"domain": domain},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise APIClientError(response.status_code, _extract_detail(response))
        return response.json()  # type: ignore[no-any-return]

    def reindex(self, domain: str | None = None) -> dict[str, Any]:
        return self._request("POST", "/reindex", json={"domain": domain})

    def get_document(self, doc_id: str) -> dict[str, Any]:
        return self._request("GET", f"/documents/{doc_id}")

    # -- sessions -----------------------------------------------------------

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/sessions/{session_id}")

    # -- evaluation -----------------------------------------------------------

    def evaluate(self, k: int = 5) -> dict[str, Any]:
        return self._request("POST", "/evaluate", json={"k": k})

    # -- feedback & monitoring (Milestone 11) --------------------------------

    def submit_feedback(
        self,
        session_id: str,
        rating: str,
        comment: str | None = None,
        query: str | None = None,
        answer_excerpt: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/feedback",
            json={
                "session_id": session_id,
                "rating": rating,
                "comment": comment,
                "query": query,
                "answer_excerpt": answer_excerpt,
            },
        )

    def monitoring_summary(self) -> dict[str, Any]:
        return self._request("GET", "/monitoring/summary")
