# Milestone 1 — Business Understanding & Problem Framing

## 1. Project Overview

**Project:** LLM-powered RAG Enterprise Document Intelligence System
**Repo (working name):** `RAG-Enterprise-Intelligence`

Large enterprises sit on document collections that are too large for any single employee to read, too unstructured for keyword search to serve well, and too high-stakes for a naive LLM chatbot that hallucinates. This platform is a **Retrieval-Augmented Generation system purpose-built for regulated, document-heavy enterprises** — banking, insurance, legal, HR, and technical operations — where every answer must be traceable to a source passage and every retrieval must respect access boundaries.

This is not a "chat with your PDF" demo. The design target is a system where:

- Every claim in an answer is backed by a citation (document, page, chunk ID).
- Retrieval quality is measured, not assumed (Precision@K, faithfulness, hallucination rate).
- Multiple LLM and vector-store backends are swappable via configuration, not code changes.
- Ingestion, retrieval, and generation are independently observable and testable.

## 2. Business Problem

Knowledge workers in large organizations lose significant time locating authoritative answers buried in policy manuals, contracts, filings, and SOPs. The cost shows up three ways:

1. **Search cost** — keyword search over document repositories returns pages, not answers, forcing manual reading.
2. **Risk cost** — an employee acting on an outdated or misremembered policy (e.g., a stale KYC threshold, an expired SOP revision) creates compliance exposure.
3. **Escalation cost** — ambiguous questions get routed to subject-matter experts (compliance officers, legal counsel) who spend time answering questions a well-grounded system could resolve directly, citing the source.

A RAG system addresses this only if it is **trustworthy**: it must decline to answer when it lacks support in the retrieved context, rather than filling gaps with plausible-sounding text. That constraint — bias toward "I don't have enough information" over confident fabrication — is a first-class design requirement here, not an afterthought.

## 3. Representative Personas & Use Cases

| Persona | Domain | Example Need |
|---|---|---|
| Compliance Analyst | Banking policy | "What is the current threshold for enhanced due diligence on wire transfers over $10K?" |
| Claims Adjuster | Insurance | "Does policy form HO-3 cover water damage from a burst pipe if the property was unoccupied for 45 days?" |
| Legal Associate | Contracts | "Which of our vendor MSAs have an auto-renewal clause with less than 60 days' notice?" |
| HR Business Partner | HR policy | "What is the parental leave policy for employees in their first year, per the 2025 handbook revision?" |
| Equity Research Analyst | Financial reports (10-K) | "What did the company cite as its top three risk factors in the most recent 10-K, compared to last year?" |
| Field Engineer | SOPs / technical docs | "What is the lockout-tagout procedure for servicing Model X-200 before line power is restored?" |
| Research Scientist | ArXiv papers | "Summarize the evaluation methodology used in this paper and how it compares to RAGAS-style metrics." |

## 4. Document Types & Knowledge Domains

| Domain | Document Types | Source (this project) |
|---|---|---|
| Banking / Financial Compliance | Policy PDFs, SEC 10-K filings | SEC EDGAR 10-K corpus |
| Insurance | Policy wording documents, claims forms | Public insurance policy PDFs / synthetic policy set |
| Legal | Contracts, MSAs, NDAs | CUAD (Contract Understanding Atticus Dataset) or public contract samples |
| HR | Employee handbooks, policy manuals | Public/sample HR policy PDFs |
| Research | Academic papers | ArXiv API (cs.CL / cs.LG subset) |
| Technical Ops | SOPs, runbooks, technical documentation | Public technical documentation repos (e.g., open-source project docs) |
| Scanned / Legacy | Scanned forms, older filings | Kaggle document-intelligence datasets (scanned + OCR ground truth) |

File formats in scope: **PDF (text + scanned/OCR), DOCX, PPTX, HTML, Markdown, TXT, CSV**.

## 5. Expected User Query Patterns

Query behavior differs meaningfully by domain, which drives retrieval and prompting design (Milestones 6–7):

- **Lookup queries** — single fact, single source ("What is the notice period in Section 8.2?"). Needs precise chunking and high-precision retrieval.
- **Comparative queries** — span multiple documents or multiple versions ("How did the risk factors change year over year?"). Needs multi-document retrieval and a synthesis-aware prompt.
- **Conditional/eligibility queries** — require chained reasoning over clauses ("Is this claim covered if X and Y but not Z?"). Highest hallucination risk; requires strict grounding and explicit "insufficient context" fallback.
- **Summarization queries** — "Summarize this SOP" or "Summarize the methodology section." Needs full-document or section-level context assembly, not just top-K chunks.
- **Exploratory queries** — "What policies exist about remote work?" Needs broad recall before precision (different retrieval strategy than lookup).

## 6. Security Constraints

- **Role-based access control (RBAC):** retrieval must be filtered by the requesting user's entitlements (e.g., an HR document set is not visible to a field engineer's query) — enforced at the retrieval layer via metadata filtering, not just at the UI.
- **PII handling:** ingestion pipeline must detect and mask/flag PII (names, SSNs, account numbers) before it can appear in logs, traces, or cached embeddings where policy requires it.
- **Authentication & authorization:** API gateway requires authenticated requests; no anonymous access to `/query` or `/upload`.
- **Audit logging:** every query, retrieved chunk set, and generated answer is logged with a request ID for after-the-fact compliance review.
- **Secrets management:** API keys and vector-store credentials are never hardcoded; sourced from environment/secret manager, out of version control.
- **Prompt injection resistance:** content pulled from ingested documents is treated as data, not instructions — the system prompt must not be overridable by text embedded inside a retrieved chunk.

## 7. Compliance Risks

- Serving an answer based on a **superseded policy version** because ingestion didn't track document versioning/effective dates.
- **Cross-tenant leakage** in a multi-tenant deployment if namespace isolation in the vector store is misconfigured.
- **Unlogged access** to sensitive documents, which defeats audit requirements common in banking/insurance (e.g., SOX, GDPR-adjacent obligations).
- Citations that **misattribute** a claim to the wrong document/page, which is itself a compliance failure even if the underlying claim is correct.

## 8. Hallucination Risks & Mitigations (design intent — implemented in later milestones)

| Risk | Mitigation |
|---|---|
| Answering beyond retrieved context | Response Validator step checks answer claims against retrieved passages before returning; unsupported claims trigger an "insufficient evidence" response instead of a guess |
| Stale/conflicting document versions retrieved together | Metadata-aware retrieval filters by effective date; conflicting sources surfaced explicitly rather than silently merged |
| Over-confident tone masking low retrieval confidence | Confidence score attached to every answer, derived from retrieval/rerank scores, surfaced in the UI |
| Chunking that severs clause context (e.g., splits a contract clause from its exception) | Table-aware and semantic chunking strategies evaluated specifically against this failure mode (Milestone 4) |

## 9. Retrieval Challenges

- **Long, clause-structured legal/contract text** where meaning depends on cross-references within the same document.
- **Numeric- and table-heavy financial filings** where naive text chunking destroys row/column relationships.
- **Scanned documents** where OCR noise degrades embedding quality and requires confidence-aware handling.
- **Domain vocabulary drift** — the same term ("coverage," "termination") means different things across insurance vs. HR vs. legal corpora, which argues for domain-aware or namespace-isolated retrieval rather than one global index.
- **Recency vs. relevance trade-off** — a highly similar but outdated clause should usually lose to a less textually similar but current one.

## 10. Success Metrics (targets refined in Milestone 8 — Evaluation Framework)

- Faithfulness / groundedness of answers to retrieved context (RAGAS faithfulness score)
- Citation accuracy (cited chunk actually supports the claim)
- Retrieval Precision@K / Recall@K per domain
- End-to-end answer latency (p50/p95)
- Hallucination rate on an adversarial eval set (questions with no answer in the corpus)

## 11. Explicit Non-Goals (v1)

- Not a general-purpose chatbot — no open-domain knowledge outside the ingested corpus.
- Not a document editing/generation tool — read and answer, not draft contracts.
- Not real-time-streaming ingestion in v1 — batch/incremental ingestion is the initial scope; a streaming ingestion path is noted as a future improvement.

---
*Next: Milestone 1 continues in `02_architecture.md` — system architecture and diagrams.*
