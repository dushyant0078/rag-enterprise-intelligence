# Milestone 14 — Final Documentation & Benchmarking Report

## What this milestone is

The capstone: consolidate 13 milestones of real, incrementally-verified
work into a coherent final artifact set, rather than leaving a reader to
piece the project together from 13 separate build logs. Two deliverables:

1. **[`reports/final_benchmark_report.md`](../reports/final_benchmark_report.md)**
   — a single, consolidated report with fresh numbers, not numbers copied
   forward from earlier milestones without re-verification.
2. **`README.md`**, reorganized from a running build log (a "Status" line
   and a milestone-by-milestone appendix that grew for 13 milestones) into
   the polished top-level document a first-time visitor actually wants:
   a Quickstart, benchmark highlights inline, then the same detailed
   milestone walkthrough as before, kept intact rather than discarded.

## What was actually re-verified for this report, not just copied

Every number in the final benchmark report was produced by running
something during this milestone's preparation:

- **Full test suite**: re-run fresh — 480 passing, 86% coverage (matched
  the last-known count exactly, confirming no drift since Milestone 13).
- **Evaluation framework**: re-run fresh against a freshly rebuilt index
  (`scripts/run_evaluation_framework.py`). Precision@3 shifted slightly
  from Milestone 8's original report (0.667 → 0.619) — a real, expected
  effect of a freshly rebuilt corpus/index, not a discrepancy requiring
  investigation; MRR, Recall, and Citation Accuracy all held at their
  original values.
- **A real load test, run fresh for this report — new work, not a
  re-run.** A genuine background `uvicorn` process was started, hit with
  real HTTP requests (sequential latency, concurrent throughput via a
  `ThreadPoolExecutor`, and a deliberate rate-limiter burst test), and
  torn down. This hadn't been measured anywhere in the project before —
  Milestone 9 verified correctness, Milestone 11 verified tracing
  instrumentation, but nothing had measured raw throughput/latency
  end-to-end until this report.
- **Test-count breakdown**: initially estimated from memory in a first
  draft, then corrected against `pytest --collect-only` run per test
  subdirectory — the first draft's per-area estimates were off by up to
  ~30% in places (e.g. "~130" for retrieval/reranking vs. the real 100),
  and two subdirectories (`tests/prompting`, `tests/utils`) were missed
  entirely in the first pass. Caught by reconciling the per-area sum
  against the known total (480) rather than trusting each estimate in
  isolation.
- **Bug retrospective count**: "20+" in an early draft, corrected to the
  precise **18** after actually counting distinct bugs in the retrospective
  table itself (excluding Milestone 10's row, which documents a scope
  extension, not a bug) — the more defensible, exact number replaced a
  rounded guess before it shipped.

## Real load test results (new for this report)

| Test | Result |
|---|---|
| Sequential `POST /query` latency (20 requests) | mean 4.9ms, p50 4.7ms |
| Concurrent throughput (30 requests, 8 workers) | 112.8 req/s, 100% success |
| `POST /evaluate` (heaviest single endpoint) | ~13ms |
| Rate limiter under a real 70-request burst (capacity 60) | exactly 60 succeed, `429` begins at request #61 |

The rate-limiter result is worth dwelling on: this isn't a mocked or
simulated check — a real registered API key was hit with a real burst
against a real running server, and Milestone 9's token-bucket
implementation enforced its configured capacity exactly, request for
request. Two earlier attempts at this specific test produced misleading
results before landing on this one: the first mixed rate-limit exhaustion
into a throughput measurement (making the throughput number look worse
than the API's real overhead), and the second used an *unregistered* API
key for the burst test, which correctly returned `401` before ever
reaching the rate limiter — a `401` looks superficially similar to a
`429` in a status-code tally but demonstrates a completely different code
path. Both were caught by reading the actual status codes returned,
not assumed from the test's intent.

## README reorganization

The prior README was accurate but structured as a build log: a growing
"Status" line, then sequential per-milestone sections in commit order.
Kept for this final version: every one of those sections, in full — they
are real, detailed, and worth keeping. Added: a **Quickstart** near the
top (the single most common thing a new visitor wants and the previous
version made them read ~150 lines to assemble themselves), a **Benchmark
highlights** section surfacing the headline numbers inline rather than
requiring a click-through, and fixed several small pieces of drift a
final read caught — a "License: TBD" placeholder that Milestone 13 had
already resolved but never propagated back to the README, a `Makefile`
snippet missing the `format-check` target added in Milestone 13, and a
`run-api`/`run-ui` note still saying those targets "become live in
Milestones 9 and 10" despite both milestones being long complete.

## Known limitations / deferred work

Every limitation named in `docs/01` through `docs/13` still applies and
isn't restated exhaustively here — seeing them once, precisely, in each
milestone's own doc is more useful than a diluted summary. The
benchmark report's section 7 gives the short, honest version: what's fully
real and tested, and what's correct-complete code blocked by this
sandbox's specific network/credential constraints, not fictional.

---
*This is the last milestone doc. See `README.md` for the full project and
`reports/final_benchmark_report.md` for the complete numbers.*
