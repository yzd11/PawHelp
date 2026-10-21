# Changelog

## 0.1.0 — Unreleased

### Conversation

- LangGraph workflow: reference resolution → intent classification →
  deterministic routing → knowledge retrieval or tool agent → SSE stream.
- Order selection and ticket confirmation as graph interrupts, resumable after a
  restart through the SQLite checkpointer.
- Layered session context with a token budget and asynchronous summaries.
- Guards against rewrites that invent or replace an explicit order number.
- Sidebar titles generated once in the background; greetings are skipped and
  short questions are used as-is, falling back to the first question on failure.

### Knowledge and retrieval

- Markdown ingestion into structured chunks; BGE-M3 dense + Milvus BM25 hybrid
  recall, RRF fusion, BGE reranking and a calibrated evidence gate with citations.
- 500-question retrieval benchmark (410 answerable, 90 out-of-scope) comparing
  four strategies by query type.
- Knowledge flywheel: low-confidence questions are normalized, deduplicated,
  reviewed by an operator and written back to the knowledge base.
- Review page shows evidence-gate refusals as a sentence with score and threshold,
  not the stored confidence JSON.
- Invoice requests are accepted within 30 days of confirmed receipt (was 90); the
  benchmark rows and the evidence-gate calibration were updated with it. The
  96 labelled cases were re-collected locally and re-reviewed; the threshold (0.127)
  and its independent validation result are unchanged.
- The four product manuals name the series they cover in their title, so every
  section chunk carries it; a model-specific question such as "MH-FD30 的录音呼唤
  最长能录多久" is no longer refused for evidence that does not name the model. The
  separate "本篇适用于" intro chunks were dropped (101 → 97 document chunks) and the
  calibration was re-collected; the threshold (0.127) still passes validation.
- Every write outside a Markdown rebuild is checked against existing knowledge
  first: a model compares the entry with its nearest matches and refuses a
  duplicate or a contradiction (manual entries and edits get 409, approved
  reviews stay pending, mined QA is staged as discarded with its reason, migration
  0012), naming the existing entry by its ID with a one-sentence reason in Chinese.
  The check fails closed: if it cannot run, nothing is written. Seven old
  manual and mined entries that contradicted or repeated the shipping, return and
  promotion documents were removed.
- A Markdown rebuild rechecks the manual, approved and mined entries it keeps
  against the rebuilt documents and lists any that are now duplicates or
  contradictions on /kb; it never deletes them itself.
- The evidence-gate calibration is collected on the served path (model rewrite,
  question split, whole knowledge base) instead of the raw query. Its fingerprint
  covers the documents, the chunking/retrieval code, the rewrite prompt, the models
  and the retrieval settings, so changing any of them leaves the gate uncalibrated;
  /kb warns when manual or mined entries changed after calibration. The 96 labels
  were re-reviewed on the new evidence and are unchanged; the threshold rose from
  0.127 to 0.174 and independent validation passes (0/24 unsafe, recall 0.95).
- Optional TypeSafe Jev (`jev-1.14.0`) evidence-sufficiency judge. `EVIDENCE_JUDGE=shadow`
  logs Jev's P(yes) next to the LLM verdict; `jev` lets Jev accept or refuse outside a
  calibrated band and hands the band, Jev errors and calibration mismatches to the LLM.
  The default stays `llm` and needs no key. On the archived 500-question run (498 scored)
  the thresholds chosen on training cases are P ≤ 0.05 refuse / P ≥ 0.60 accept; on the
  served validation path Jev matches the LLM (0 false refusals, 0 false accepts) and decides
  93.5% of cases itself, AUC 0.999. A full served run in `jev` mode had Jev decide 372 of the
  403 evidence checks (92.3%) with no false refusal or false accept; false refusals went
  18 → 17 of 410, out-of-scope refusals stayed 90/90, and the remaining over-strict refusals
  all came from the band handed to the LLM.

### Tools

- Tool registry with JSON-schema validation, timeouts, read-only retries,
  confirmation-gated writes and an audit log.
- Logistics and after-sales demo services over MCP with an allow-list policy.
- The order prefetched for a turn reaches the agent as its `query_order` call and
  result, so a logistics turn no longer queries the same order twice (one model
  round and one tool call fewer).

### Operations

- `scripts.start` brings up containers, migrations, MCP servers, the topic classifier
  and the app in one command, and stops them together on Ctrl+C.
- Only the served app writes `log/app.log` and the SQLite conversation checkpoints;
  apps built in tests keep checkpoints in memory, so test runs no longer mix fake
  conversations into the real context log. The log handler is detached at shutdown.
- Faster cold start: retrieval models warm up in the background at startup
  (first chat 37.5s → 13.2s) and the text-splitter import no longer
  loads sentence-transformers (import 27s → 7s).
- Admin pages for knowledge, RAG evaluation, review, cost/observability and topics.
- Self-hosted Langfuse export; token usage and estimated cost per intent.
- Offline 17-label topic classifier (RoBERTa-wwm-ext) for the question pool.

### Quality

- Offline regression suite (fake models, in-memory checkpoints) in CI; live-MySQL
  integration tests run against an explicit disposable database.
