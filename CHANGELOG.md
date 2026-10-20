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
  0012). The check fails closed: if it cannot run, nothing is written. Seven old
  manual and mined entries that contradicted or repeated the shipping, return and
  promotion documents were removed.

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
