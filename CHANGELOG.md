# Changelog

## 0.1.0 — Unreleased

### Conversation

- LangGraph workflow: reference resolution → intent classification →
  deterministic routing → knowledge retrieval or tool agent → SSE stream.
- Order selection and ticket confirmation as graph interrupts, resumable after a
  restart through the SQLite checkpointer.
- Layered session context with a token budget and asynchronous summaries.
- Guards against rewrites that invent or replace an explicit order number.

### Knowledge and retrieval

- Markdown ingestion into structured chunks; BGE-M3 dense + Milvus BM25 hybrid
  recall, RRF fusion, BGE reranking and a calibrated evidence gate with citations.
- 300-question retrieval benchmark comparing four strategies by query type.
- Knowledge flywheel: low-confidence questions are normalized, deduplicated,
  reviewed by an operator and written back to the knowledge base.

### Tools

- Tool registry with JSON-schema validation, timeouts, read-only retries,
  confirmation-gated writes and an audit log.
- Logistics and after-sales demo services over MCP with an allow-list policy.

### Operations

- Admin pages for knowledge, RAG evaluation, review, cost/observability and topics.
- Self-hosted Langfuse export; token usage and estimated cost per intent.
- Offline 17-label topic classifier (RoBERTa-wwm-ext) for the question pool.

### Quality

- Offline regression suite (fake models, in-memory checkpoints) in CI; live-MySQL
  integration tests run against an explicit disposable database.
