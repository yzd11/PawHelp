"""Recoverable MySQL-to-Milvus knowledge vectorization flow."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from app.database.models import KnowledgeChunk
from app.knowledge.vector_store import milvus_call


class PendingKnowledge(Protocol):
    async def list_pending(self, *, limit: int) -> list[KnowledgeChunk]: ...

    async def mark_vectorized(self, rows: Sequence[KnowledgeChunk]) -> None: ...


class ReindexKnowledge(Protocol):
    async def list_after_id(
        self, *, after_id: int, limit: int
    ) -> list[KnowledgeChunk]: ...


class ReconcileKnowledge(Protocol):
    async def list_vectorized_after_id(
        self, *, after_id: int, limit: int
    ) -> list[KnowledgeChunk]: ...

    async def mark_pending(self, rows: Sequence[KnowledgeChunk]) -> None: ...


class DenseEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class KnowledgeVectorStore(Protocol):
    def ensure_collection(self) -> None: ...

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadata: Sequence[dict[str, str]],
    ) -> None: ...

    def existing_ids(self, ids: Sequence[str]) -> set[str]: ...


def vector_text(row: KnowledgeChunk) -> str:
    """Keep metadata out of the representation by contract."""

    return f"分类：{row.category}\n问题：{row.questions}\n答案：{row.answer}"


def _metadata(rows: Sequence[KnowledgeChunk]) -> list[dict[str, str]]:
    return [
        {
            "category": row.category,
            "content_type": row.content_type or "",
            "section_path": row.section_path or "",
        }
        for row in rows
    ]


async def vectorize_pending(
    repository: PendingKnowledge,
    embedder: DenseEmbedder,
    vector_store: KnowledgeVectorStore,
    commit: Callable[[], Awaitable[None]],
    *,
    batch_size: int,
) -> int:
    """Upsert pending rows in committed batches so a rerun resumes safely."""

    await milvus_call(vector_store.ensure_collection)
    completed = 0
    while rows := await repository.list_pending(limit=batch_size):
        ids = [str(row.id) for row in rows]
        texts = [vector_text(row) for row in rows]
        vectors = await asyncio.to_thread(
            embedder.embed,
            texts,
        )
        await milvus_call(vector_store.upsert, ids, vectors, texts, _metadata(rows))
        await repository.mark_vectorized(rows)
        await commit()
        completed += len(rows)
    return completed


async def reconcile_missing_vectors(
    repository: ReconcileKnowledge,
    vector_store: KnowledgeVectorStore,
    commit: Callable[[], Awaitable[None]],
    *,
    batch_size: int,
) -> int:
    """Turn false ``done`` rows back into recoverable ``pending`` rows."""

    repaired = 0
    after_id = 0
    while rows := await repository.list_vectorized_after_id(
        after_id=after_id,
        limit=batch_size,
    ):
        ids = [str(row.id) for row in rows]
        existing = await milvus_call(vector_store.existing_ids, ids)
        missing = [row for row in rows if str(row.id) not in existing]
        if missing:
            await repository.mark_pending(missing)
            await commit()
            repaired += len(missing)
        after_id = rows[-1].id
    return repaired


async def rebuild_collection(
    repository: ReindexKnowledge,
    embedder: DenseEmbedder,
    vector_store: KnowledgeVectorStore,
    *,
    batch_size: int,
) -> int:
    """Idempotently copy every MySQL chunk into a fresh Milvus collection.

    This intentionally ignores ``vectorize_status``: existing rows may already
    point at the legacy dense-only collection. Milvus upsert keeps reruns safe.
    """

    await milvus_call(vector_store.ensure_collection)
    completed = 0
    after_id = 0
    while rows := await repository.list_after_id(after_id=after_id, limit=batch_size):
        ids = [str(row.id) for row in rows]
        texts = [vector_text(row) for row in rows]
        vectors = await asyncio.to_thread(embedder.embed, texts)
        await milvus_call(vector_store.upsert, ids, vectors, texts, _metadata(rows))
        completed += len(rows)
        after_id = rows[-1].id
    return completed
