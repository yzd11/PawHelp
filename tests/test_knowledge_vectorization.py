"""Recovery behavior at the MySQL-to-Milvus vectorization seam."""

from types import SimpleNamespace

import pytest

from app.knowledge.embedding import VECTOR_DIMENSION
from app.knowledge.vectorization import (
    rebuild_collection,
    reconcile_missing_vectors,
    vectorize_pending,
)


class InMemoryPendingKnowledge:
    def __init__(self) -> None:
        self.rows = [
            SimpleNamespace(
                id=index,
                category="配送服务",
                questions="配送费用如何计算？",
                answer=f"运费说明 {index}",
                section_path="配送服务 / 运费",
                content_type="faq",
                vectorize_status="pending",
                vector_id=None,
            )
            for index in range(1, 4)
        ]

    async def list_pending(self, *, limit: int):
        return [row for row in self.rows if row.vectorize_status == "pending"][:limit]

    async def mark_vectorized(self, rows) -> None:
        for row in rows:
            row.vector_id = str(row.id)
            row.vectorize_status = "done"

    async def list_after_id(self, *, after_id: int, limit: int):
        return [row for row in self.rows if row.id > after_id][:limit]

    async def list_vectorized_after_id(self, *, after_id: int, limit: int):
        return [
            row
            for row in self.rows
            if row.id > after_id and row.vectorize_status == "done"
        ][:limit]

    async def mark_pending(self, rows) -> None:
        for row in rows:
            row.vector_id = None
            row.vectorize_status = "pending"


class FixedEmbedder:
    def embed(self, texts):
        return [[float(index)] * VECTOR_DIMENSION for index, _ in enumerate(texts)]


class InterruptingStore:
    def __init__(self) -> None:
        self.rows: dict[str, list[float]] = {}
        self.calls = 0
        self.fail_on_call: int | None = 2

    def ensure_collection(self) -> None:
        return None

    def upsert(self, ids, vectors, texts, metadata) -> None:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated Milvus interruption")
        self.rows.update(zip(ids, vectors, strict=True))

    def existing_ids(self, ids):
        return set(ids) & set(self.rows)


@pytest.mark.asyncio
async def test_interrupted_vectorization_resumes_pending_rows_without_duplicates() -> None:
    repository = InMemoryPendingKnowledge()
    store = InterruptingStore()
    commits = 0

    async def commit() -> None:
        nonlocal commits
        commits += 1

    with pytest.raises(RuntimeError, match="simulated Milvus interruption"):
        await vectorize_pending(
            repository,
            FixedEmbedder(),
            store,
            commit,
            batch_size=1,
        )

    assert [row.vectorize_status for row in repository.rows] == ["done", "pending", "pending"]

    store.fail_on_call = None
    completed = await vectorize_pending(
        repository,
        FixedEmbedder(),
        store,
        commit,
        batch_size=1,
    )

    assert completed == 2
    assert commits == 3
    assert {row.vector_id for row in repository.rows} == {"1", "2", "3"}
    assert set(store.rows) == {"1", "2", "3"}


@pytest.mark.asyncio
async def test_collection_rebuild_includes_rows_already_marked_done() -> None:
    repository = InMemoryPendingKnowledge()
    repository.rows[0].vectorize_status = "done"
    repository.rows[0].vector_id = "1"
    store = InterruptingStore()
    store.fail_on_call = None

    completed = await rebuild_collection(
        repository,
        FixedEmbedder(),
        store,
        batch_size=2,
    )

    assert completed == 3
    assert set(store.rows) == {"1", "2", "3"}


@pytest.mark.asyncio
async def test_reconciliation_marks_done_rows_missing_from_milvus_pending() -> None:
    repository = InMemoryPendingKnowledge()
    for row in repository.rows[:2]:
        row.vectorize_status = "done"
        row.vector_id = str(row.id)
    store = InterruptingStore()
    store.fail_on_call = None
    store.rows["2"] = [0.0] * VECTOR_DIMENSION
    commits = 0

    async def commit() -> None:
        nonlocal commits
        commits += 1

    repaired = await reconcile_missing_vectors(
        repository,
        store,
        commit,
        batch_size=2,
    )

    assert repaired == 1
    assert [row.vectorize_status for row in repository.rows] == [
        "pending",
        "done",
        "pending",
    ]
    assert commits == 1
