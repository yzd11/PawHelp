"""Concurrent knowledge searches must not share a client across worker threads."""

import asyncio
import threading
import time

from app.knowledge.search import HybridKnowledgeSearch, RetrievalStrategy


class ThreadRecordingStore:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def search_bm25(self, query: str, *, limit: int, filter_expression: str):
        self.thread_ids.append(threading.get_ident())
        time.sleep(0.05)
        return []


def test_concurrent_bm25_calls_use_one_milvus_worker() -> None:
    store = ThreadRecordingStore()
    search = HybridKnowledgeSearch(None, None, store)  # type: ignore[arg-type]

    async def run() -> None:
        await asyncio.gather(
            search._recall(RetrievalStrategy.BM25, None, "邮费", ""),
            search._recall(RetrievalStrategy.BM25, None, "运费", ""),
        )

    asyncio.run(run())

    assert len(store.thread_ids) == 2
    assert len(set(store.thread_ids)) == 1
