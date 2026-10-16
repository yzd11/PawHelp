"""Exercise real adapters at the shared-model seam without crashing the test process."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import time

import pytest

from app.database.models import KnowledgeChunk
from app.knowledge.embedding import BgeM3Embedder
from app.knowledge.reranking import BgeReranker


class SharedModelProbe:
    """Like FlagEmbedding, an inference call mutates shared model state."""

    def __init__(self):
        self.lock = Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def call(self):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.04)  # release GIL; overlapping calls exercise the actual race
        finally:
            with self.lock:
                self.active -= 1

    def compute_score(self, pairs, **kwargs):
        self.call()
        return [0.9] * len(pairs)

    def encode(self, texts, **kwargs):
        self.call()

        class Vectors:
            def tolist(self):
                return [[0.0] * 1024 for _ in texts]

        return {"dense_vecs": Vectors()}


@pytest.mark.parametrize("kind", ["embed", "rerank"])
def test_shared_model_inference_is_serialized_across_parallel_queries(kind):
    probe = SharedModelProbe()
    adapter = BgeM3Embedder("test") if kind == "embed" else BgeReranker("test")
    adapter._model = probe
    ready = Barrier(3)

    def execute():
        ready.wait(timeout=2)
        if kind == "embed":
            result = adapter.embed(["这个能退吗"])
            assert len(result[0]) == 1024
        else:
            chunks = [KnowledgeChunk(id=1, questions="退货条件", answer="签收七天内", category="退款退货")]
            result = adapter.rerank("这个能退吗", chunks, limit=1)
            assert result[0].chunk.id == 1
            assert result[0].score == 0.9

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(execute) for _ in range(3)]
        for future in futures:
            future.result(timeout=5)
    assert probe.calls == 3
    assert probe.peak == 1, "shared mutable model inference calls overlapped"
