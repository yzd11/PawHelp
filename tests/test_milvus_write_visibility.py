"""A committed knowledge batch must be visible before MySQL marks it done."""

from app.knowledge.embedding import VECTOR_DIMENSION
from app.knowledge.vector_store import MilvusKnowledgeStore


class FlushGatedClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def upsert(self, *, collection_name: str, data: list[dict[str, object]]) -> None:
        assert collection_name == "knowledge_hybrid"
        assert len(data) == 1
        self.calls.append("upsert")

    def flush(self, *, collection_name: str) -> None:
        assert collection_name == "knowledge_hybrid"
        self.calls.append("flush")

    def has_collection(self, *, collection_name: str) -> bool:
        assert collection_name == "knowledge_hybrid"
        self.calls.append("has_collection")
        return True

    def load_collection(self, *, collection_name: str) -> None:
        assert collection_name == "knowledge_hybrid"
        self.calls.append("load_collection")

    def search(self, *, collection_name: str, **kwargs: object) -> list[list[object]]:
        assert collection_name == "knowledge_hybrid"
        self.calls.append("search")
        return [[]]


def test_upsert_flushes_batch_before_reporting_success() -> None:
    store = MilvusKnowledgeStore("http://milvus.invalid")
    client = FlushGatedClient()
    store._client = client  # type: ignore[assignment]

    store.upsert(
        ["7"],
        [[0.0] * VECTOR_DIMENSION],
        ["分类：售后\n问题：如何退货\n答案：申请退货"],
        [{"category": "售后", "content_type": "faq", "section_path": "退货"}],
    )

    assert client.calls == ["upsert", "flush"]


def test_existing_collection_is_loaded_before_vectorization() -> None:
    store = MilvusKnowledgeStore("http://milvus.invalid")
    client = FlushGatedClient()
    store._client = client  # type: ignore[assignment]

    store.ensure_collection()

    assert client.calls == ["has_collection", "load_collection"]


def test_first_online_search_loads_existing_collection() -> None:
    store = MilvusKnowledgeStore("http://milvus.invalid")
    client = FlushGatedClient()
    store._client = client  # type: ignore[assignment]

    assert store.search_bm25("邮费", limit=5) == []

    assert client.calls == ["has_collection", "load_collection", "search"]
