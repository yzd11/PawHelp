"""Milvus diagnostics used by the knowledge console."""

from app.knowledge.vector_store import MilvusKnowledgeStore


class ResyncedClient:
    """Milvus after a re-sync: 44 deleted rows still counted in statistics."""

    def has_collection(self, *, collection_name: str) -> bool:
        return True

    def get_collection_stats(self, *, collection_name: str) -> dict[str, int]:
        return {"row_count": 156}

    def query(self, *, collection_name: str, filter: str, output_fields: list[str], **_: object) -> list[dict[str, int]]:
        assert output_fields == ["count(*)"]
        return [{"count(*)": 112}]


def test_entity_count_excludes_rows_deleted_by_a_resync() -> None:
    store = MilvusKnowledgeStore("http://milvus.invalid")
    store._client = ResyncedClient()  # type: ignore[assignment]
    store._ensured = True

    assert store.count_entities() == 112
