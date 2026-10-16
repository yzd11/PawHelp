"""Milvus collection adapter for stable knowledge vectors."""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, ParamSpec, TypeVar

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.knowledge.embedding import VECTOR_DIMENSION


_P = ParamSpec("_P")
_T = TypeVar("_T")
_MILVUS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="milvus")


async def milvus_call(fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """Keep the shared synchronous client on one worker across async requests."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_MILVUS_EXECUTOR, partial(fn, *args, **kwargs))


class MilvusKnowledgeStore:
    """Own the minimal Milvus schema and application-provided string IDs."""

    def __init__(self, uri: str, collection_name: str = "knowledge_hybrid") -> None:
        self._uri = uri
        self._client: MilvusClient | None = None
        self._collection_name = collection_name
        self._ensured = False

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._uri)
        return self._client

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def count_entities(self) -> int:
        """Count live entities for dual-write diagnostics.

        Collection statistics keep counting deleted rows until compaction, so a
        Markdown re-sync would report a false mismatch; count(*) does not.
        """

        if not self.client.has_collection(collection_name=self._collection_name):
            return 0
        self.ensure_collection()
        rows = self.client.query(
            collection_name=self._collection_name,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(rows[0]["count(*)"])

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Return the requested primary keys that are actually present."""

        if not ids or not self.client.has_collection(
            collection_name=self._collection_name
        ):
            return set()
        self.ensure_collection()
        existing: set[str] = set()
        for offset in range(0, len(ids), 1000):
            batch = list(ids[offset : offset + 1000])
            rows = self.client.query(
                collection_name=self._collection_name,
                ids=batch,
                output_fields=["id"],
            )
            existing.update(str(row["id"]) for row in rows)
        return existing

    def ensure_collection(self) -> None:
        if self._ensured:
            return
        if self.client.has_collection(collection_name=self._collection_name):
            self.client.load_collection(collection_name=self._collection_name)
            self._ensured = True
            return
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=8192,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
            enable_match=True,
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=VECTOR_DIMENSION,
        )
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="section_path", datatype=DataType.VARCHAR, max_length=512)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
            )
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        self.client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(collection_name=self._collection_name)
        self._ensured = True

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadata: Sequence[dict[str, str]],
    ) -> None:
        if not (len(ids) == len(vectors) == len(texts) == len(metadata)):
            raise ValueError("knowledge IDs, vectors, texts, and metadata must align")
        if any(len(vector) != VECTOR_DIMENSION for vector in vectors):
            raise ValueError("Milvus knowledge vectors must have 1024 dimensions")
        self.client.upsert(
            collection_name=self._collection_name,
            data=[
                {
                    "id": vector_id,
                    "text": text,
                    "dense_vector": list(vector),
                    "category": values.get("category", ""),
                    "content_type": values.get("content_type", ""),
                    "section_path": values.get("section_path", ""),
                }
                for vector_id, vector, text, values in zip(
                    ids, vectors, texts, metadata, strict=True
                )
            ],
        )
        # The caller marks these rows done in MySQL as soon as this method
        # returns. Flush first so a committed done row is searchable in Milvus.
        self.client.flush(collection_name=self._collection_name)

    def delete(self, ids: Sequence[str]) -> None:
        """Delete exact stable primary keys; missing keys are harmless."""

        if not ids or not self.client.has_collection(collection_name=self._collection_name):
            return
        self.client.delete(
            collection_name=self._collection_name,
            ids=list(ids),
        )

    def search_dense(
        self, vector: Sequence[float], *, limit: int, filter_expression: str = ""
    ) -> list[tuple[str, float]]:
        if len(vector) != VECTOR_DIMENSION:
            raise ValueError("Milvus query vectors must have 1024 dimensions")
        self.ensure_collection()
        results = self.client.search(
            collection_name=self._collection_name,
            data=[list(vector)],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {}},
            limit=limit,
            filter=filter_expression,
        )
        return self._hits(results)

    def search_bm25(
        self, query: str, *, limit: int, filter_expression: str = ""
    ) -> list[tuple[str, float]]:
        self.ensure_collection()
        results = self.client.search(
            collection_name=self._collection_name,
            data=[query],
            anns_field="sparse_vector",
            search_params={"metric_type": "BM25", "params": {}},
            limit=limit,
            filter=filter_expression,
        )
        return self._hits(results)

    def search_hybrid(
        self,
        vector: Sequence[float],
        query: str,
        *,
        dense_limit: int,
        bm25_limit: int,
        limit: int,
        filter_expression: str = "",
    ) -> list[tuple[str, float]]:
        if len(vector) != VECTOR_DIMENSION:
            raise ValueError("Milvus query vectors must have 1024 dimensions")
        self.ensure_collection()
        dense = AnnSearchRequest(
            data=[list(vector)],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=dense_limit,
            filter=filter_expression or None,
        )
        lexical = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=bm25_limit,
            filter=filter_expression or None,
        )
        results = self.client.hybrid_search(
            collection_name=self._collection_name,
            reqs=[dense, lexical],
            ranker=RRFRanker(),
            limit=limit,
        )
        return self._hits(results)

    @staticmethod
    def _hits(results: list[list[dict[str, object]]]) -> list[tuple[str, float]]:
        if not results:
            return []
        return [
            (str(hit["id"]), float(hit["distance"]))
            for hit in results[0]
        ]
