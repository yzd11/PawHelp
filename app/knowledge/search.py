"""Dense, BM25, hybrid, and reranked retrieval with MySQL authority."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import json
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.models import KnowledgeChunk
from app.knowledge.embedding import BgeM3Embedder
from app.knowledge.query_understanding import IdentityQueryUnderstanding, QueryPlan
from app.knowledge.reranking import BgeReranker
from app.knowledge.vector_store import MilvusKnowledgeStore, milvus_call
from app.repositories.knowledge import KnowledgeRepository


class RetrievalStrategy(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass(frozen=True)
class KnowledgeMatch:
    chunk: KnowledgeChunk
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    matches: tuple[KnowledgeMatch, ...]
    query_plan: QueryPlan
    strategy: RetrievalStrategy
    confident: bool


class QueryPlanner(Protocol):
    async def understand(self, query: str) -> QueryPlan: ...


class KnowledgeSearch(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
        limit: int = 10,
        categories: Sequence[str] = (),
        content_types: Sequence[str] = (),
        allowed_chunk_ids: Sequence[int] | None = None,
    ) -> RetrievalResult: ...


def _string_list(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def build_filter_expression(
    categories: Sequence[str], content_types: Sequence[str],
    allowed_chunk_ids: Sequence[int] | None = None,
) -> str:
    """Build only allowlisted equality filters; never accept raw expressions."""

    clauses: list[str] = []
    if categories:
        clauses.append(f"category in {_string_list(categories)}")
    if content_types:
        clauses.append(f"content_type in {_string_list(content_types)}")
    if allowed_chunk_ids is not None:
        if not allowed_chunk_ids:
            raise ValueError("evaluation corpus contains no vectorized knowledge chunks")
        clauses.append(f"id in {_string_list([str(chunk_id) for chunk_id in allowed_chunk_ids])}")
    return " and ".join(clauses)


class HybridKnowledgeSearch:
    """Understand, retrieve, hydrate, and optionally rerank one standalone query."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: BgeM3Embedder,
        vector_store: MilvusKnowledgeStore,
        reranker: BgeReranker | None = None,
        query_planner: QueryPlanner | None = None,
        *,
        confidence_threshold: float = 0.5,
        dense_limit: int = 50,
        bm25_limit: int = 50,
        rerank_limit: int = 10,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._query_planner = query_planner or IdentityQueryUnderstanding()
        self._confidence_threshold = confidence_threshold
        self._dense_limit = dense_limit
        self._bm25_limit = bm25_limit
        self._rerank_limit = rerank_limit

    async def retrieve(
        self,
        query: str,
        *,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
        limit: int = 10,
        categories: Sequence[str] = (),
        content_types: Sequence[str] = (),
        allowed_chunk_ids: Sequence[int] | None = None,
    ) -> RetrievalResult:
        plan = await self._query_planner.understand(query)
        filter_expression = build_filter_expression(
            categories, content_types, allowed_chunk_ids
        )
        lexical_query = " ".join([plan.normalized_query, *plan.synonyms])

        vector: list[float] | None = None
        if strategy is not RetrievalStrategy.BM25:
            vector = (await asyncio.to_thread(self._embedder.embed, [plan.normalized_query]))[0]

        raw_hits = await self._recall(
            strategy, vector, lexical_query, filter_expression
        )
        raw_ids = [vector_id for vector_id, _ in raw_hits]
        async with self._session_factory() as session:
            rows = await KnowledgeRepository(session).get_vectorized_by_ids(raw_ids)

        if strategy is RetrievalStrategy.HYBRID_RERANK:
            if self._reranker is None:
                raise RuntimeError("hybrid_rerank requires a configured reranker")
            ranked = await asyncio.to_thread(
                self._reranker.rerank,
                plan.normalized_query,
                rows,
                limit=min(limit, self._rerank_limit),
            )
            matches = tuple(
                KnowledgeMatch(chunk=item.chunk, score=float(item.score))
                for item in ranked
            )
        else:
            score_by_id = {vector_id: score for vector_id, score in raw_hits}
            matches = tuple(
                KnowledgeMatch(row, float(score_by_id.get(str(row.id), 0.0)))
                for row in rows[:limit]
            )

        confident = bool(matches) and (
            strategy is not RetrievalStrategy.HYBRID_RERANK
            or matches[0].score >= self._confidence_threshold
        )
        return RetrievalResult(matches, plan, strategy, confident)

    async def _recall(
        self,
        strategy: RetrievalStrategy,
        vector: list[float] | None,
        lexical_query: str,
        filter_expression: str,
    ) -> list[tuple[str, float]]:
        if strategy is RetrievalStrategy.DENSE:
            return await milvus_call(
                self._vector_store.search_dense,
                vector,
                limit=self._dense_limit,
                filter_expression=filter_expression,
            )
        if strategy is RetrievalStrategy.BM25:
            return await milvus_call(
                self._vector_store.search_bm25,
                lexical_query,
                limit=self._bm25_limit,
                filter_expression=filter_expression,
            )
        return await milvus_call(
            self._vector_store.search_hybrid,
            vector,
            lexical_query,
            dense_limit=self._dense_limit,
            bm25_limit=self._bm25_limit,
            limit=max(self._dense_limit, self._bm25_limit),
            filter_expression=filter_expression,
        )

    async def search(self, query: str, *, limit: int = 5) -> list[KnowledgeChunk]:
        """Compatibility helper for the local knowledge console."""

        result = await self.retrieve(query, limit=limit)
        return [match.chunk for match in result.matches]


SemanticKnowledgeSearch = HybridKnowledgeSearch
