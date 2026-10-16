"""Run one semantic knowledge query against local BGE-M3, Milvus, and MySQL."""

import argparse
import asyncio
import json

from app.config import get_settings
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.knowledge.embedding import BgeM3Embedder
from app.knowledge.query_understanding import QueryUnderstandingService
from app.knowledge.reranking import BgeReranker
from app.knowledge.search import HybridKnowledgeSearch, RetrievalStrategy
from app.knowledge.vector_store import MilvusKnowledgeStore
from app.llm import create_chat_model


async def run(query: str, strategy: RetrievalStrategy) -> dict[str, object]:
    settings = get_settings()
    engine = create_async_engine_from_settings(settings)
    factory = create_session_factory(engine)
    search = HybridKnowledgeSearch(
        factory,
        BgeM3Embedder(
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
        ),
        MilvusKnowledgeStore(settings.milvus_uri, settings.milvus_hybrid_collection),
        BgeReranker(
            settings.reranker_model,
            batch_size=settings.reranker_batch_size,
            query_max_length=settings.reranker_query_max_length,
            passage_max_length=settings.reranker_passage_max_length,
        ),
        QueryUnderstandingService(
            create_chat_model(settings), method=settings.structured_output_method
        ),
        confidence_threshold=settings.knowledge_rerank_threshold,
        dense_limit=settings.dense_recall_limit,
        bm25_limit=settings.bm25_recall_limit,
        rerank_limit=settings.rerank_limit,
    )
    try:
        result = await search.retrieve(query, strategy=strategy, limit=10)
        return {
            "strategy": result.strategy.value,
            "normalized_query": result.query_plan.normalized_query,
            "confident": result.confident,
            "matches": [
                {
                    "chunk_id": match.chunk.id,
                    "question": match.chunk.questions,
                    "answer": match.chunk.answer,
                    "category": match.chunk.category,
                    "score": match.score,
                }
                for match in result.matches
            ],
        }
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in RetrievalStrategy],
        default=RetrievalStrategy.HYBRID_RERANK.value,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run(args.query, RetrievalStrategy(args.strategy))),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
