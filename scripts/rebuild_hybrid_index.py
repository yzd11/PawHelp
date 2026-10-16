"""Idempotently rebuild the retrieval hybrid collection from authoritative MySQL."""

import asyncio

from app.config import get_settings
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.knowledge.embedding import BgeM3Embedder
from app.knowledge.vector_store import MilvusKnowledgeStore
from app.knowledge.vectorization import rebuild_collection
from app.repositories.knowledge import KnowledgeRepository


async def run() -> int:
    settings = get_settings()
    engine = create_async_engine_from_settings(settings)
    factory = create_session_factory(engine)
    embedder = BgeM3Embedder(
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        max_length=settings.embedding_max_length,
    )
    vector_store = MilvusKnowledgeStore(
        settings.milvus_uri,
        settings.milvus_hybrid_collection,
    )
    try:
        async with factory() as session:
            return await rebuild_collection(
                KnowledgeRepository(session),
                embedder,
                vector_store,
                batch_size=settings.vectorize_batch_size,
            )
    finally:
        await engine.dispose()


def main() -> None:
    count = asyncio.run(run())
    print(f"reindexed={count} collection=knowledge_hybrid")


if __name__ == "__main__":
    main()
