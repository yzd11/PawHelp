"""Idempotently add the small exact-model fixture used by the retrieval evaluation."""

import asyncio
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.database.models import KnowledgeChunk
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.knowledge.chunking import chunk_markdown
from app.repositories.knowledge import KnowledgeRepository


SOURCE = Path("knowledge/product-faq/models.md")


async def run() -> int:
    settings = get_settings()
    engine = create_async_engine_from_settings(settings)
    factory = create_session_factory(engine)
    try:
        drafts = chunk_markdown(SOURCE.read_text(encoding="utf-8"), content_type="faq")
        inserted = 0
        async with factory() as session, session.begin():
            repository = KnowledgeRepository(session)
            for draft in drafts:
                existing = await session.scalar(
                    select(KnowledgeChunk.id).where(
                        KnowledgeChunk.category == draft.category,
                        KnowledgeChunk.questions == draft.questions,
                        KnowledgeChunk.answer == draft.answer,
                    )
                )
                if existing is not None:
                    continue
                await repository.add_drafts([draft], link_neighbors=False)
                inserted += 1
        return inserted
    finally:
        await engine.dispose()


def main() -> None:
    inserted = asyncio.run(run())
    print(f"inserted={inserted} status=pending")


if __name__ == "__main__":
    main()
