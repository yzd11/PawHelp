"""Mine at most one bounded batch of historical conversations into knowledge."""

import argparse
import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from app.config import get_settings
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.knowledge.mining import ConversationKnowledgeExtractor
from app.llm import create_chat_model
from app.repositories.knowledge_mining import KnowledgeMiningRepository


async def run(batch_size: int | None = None) -> tuple[int, int, int, int]:
    settings = get_settings()
    engine = create_async_engine_from_settings(settings)
    factory = create_session_factory(engine)
    extractor = ConversationKnowledgeExtractor(
        create_chat_model(settings),
        method=settings.structured_output_method,
    )
    try:
        async with factory() as session:
            repository = KnowledgeMiningRepository(session)
            transcripts = await repository.load_unstaged_transcripts(
                limit=batch_size or settings.conversation_mining_batch_size
            )
            if not transcripts:
                return 0, 0, 0, 0
            candidates = await extractor.extract(transcripts)
            batch_no = f"{datetime.now(UTC):%Y%m%d%H%M%S}-{uuid4().hex[:8]}"
            await repository.add_extracted(batch_no, candidates)
            represented_sources = {
                candidate.source_conversation_id for candidate in candidates
            }
            await repository.add_empty_sources(
                batch_no,
                [
                    transcript.conversation_id
                    for transcript in transcripts
                    if transcript.conversation_id not in represented_sources
                ],
            )
            kept, discarded = await repository.deduplicate_and_promote()
            await session.commit()
            return len(transcripts), len(candidates), kept, discarded
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    conversations, extracted, kept, discarded = asyncio.run(run(args.batch_size))
    print(
        f"conversations={conversations} extracted={extracted} "
        f"kept={kept} discarded={discarded}"
    )


if __name__ == "__main__":
    main()
