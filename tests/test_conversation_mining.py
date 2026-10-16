"""One bounded conversation-mining batch through its database boundary."""

from uuid import uuid4

from sqlalchemy import select

from app.database.models import Conversation, KnowledgeChunk, QAExtractionStaging
from app.knowledge.mining import ConversationKnowledgeExtractor
from app.repositories.conversations import ConversationRepository
from app.repositories.knowledge_mining import (
    KnowledgeCandidate,
    KnowledgeMiningRepository,
    conversation_source_ref,
)
from app.repositories.messages import MessageRepository
from tests.fakes import FakeStructuredModel


async def test_batch_is_traced_deduplicated_promoted_and_not_selected_again(
    mysql_session,
) -> None:
    repository = KnowledgeMiningRepository(mysql_session)
    existing_ids = list(await mysql_session.scalars(select(Conversation.id)))
    mysql_session.add_all(
        [
            QAExtractionStaging(
                batch_no="test-existing",
                source_ref=conversation_source_ref(conversation_id),
                question=f"existing-{conversation_id}",
                answer="already processed",
                status="discarded",
            )
            for conversation_id in existing_ids
        ]
    )
    conversations = ConversationRepository(mysql_session)
    messages = MessageRepository(mysql_session)
    await conversations.create(str(uuid4()))
    first = await conversations.create(str(uuid4()))
    second = await conversations.create(str(uuid4()))
    no_knowledge = await conversations.create(str(uuid4()))
    for conversation in (first, second, no_knowledge):
        await messages.add_user(conversation.id, "换一种说法问配送费是多少？")
        await messages.add_assistant(conversation.id, "配送费用会在订单结算页展示。")
    await mysql_session.flush()

    transcripts = await repository.load_unstaged_transcripts(limit=3)
    assert {item.conversation_id for item in transcripts} == {
        first.id,
        second.id,
        no_knowledge.id,
    }

    model = FakeStructuredModel(
        result={
            "items": [
                {
                    "source_conversation_id": first.id,
                    "question": "邮费在哪里看？",
                    "answer": "配送费用会在订单结算页展示。",
                },
                {
                    "source_conversation_id": second.id,
                    "question": " 邮费在哪里看？ ",
                    "answer": "配送费用会在订单结算页展示。",
                },
            ]
        }
    )
    candidates = await ConversationKnowledgeExtractor(
        model,
        method="json_mode",
    ).extract(transcripts)
    await repository.add_extracted("test-batch", candidates)
    await repository.add_empty_sources("test-batch", [no_knowledge.id])
    kept, discarded = await repository.deduplicate_and_promote()

    assert (kept, discarded) == (1, 1)
    staged = list(
        await mysql_session.scalars(
            select(QAExtractionStaging)
            .where(QAExtractionStaging.batch_no == "test-batch")
            .order_by(QAExtractionStaging.id)
        )
    )
    assert [row.status for row in staged] == ["kept", "discarded", "discarded"]
    assert staged[-1].source_ref == conversation_source_ref(no_knowledge.id)
    promoted = list(
        await mysql_session.scalars(
            select(KnowledgeChunk).where(
                KnowledgeChunk.section_path == conversation_source_ref(first.id)
            )
        )
    )
    assert len(promoted) == 1
    assert promoted[0].content_type == "conversation"
    assert promoted[0].vectorize_status == "pending"
    assert await repository.load_unstaged_transcripts(limit=20) == []


async def test_mining_deduplicates_against_authoritative_knowledge(
    mysql_session,
) -> None:
    mysql_session.add(
        KnowledgeChunk(
            category="配送服务",
            questions="邮费在哪里看？",
            answer="配送费用会在订单结算页展示。",
            content_type="faq",
            source_kind="manual",
            vectorize_status="pending",
        )
    )
    repository = KnowledgeMiningRepository(mysql_session)
    await repository.add_extracted(
        "cross-source-dedup",
        [
            KnowledgeCandidate(
                source_conversation_id=999,
                question=" 邮费在哪里看？ ",
                answer="配送费用会在订单结算页展示。",
            )
        ],
    )

    kept, discarded = await repository.deduplicate_and_promote()

    assert (kept, discarded) == (0, 1)
    staged = await mysql_session.scalar(
        select(QAExtractionStaging).where(
            QAExtractionStaging.batch_no == "cross-source-dedup"
        )
    )
    assert staged is not None
    assert staged.status == "discarded"
