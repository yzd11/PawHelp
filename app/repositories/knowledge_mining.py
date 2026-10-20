"""Persistence operations for one bounded conversation-mining batch."""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    Conversation,
    KnowledgeChunk,
    Message,
    QAExtractionStaging,
)
from app.repositories.knowledge import knowledge_identity


CONVERSATION_SOURCE_PREFIX = "conversation:"
NO_REUSABLE_KNOWLEDGE = "无可复用知识"


def conversation_source_ref(conversation_id: int) -> str:
    return f"{CONVERSATION_SOURCE_PREFIX}{conversation_id}"


def _unstaged_conversation_predicates() -> tuple[object, object, object]:
    already_staged = exists().where(
        QAExtractionStaging.source_ref
        == func.concat(CONVERSATION_SOURCE_PREFIX, Conversation.id)
    )
    has_user_message = exists().where(
        Message.conversation_id == Conversation.id,
        Message.role == "user",
        Message.content.is_not(None),
        Message.content != "",
    )
    has_assistant_message = exists().where(
        Message.conversation_id == Conversation.id,
        Message.role == "assistant",
        Message.content.is_not(None),
        Message.content != "",
    )
    return ~already_staged, has_user_message, has_assistant_message


@dataclass(frozen=True)
class Transcript:
    conversation_id: int
    messages: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class KnowledgeCandidate:
    source_conversation_id: int
    question: str
    answer: str


class KnowledgeMiningRepository:
    """Select new sources, stage extraction, then promote unique candidates."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def load_unstaged_transcripts(self, *, limit: int) -> list[Transcript]:
        conversations = list(
            await self._session.scalars(
                select(Conversation)
                .where(*_unstaged_conversation_predicates())
                .order_by(Conversation.id)
                .limit(limit)
            )
        )
        if not conversations:
            return []
        ids = [row.id for row in conversations]
        messages = list(
            await self._session.scalars(
                select(Message)
                .where(
                    Message.conversation_id.in_(ids),
                    Message.role.in_(("user", "assistant")),
                    Message.content.is_not(None),
                    Message.content != "",
                )
                .order_by(Message.conversation_id, Message.id)
            )
        )
        grouped: dict[int, list[dict[str, str]]] = {value: [] for value in ids}
        for message in messages:
            grouped[message.conversation_id].append(
                {"role": message.role, "content": message.content or ""}
            )
        return [
            Transcript(row.id, tuple(grouped[row.id]))
            for row in conversations
            if grouped[row.id]
        ]

    async def stats(self) -> dict[str, int]:
        """Summarize the staging lifecycle and sources still awaiting extraction."""

        status_rows = await self._session.execute(
            select(
                QAExtractionStaging.status,
                func.count(QAExtractionStaging.id),
            ).group_by(QAExtractionStaging.status)
        )
        status_counts = {
            str(status): int(count) for status, count in status_rows.all()
        }
        batches = await self._session.scalar(
            select(func.count(func.distinct(QAExtractionStaging.batch_no)))
        )
        unprocessed = await self._session.scalar(
            select(func.count(Conversation.id)).where(
                *_unstaged_conversation_predicates()
            )
        )
        return {
            "unprocessed_conversations": int(unprocessed or 0),
            "extracted": status_counts.get("extracted", 0),
            "kept": status_counts.get("kept", 0),
            "discarded": status_counts.get("discarded", 0),
            "batches": int(batches or 0),
        }

    async def add_extracted(
        self,
        batch_no: str,
        candidates: Sequence[KnowledgeCandidate],
        rejections: Sequence[str | None] = (),
    ) -> None:
        """Stage candidates; a rejected one is discarded with its reason and never promoted."""

        reasons = list(rejections) or [None] * len(candidates)
        self._session.add_all(
            [
                QAExtractionStaging(
                    batch_no=batch_no,
                    source_ref=conversation_source_ref(candidate.source_conversation_id),
                    question=candidate.question,
                    answer=candidate.answer,
                    status="extracted" if reason is None else "discarded",
                    discard_reason=reason[:500] if reason else None,
                )
                for candidate, reason in zip(candidates, reasons, strict=True)
            ]
        )
        await self._session.flush()

    async def add_empty_sources(
        self,
        batch_no: str,
        conversation_ids: Sequence[int],
    ) -> None:
        """Record sources with no reusable QA so scheduled runs do not resend them."""

        self._session.add_all(
            [
                QAExtractionStaging(
                    batch_no=batch_no,
                    source_ref=conversation_source_ref(conversation_id),
                    question=NO_REUSABLE_KNOWLEDGE,
                    answer=NO_REUSABLE_KNOWLEDGE,
                    status="discarded",
                )
                for conversation_id in conversation_ids
            ]
        )
        await self._session.flush()

    async def deduplicate_and_promote(self) -> tuple[int, int]:
        """Promote only new globally unique candidates in one transaction."""

        existing = list(
            await self._session.scalars(
                select(QAExtractionStaging)
                .where(QAExtractionStaging.status == "kept")
                .order_by(QAExtractionStaging.id)
            )
        )
        seen = {knowledge_identity(row.question, row.answer) for row in existing}
        knowledge_rows = await self._session.execute(
            select(KnowledgeChunk.questions, KnowledgeChunk.answer)
        )
        seen.update(
            knowledge_identity(question, answer)
            for question, answer in knowledge_rows.all()
        )
        pending = list(
            await self._session.scalars(
                select(QAExtractionStaging)
                .where(QAExtractionStaging.status == "extracted")
                .order_by(QAExtractionStaging.id)
            )
        )
        kept = 0
        discarded = 0
        for row in pending:
            identity = knowledge_identity(row.question, row.answer)
            if identity in seen:
                row.status = "discarded"
                discarded += 1
                continue
            seen.add(identity)
            row.status = "kept"
            self._session.add(
                KnowledgeChunk(
                    category="历史会话",
                    questions=row.question,
                    answer=row.answer,
                    section_path=row.source_ref,
                    content_type="conversation",
                    source_kind="conversation",
                    source_path=row.source_ref,
                    is_key_clause=False,
                    vectorize_status="pending",
                )
            )
            kept += 1
        await self._session.flush()
        return kept, discarded
