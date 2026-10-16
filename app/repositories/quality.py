"""Persistence for low-confidence questions and faithfulness cases."""

from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import FaithCase, LowConfidenceQuestion


class FaithCaseNotFound(Exception):
    """The case is absent from the persistent review ledger."""


class FaithCaseConflict(Exception):
    """The review form no longer describes the current ledger row."""


class QualityRepository:
    """Write quality signals without exposing database details to services."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_low_confidence(
        self,
        *,
        conversation_id: int | None,
        raw_question: str,
        source: str,
        reason: str | None,
        retrieved_chunks: list[dict] | None = None,
        event_key: str | None = None,
    ) -> LowConfidenceQuestion:
        if event_key:
            existing = await self._session.scalar(select(LowConfidenceQuestion).where(LowConfidenceQuestion.event_key == event_key))
            if existing is not None:
                return existing
        row = LowConfidenceQuestion(
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason,
            retrieved_chunks=retrieved_chunks,
            event_key=event_key,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def upsert_faith_case(
        self,
        *,
        eval_id: str,
        bucket: str,
        query: str,
        strategy: str,
        answer: str,
        reason: str,
        citations: Sequence[dict[str, object]] | None,
        judge_model: str | None,
    ) -> None:
        statement = insert(FaithCase).values(
            eval_id=eval_id,
            bucket=bucket,
            query=query,
            strategy=strategy,
            answer=answer,
            reason=reason,
            citations=list(citations) if citations is not None else None,
            judge_model=judge_model,
        )
        await self._session.execute(
            statement.on_duplicate_key_update(
                bucket=statement.inserted.bucket,
                query=statement.inserted.query,
                strategy=statement.inserted.strategy,
                answer=statement.inserted.answer,
                reason=statement.inserted.reason,
                citations=statement.inserted.citations,
                judge_model=statement.inserted.judge_model,
                status="未解决",
                seen_count=FaithCase.seen_count + 1,
                last_seen_at=func.current_timestamp(),
                resolution=None,
                resolved_at=None,
            )
        )

    async def list_faith_cases(self, eval_ids: Sequence[str]) -> list[FaithCase]:
        if not eval_ids:
            return []
        result = await self._session.execute(
            select(FaithCase).where(FaithCase.eval_id.in_(eval_ids))
        )
        return list(result.scalars().all())

    async def review_faith_case(
        self,
        *,
        eval_id: str,
        answer: str,
        judge_reason: str,
        expected_seen_count: int,
        expected_status: str,
        expected_resolution: str | None,
        status: str,
        resolution: str | None,
    ) -> FaithCase:
        result = await self._session.execute(
            select(FaithCase).where(FaithCase.eval_id == eval_id).with_for_update()
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise FaithCaseNotFound
        if (
            row.answer != answer
            or row.reason != judge_reason
            or row.seen_count != expected_seen_count
            or row.status != expected_status
            or row.resolution != expected_resolution
        ):
            raise FaithCaseConflict
        row.status = status
        row.resolution = resolution
        row.resolved_at = (
            datetime.now(timezone.utc).replace(tzinfo=None)
            if status != "未解决" else None
        )
        await self._session.flush()
        return row
