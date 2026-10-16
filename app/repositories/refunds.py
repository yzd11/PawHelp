"""Idempotent persistence for user-submitted refund applications."""

from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import RefundRequest


class RefundRequestConflict(ValueError):
    """The same offer was previously submitted with different details."""


class RefundRequestRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_or_get(
        self, *, conversation_id: int, source_message_id: int,
        order_id: str, order_user_id: str, reason: str, description: str | None,
    ) -> RefundRequest:
        request_no = f"REFUND-{source_message_id}"
        await self._session.execute(
            insert(RefundRequest).values(
                request_no=request_no, conversation_id=conversation_id,
                source_message_id=source_message_id, order_id=order_id, order_user_id=order_user_id,
                reason=reason, description=description,
                status="待人工审核", created_at=func.utc_timestamp(),
            ).prefix_with("IGNORE")
        )
        row = await self._session.get(RefundRequest, request_no)
        if row is None:
            raise RuntimeError("refund request insertion did not produce a row")
        if (
            row.conversation_id != conversation_id or row.source_message_id != source_message_id
            or row.order_id != order_id or row.reason != reason or row.description != description
            or row.order_user_id not in (None, order_user_id)
        ):
            raise RefundRequestConflict("offer already submitted with different details")
        return row
