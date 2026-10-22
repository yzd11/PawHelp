"""Idempotent human-support ticket persistence."""

from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Ticket


class TicketRepository:
    """Use a stable tool-derived ticket number as the retry boundary."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_or_get(
        self,
        *,
        ticket_no: str,
        conversation_id: int,
        description: str,
        ticket_type: str,
        status: str,
    ) -> tuple[Ticket, bool]:
        """Insert once; a duplicate ticket identity is a successful retry."""

        result = await self._session.execute(
            insert(Ticket)
            .values(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
                status=status,
                created_at=func.utc_timestamp(),
            )
            .prefix_with("IGNORE")
        )
        row = await self._session.get(Ticket, ticket_no)
        if row is None:
            raise RuntimeError("ticket insertion did not produce a row")
        return row, result.rowcount == 1
