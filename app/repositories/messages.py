"""Ordered message persistence."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Message


class MessageRepository:
    """Persist and load conversation messages in database order."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_user(self, conversation_id: int, content: str) -> Message:
        """Append one user message."""

        row = Message(conversation_id=conversation_id, role="user", content=content)
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_ordered(self, conversation_id: int) -> list[Message]:
        """Load canonical history in monotonically assigned message-id order."""

        rows = await self._session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(rows)

    async def add_assistant(
        self,
        conversation_id: int,
        content: str | None,
        *,
        tool_calls: dict[str, object] | list[object] | None = None,
        suggested_actions: list[str] | None = None,
        source_refs: list[dict[str, object]] | None = None,
        related_order_id: str | None = None,
        tool_trace: list[dict[str, object]] | None = None,
        reply_to_message_id: int | None = None,
        retrieved_chunks: list[dict] | None = None,
    ) -> Message:
        """Append a model response or model-issued tool-call request."""

        row = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            suggested_actions=suggested_actions,
            source_refs=source_refs,
            related_order_id=related_order_id,
            tool_trace=tool_trace,
            reply_to_message_id=reply_to_message_id,
            retrieved_chunks=retrieved_chunks,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def add_tool(
        self, conversation_id: int, content: str, tool_call_id: str
    ) -> Message:
        """Append a trusted tool result with its model correlation identifier."""

        row = Message(
            conversation_id=conversation_id,
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        )
        self._session.add(row)
        await self._session.flush()
        return row
