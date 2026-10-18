"""Ownership-aware conversation persistence."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.models import Conversation, Message


class ConversationNotFound(LookupError):
    """The public-equivalent result for missing and foreign conversations."""

    def __init__(self, conversation_id: int | None = None) -> None:
        super().__init__("conversation not found")


class ConversationRepository:
    """Use one caller-owned session for a conversation unit of work."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_owned(self, conversation_id: int, user_id: str) -> Conversation:
        """Return an owned conversation without revealing foreign existence."""

        row = await self._session.get(Conversation, conversation_id)
        if row is None or row.user_id != user_id:
            raise ConversationNotFound(conversation_id)
        return row

    async def create(self, user_id: str) -> Conversation:
        """Create a new active conversation and obtain its database identity."""

        row = Conversation(user_id=user_id, status="进行中")
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_owned(self, user_id: str) -> list[dict[str, object]]:
        first_question = select(Message.content).where(Message.conversation_id == Conversation.id, Message.role == "user").order_by(Message.id).limit(1).scalar_subquery()
        rows = await self._session.execute(select(Conversation, first_question.label("preview")).where(Conversation.user_id == user_id).order_by(Conversation.id.desc()))
        return [{"id": conversation.id, "preview": (conversation.title or preview or "新会话")[:80], "title": conversation.title, "has_summary": bool(conversation.summary_upto_msg_id), "created_at": conversation.created_at.isoformat(), "status": conversation.status} for conversation, preview in rows]
