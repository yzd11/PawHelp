"""Explicit ticket-click boundary: ownership, offered action, and idempotency."""

from datetime import datetime, timezone
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import Settings
from app.database.models import Conversation, Message, Ticket, ToolAuditLog
from app.main import create_app
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository
from tests.fakes import FakeCombinedModel


async def test_ticket_is_written_only_by_owned_explicit_click(mysql_session):
    factory = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    owner = uuid4()
    async with factory() as session, session.begin():
        conversation = await ConversationRepository(session).create(str(owner))
        conversation_id = conversation.id
        await MessageRepository(session).add_user(conversation_id, "我要投诉订单1001")
        offered = await MessageRepository(session).add_assistant(
            conversation_id, "很抱歉", suggested_actions=["转人工", "建工单"]
        )
        not_offered = await MessageRepository(session).add_assistant(
            conversation_id, "普通回复", suggested_actions=[]
        )
        offered_id, not_offered_id = offered.id, not_offered.id

    settings = Settings(
        chat_base_url="http://localhost:11434/v1", openai_model="fake", openai_api_key="fake",
        database_url="mysql+asyncmy://user:pass@localhost/db",
    )
    app = create_app(
        settings=settings, model=FakeCombinedModel(), session_factory=factory,
        tool_chat_service=object(),  # No chat endpoint is called in this test.
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            path = f"/v1/conversations/{conversation_id}/tickets"
            payload = {
                "user_id": str(owner), "message_id": offered_id,
                "ticket_type": "咨询", "description": "请核查页面付款时的错误提示",
            }
            missing_details = await client.post(path, json={"user_id": str(owner), "message_id": offered_id})
            assert missing_details.status_code == 422
            blank_description = await client.post(path, json={**payload, "description": "   "})
            assert blank_description.status_code == 422
            invalid_type = await client.post(path, json={**payload, "ticket_type": "其他"})
            assert invalid_type.status_code == 422
            denied = await client.post(path, json={**payload, "message_id": not_offered_id})
            assert denied.status_code == 403
            async with factory() as session:
                assert list(await session.scalars(select(Ticket).where(Ticket.conversation_id == conversation_id))) == []

            wrong_owner = await client.post(path, json={**payload, "user_id": str(uuid4())})
            assert wrong_owner.status_code == 404

            first = await client.post(path, json=payload)
            second = await client.post(path, json=payload)
            assert first.status_code == second.status_code == 200
            assert first.json() == second.json()
            assert first.json()["created_at"].endswith("Z")
            async with factory() as session:
                tickets = list(await session.scalars(select(Ticket).where(Ticket.conversation_id == conversation_id)))
                assert len(tickets) == 1
                assert tickets[0].ticket_type == "咨询"
                assert tickets[0].description == payload["description"]
                assert datetime.fromisoformat(first.json()["created_at"]).astimezone(timezone.utc).replace(tzinfo=None) == tickets[0].created_at
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(ToolAuditLog).where(ToolAuditLog.conversation_id == conversation_id))
            await session.execute(delete(Ticket).where(Ticket.conversation_id == conversation_id))
            await session.execute(delete(Message).where(Message.conversation_id == conversation_id))
            await session.execute(delete(Conversation).where(Conversation.id == conversation_id))
