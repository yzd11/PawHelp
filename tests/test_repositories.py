"""Persistence behavior at the repository boundary."""

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.repositories.conversations import ConversationNotFound, ConversationRepository
from app.repositories.messages import MessageRepository
from app.repositories.tickets import TicketRepository


async def test_conversation_identity_is_database_generated_and_owner_checked(
    mysql_session,
) -> None:
    owner_id = str(uuid4())
    repo = ConversationRepository(mysql_session)

    conversation = await repo.create(owner_id)

    assert isinstance(conversation.id, int)
    assert conversation.id > 0
    assert conversation.status == "进行中"
    assert await repo.get_owned(conversation.id, owner_id) is conversation
    with pytest.raises(ConversationNotFound):
        await repo.get_owned(conversation.id, str(uuid4()))


async def test_messages_are_appended_in_database_order(mysql_session) -> None:
    conversation = await ConversationRepository(mysql_session).create(str(uuid4()))
    repo = MessageRepository(mysql_session)

    await repo.add_user(conversation.id, "先问")
    await repo.add_assistant(conversation.id, "先答")
    await repo.add_tool(conversation.id, '{"status":"ok"}', "call-1")

    rows = await repo.list_ordered(conversation.id)
    assert [(row.role, row.content) for row in rows] == [
        ("user", "先问"),
        ("assistant", "先答"),
        ("tool", '{"status":"ok"}'),
    ]


async def test_ticket_uses_the_conversation_numeric_identity(mysql_session) -> None:
    conversation = await ConversationRepository(mysql_session).create(str(uuid4()))
    repo = TicketRepository(mysql_session)

    ticket, created = await repo.create_or_get(
        ticket_no="TICKET-TEST-1",
        conversation_id=conversation.id,
        description="包裹破损",
        ticket_type="投诉",
        status="待处理",
    )

    assert created is True
    assert ticket.conversation_id == conversation.id
    assert ticket.status == "待处理"
