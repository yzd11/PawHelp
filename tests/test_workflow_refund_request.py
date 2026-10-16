"""Refund form submission is explicit, owned, bound and idempotent."""

from datetime import datetime, timezone
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import Settings
from app.database.models import Conversation, Message, RefundRequest, Order, Shipment, ShipmentEvent
from app.repositories.orders import OrderRepository
from app.main import create_app
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository
from tests.fakes import FakeCombinedModel


async def test_refund_application_requires_explicit_bound_form_submission(mysql_session):
    factory = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    owner = uuid4()
    async with factory() as session, session.begin():
        conversation = await ConversationRepository(session).create(str(owner))
        conversation_id = conversation.id
        await OrderRepository(session).seed_demo(str(owner))
        await MessageRepository(session).add_user(conversation_id, "订单1001能退吗")
        offered = await MessageRepository(session).add_assistant(
            conversation_id, "请核对政策。[1]", suggested_actions=["申请退款"],
            related_order_id="1001",
        )
        not_offered = await MessageRepository(session).add_assistant(
            conversation_id, "普通回答", suggested_actions=[],
        )
        unpaid_offer = await MessageRepository(session).add_assistant(
            conversation_id, "不应申请退款", suggested_actions=["申请退款"], related_order_id="1003",
        )
        unpaid_id = unpaid_offer.id
        offered_id, not_offered_id = offered.id, not_offered.id

    app = create_app(
        settings=Settings(
            chat_base_url="http://localhost:11434/v1", openai_model="fake",
            openai_api_key="fake", database_url="mysql+asyncmy://user:pass@localhost/db",
        ),
        model=FakeCombinedModel(), session_factory=factory,
        tool_chat_service=object(),
    )
    path = f"/v1/conversations/{conversation_id}/refund-requests"
    payload = {
        "user_id": str(owner), "message_id": offered_id,
        "order_id": "1001", "reason": "七天无理由", "description": "商品完好",
    }
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            history = await client.get(
                f"/v1/conversations/{conversation_id}/messages",
                headers={"X-Demo-User-Id": str(owner)},
            )
            assert history.status_code == 200
            offered_public = next(item for item in history.json() if item["id"] == offered_id)
            assert offered_public["suggested_actions"] == ["申请退款"]
            assert offered_public["related_order_id"] == "1001"
            assert (await client.post(path, json={**payload, "reason": "随便"})).status_code == 422
            assert (await client.post(path, json={**payload, "user_id": str(uuid4())})).status_code == 404
            assert (await client.post(path, json={**payload, "message_id": not_offered_id})).status_code == 403
            assert (await client.post(path, json={**payload, "order_id": "1002"})).status_code == 403
            assert (await client.post(path, json={**payload, "message_id": unpaid_id, "order_id": "1003"})).status_code == 409
            async with factory() as session:
                assert list(await session.scalars(select(RefundRequest).where(RefundRequest.conversation_id == conversation_id))) == []

            first = await client.post(path, json=payload)
            repeated = await client.post(path, json=payload)
            conflict = await client.post(path, json={**payload, "reason": "质量问题"})
            assert first.status_code == repeated.status_code == 200
            assert first.json() == repeated.json()
            assert conflict.status_code == 409
            assert first.json()["status"] == "待人工审核"
            assert first.json()["created_at"].endswith("Z")
            async with factory() as session:
                rows = list(await session.scalars(select(RefundRequest).where(RefundRequest.conversation_id == conversation_id)))
                assert len(rows) == 1
                assert rows[0].order_id == "1001"
                assert rows[0].order_user_id == str(owner)
                assert rows[0].reason == "七天无理由"
                assert datetime.fromisoformat(first.json()["created_at"]).astimezone(timezone.utc).replace(tzinfo=None) == rows[0].created_at
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(RefundRequest).where(RefundRequest.conversation_id == conversation_id))
            await session.execute(delete(Message).where(Message.conversation_id == conversation_id))
            await session.execute(delete(Conversation).where(Conversation.id == conversation_id))
            tracking = select(Shipment.tracking_no).where(Shipment.user_id == str(owner))
            await session.execute(delete(ShipmentEvent).where(ShipmentEvent.tracking_no.in_(tracking)))
            await session.execute(delete(Shipment).where(Shipment.user_id == str(owner)))
            await session.execute(delete(Order).where(Order.user_id == str(owner)))
