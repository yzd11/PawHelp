"""DB/API/tool acceptance for owned, persistent demonstration data."""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from random import Random
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import Settings
from app.database.models import Order, Shipment, ShipmentEvent
from app.main import create_app
from app.repositories.orders import OrderNotFound, OrderRepository
from app.services.orders import OrderData
from app.tools.business import ToolRuntime, build_business_tools
from tests.fakes import FakeCombinedModel


@pytest.fixture
async def owned_data(mysql_session):
    factory = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    users = [str(uuid4()), str(uuid4())]
    try:
        yield factory, users
    finally:
        async with factory() as session, session.begin():
            tracking = select(Shipment.tracking_no).where(Shipment.user_id.in_(users))
            await session.execute(delete(ShipmentEvent).where(ShipmentEvent.tracking_no.in_(tracking)))
            await session.execute(delete(Shipment).where(Shipment.user_id.in_(users)))
            await session.execute(delete(Order).where(Order.user_id.in_(users)))


async def test_tools_cards_and_restart_share_the_same_persisted_snapshot(owned_data):
    factory, (owner, _) = owned_data
    data = OrderData(factory)
    cards = await data.list_orders(owner)
    assert [card["order_id"] for card in cards] == ["1001", "1002", "1003", "1004"]
    first = {tool.name: tool for tool in build_business_tools(ToolRuntime(1, factory, Random(1), user_id=owner))}
    second = {tool.name: tool for tool in build_business_tools(ToolRuntime(2, factory, Random(55), user_id=owner))}
    assert json.loads(await first["query_order"].ainvoke({"order_id": "1001"})) == cards[0]
    assert await first["query_order"].ainvoke({"order_id": "1001"}) == await second["query_order"].ainvoke({"order_id": "1001"})
    assert await OrderData(factory).list_orders(owner) == cards
    tracking = cards[0]["tracking_no"]
    by_tracking = json.loads(await first["query_logistics"].ainvoke({"tracking_no": tracking}))
    assert by_tracking == json.loads(await second["query_logistics"].ainvoke({"order_id": "1001"}))
    assert by_tracking["status"] == cards[0]["status"] == "delivered"
    assert by_tracking["trace"][-1]["occurred_at"] == cards[0]["delivered_at"]

    app = create_app(settings=Settings(), model=FakeCombinedModel(), session_factory=factory, tool_chat_service=object())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/orders", headers={"X-Demo-User-Id": owner})
        assert response.status_code == 200
        assert response.json() == cards


async def test_owned_tracking_and_unknown_orders_do_not_leak_or_invent_data(owned_data):
    factory, (owner, stranger) = owned_data
    data = OrderData(factory)
    cards = await data.list_orders(owner)
    foreign = await data.list_orders(stranger)
    assert cards[0]["tracking_no"] != foreign[0]["tracking_no"]
    tools = {tool.name: tool for tool in build_business_tools(ToolRuntime(1, factory, Random(0), user_id=owner))}
    missing = json.loads(await tools["query_order"].ainvoke({"order_id": "9999"}))
    leaked = json.loads(await tools["query_logistics"].ainvoke({"tracking_no": foreign[0]["tracking_no"]}))
    assert missing == leaked == {"code": "order_not_found", "error": "没有找到您的这笔订单"}
    with pytest.raises(OrderNotFound):
        await data.get_logistics(owner, order_id="1002", tracking_no=cards[0]["tracking_no"])
    with pytest.raises(ValueError):
        await tools["query_logistics"].ainvoke({})
    with pytest.raises(ValueError):
        await tools["query_order"].ainvoke({"order_id": "1001", "user_id": stranger})


async def test_unpaid_and_old_signed_scenarios_and_utc_event_order(owned_data):
    factory, (owner, _) = owned_data
    data = OrderData(factory)
    anchor = datetime(2026, 10, 2, 0, 0, tzinfo=timezone.utc)
    async with factory() as session, session.begin():
        assert await OrderRepository(session).seed_demo(owner, at=anchor)
    cards = await data.list_orders(owner)
    assert cards[2]["status"] == "pending_payment"
    assert cards[2]["tracking_no"] is None
    assert cards[2]["shipped_at"] is None
    assert (await data.get_logistics(owner, order_id="1003"))["trace"] == []
    assert (await data.get_logistics(owner, order_id="1003"))["status"] == "not_shipped"
    assert anchor - datetime.fromisoformat(cards[0]["delivered_at"]) == timedelta(days=2)
    assert anchor - datetime.fromisoformat(cards[3]["delivered_at"]) == timedelta(days=10)
    for card in (cards[0], cards[1], cards[3]):
        logistics = await data.get_logistics(owner, tracking_no=card["tracking_no"])
        times = [datetime.fromisoformat(item["occurred_at"]) for item in logistics["trace"]]
        assert times == sorted(times)
        assert all(item["occurred_at"].endswith("Z") for item in logistics["trace"])
        assert logistics["trace"][-1]["location"] == logistics["location"]


async def test_initialization_preserves_existing_records_and_dates(owned_data):
    factory, (owner, _) = owned_data
    data = OrderData(factory)
    before = await data.list_orders(owner)
    async with factory() as session, session.begin():
        row = await session.get(Order, (owner, "1001"))
        row.product = "人工修改的商品"
    async with factory() as session, session.begin():
        assert not await OrderRepository(session).seed_demo(owner, at=datetime(2040, 1, 1, tzinfo=timezone.utc))
    after = await OrderData(factory).list_orders(owner)
    assert after[0]["product"] == "人工修改的商品"
    assert [row["created_at"] for row in before] == [row["created_at"] for row in after]


async def test_concurrent_first_use_produces_one_complete_consistent_order_set(owned_data):
    factory, (owner, _) = owned_data
    first, second = await asyncio.gather(OrderData(factory).list_orders(owner), OrderData(factory).list_orders(owner))
    assert first == second
    assert len(first) == 4
    async with factory() as session:
        assert await session.scalar(select(func.count()).select_from(Order).where(Order.user_id == owner)) == 4
        assert await session.scalar(select(func.count()).select_from(Shipment).where(Shipment.user_id == owner)) == 3
        tracking = select(Shipment.tracking_no).where(Shipment.user_id == owner)
        assert await session.scalar(select(func.count()).select_from(ShipmentEvent).where(ShipmentEvent.tracking_no.in_(tracking))) == 8
