"""One persistent data source for demo cards, business tools and refund checks."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib

from sqlalchemy import and_, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Order, Shipment, ShipmentEvent


class OrderNotFound(LookupError):
    """Missing and foreign orders deliberately have the same public result."""


def utc_text(value: datetime | None) -> str | None:
    return None if value is None else value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


_ORDER_LABELS = {
    "pending_payment": "待付款", "processing": "待发货",
    "shipped": "已发货", "delivered": "已签收",
}
_SHIPMENT_LABELS = {
    "accepted": "已揽件", "in_transit": "运输中",
    "out_for_delivery": "派送中", "delivered": "已签收",
}


def _snapshot(order: Order, shipment: Shipment | None) -> dict[str, object]:
    return {
        "order_id": order.order_id, "product": order.product, "item": order.product,
        "label": f"{'演示订单' if order.is_demo else '订单'} {order.order_id}",
        "amount": str(order.amount), "currency": "CNY", "status": order.status,
        "status_label": _ORDER_LABELS[order.status], "is_demo": bool(order.is_demo),
        "created_at": utc_text(order.created_at), "shipped_at": utc_text(order.shipped_at),
        "delivered_at": utc_text(order.delivered_at),
        "tracking_no": None if shipment is None else shipment.tracking_no,
        "estimated_delivery_at": None if shipment is None else utc_text(shipment.estimated_delivery_at),
    }


class OrderRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def seed_demo(self, user_id: str, *, at: datetime | None = None) -> bool:
        """Initialize one user's fixtures atomically; never reset an existing order set.

        Dates are anchored once, in UTC. A concurrent initializer may wait for the
        first INSERT, then uses the persisted winner's dates for all shipment rows.
        The caller commits the entire order/shipments/events transaction together.
        """

        if not user_id:
            raise ValueError("a trusted user identity is required")
        if await self._session.scalar(select(Order.order_id).where(Order.user_id == user_id).limit(1)):
            return False
        now = (at or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)
        fixtures = (
            ("1001", "猫爬架", "299.00", "delivered", 5, 4, 2),
            ("1002", "宠物饮水机", "159.00", "shipped", 2, 1, None),
            ("1003", "猫抓板", "39.00", "pending_payment", 0, None, None),
            ("1004", "猫爬架", "299.00", "delivered", 15, 13, 10),
        )
        for oid, product, amount, status, created_days, shipped_days, delivered_days in fixtures:
            statement = insert(Order).values(
                user_id=user_id, order_id=oid, product=product, amount=Decimal(amount), status=status,
                is_demo=True, created_at=now - timedelta(days=created_days),
                shipped_at=None if shipped_days is None else now - timedelta(days=shipped_days),
                delivered_at=None if delivered_days is None else now - timedelta(days=delivered_days),
            )
            await self._session.execute(statement.on_duplicate_key_update(order_id=statement.inserted.order_id))
        # Locking reads see the winning transaction even with MySQL REPEATABLE READ.
        orders = list(await self._session.scalars(
            select(Order).where(Order.user_id == user_id).order_by(Order.order_id).with_for_update()
        ))
        for order in orders:
            if order.shipped_at is None:
                continue
            tracking = "DEMO" + hashlib.sha256(f"{user_id}\x1f{order.order_id}".encode()).hexdigest()[:24]
            delivered = order.delivered_at is not None
            last_at = order.delivered_at if delivered else order.shipped_at + timedelta(hours=12)
            location = "收货地址" if delivered else "配送中心"
            statement = insert(Shipment).values(
                tracking_no=tracking, user_id=user_id, order_id=order.order_id, carrier="演示快递",
                status="delivered" if delivered else "in_transit", location=location,
                estimated_delivery_at=None if delivered else order.shipped_at + timedelta(days=3),
                updated_at=last_at,
            )
            await self._session.execute(statement.on_duplicate_key_update(tracking_no=statement.inserted.tracking_no))
            events = [(order.shipped_at, "发货仓", "包裹已揽件")]
            if delivered:
                events.append((order.shipped_at + timedelta(hours=12), "配送中心", "包裹运输中"))
            events.append((last_at, location, "包裹已签收" if delivered else "包裹运输中"))
            for sequence, (occurred_at, event_location, description) in enumerate(events, start=1):
                statement = insert(ShipmentEvent).values(
                    tracking_no=tracking, sequence=sequence, occurred_at=occurred_at,
                    location=event_location, description=description,
                )
                await self._session.execute(statement.on_duplicate_key_update(sequence=statement.inserted.sequence))
        return True

    def _orders_query(self, user_id: str):
        return select(Order, Shipment).outerjoin(
            Shipment, and_(Order.user_id == Shipment.user_id, Order.order_id == Shipment.order_id),
        ).where(Order.user_id == user_id)

    async def list_owned(self, user_id: str) -> list[dict[str, object]]:
        rows = await self._session.execute(self._orders_query(user_id).order_by(Order.order_id))
        return [_snapshot(order, shipment) for order, shipment in rows]

    async def get_owned(self, user_id: str, order_id: str) -> dict[str, object]:
        row = (await self._session.execute(self._orders_query(user_id).where(Order.order_id == order_id))).first()
        if row is None:
            raise OrderNotFound("没有找到您的这笔订单")
        return _snapshot(*row)

    async def get_logistics(
        self, user_id: str, *, order_id: str | None = None, tracking_no: str | None = None,
    ) -> dict[str, object]:
        statement = self._orders_query(user_id)
        if order_id:
            statement = statement.where(Order.order_id == order_id)
        if tracking_no:
            statement = statement.where(Shipment.tracking_no == tracking_no)
        if not order_id and not tracking_no:
            raise OrderNotFound("没有找到您的这笔订单")
        row = (await self._session.execute(statement)).first()
        if row is None:
            raise OrderNotFound("没有找到您的这笔订单")
        order, shipment = row
        if shipment is None:
            return {
                "order_id": order.order_id, "tracking_no": None, "status": "not_shipped",
                "status_label": "尚未发货", "location": None, "trace": [], "is_demo": bool(order.is_demo),
            }
        events = await self._session.scalars(select(ShipmentEvent).where(
            ShipmentEvent.tracking_no == shipment.tracking_no,
        ).order_by(ShipmentEvent.occurred_at, ShipmentEvent.sequence))
        return {
            "order_id": order.order_id, "tracking_no": shipment.tracking_no,
            "carrier": shipment.carrier, "status": shipment.status,
            "status_label": _SHIPMENT_LABELS[shipment.status], "location": shipment.location,
            "estimated_delivery_at": utc_text(shipment.estimated_delivery_at),
            "updated_at": utc_text(shipment.updated_at), "is_demo": bool(order.is_demo),
            "trace": [{"occurred_at": utc_text(event.occurred_at), "location": event.location,
                       "description": event.description} for event in events],
        }
