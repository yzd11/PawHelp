"""Explicit test data source for network-free workflow acceptance cases."""

from app.repositories.orders import OrderNotFound


class TestToolCatalog:
    """Explicit test-only snapshot. Production never falls back to builtin logistics."""
    __test__ = False
    async def snapshot(self, runtime):
        from app.tools.business import build_business_tools
        from app.tools.registry import ToolRegistry
        return ToolRegistry([t for t in build_business_tools(runtime) if t.name != "query_faq"], runtime=runtime)


class FakeOrderData:
    async def list_orders(self, user_id):
        return [await self.get_order(user_id, str(oid)) for oid in range(1001, 1005)]

    async def get_order(self, user_id, order_id):
        if order_id not in {"1001", "1002", "1003", "1004"}:
            raise OrderNotFound()
        unpaid = order_id == "1003"
        return {
            "order_id": order_id, "label": f"演示订单 {order_id}", "product": "猫爬架",
            "item": "猫爬架", "amount": "299.00", "is_demo": True,
            "status": "pending_payment" if unpaid else "delivered",
            "status_label": "待付款" if unpaid else "已签收",
            "tracking_no": None if unpaid else f"DEMO{order_id}",
            "created_at": "2026-09-25T12:00:00Z",
            "delivered_at": None if unpaid else "2026-09-28T12:00:00Z",
        }

    async def get_logistics(self, user_id, *, order_id=None, tracking_no=None):
        oid = order_id or (tracking_no or "").removeprefix("DEMO")
        order = await self.get_order(user_id, oid)
        if tracking_no and tracking_no != order["tracking_no"]:
            raise OrderNotFound()
        return {
            "order_id": oid, "tracking_no": order["tracking_no"],
            "status": "not_shipped" if order["status"] == "pending_payment" else "delivered",
            "location": None if order["status"] == "pending_payment" else "收货地址",
            "trace": [], "is_demo": True,
        }
