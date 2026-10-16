"""Persistent owned order reads with one atomic, non-overwriting demo initialization."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.repositories.orders import OrderRepository


class OrderData:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @asynccontextmanager
    async def _repository(self, user_id: str) -> AsyncIterator[OrderRepository]:
        async with self._session_factory() as session:
            async with session.begin():
                await OrderRepository(session).seed_demo(user_id)
            # A concurrent no-op upsert does not advance a REPEATABLE READ snapshot.
            # Commit initialization before opening a fresh read transaction.
            async with session.begin():
                yield OrderRepository(session)

    async def list_orders(self, user_id: str) -> list[dict[str, object]]:
        async with self._repository(user_id) as repository:
            return await repository.list_owned(user_id)

    async def get_order(self, user_id: str, order_id: str) -> dict[str, object]:
        async with self._repository(user_id) as repository:
            return await repository.get_owned(user_id, order_id)

    async def get_logistics(
        self, user_id: str, *, order_id: str | None = None, tracking_no: str | None = None,
    ) -> dict[str, object]:
        async with self._repository(user_id) as repository:
            return await repository.get_logistics(user_id, order_id=order_id, tracking_no=tracking_no)
