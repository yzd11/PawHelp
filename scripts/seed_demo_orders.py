"""Initialize stable demo orders without overwriting existing data."""

import argparse
import asyncio
from uuid import UUID

from sqlalchemy import select

from app.config import Settings
from app.database.models import Conversation
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.repositories.orders import OrderRepository


async def seed(user_id: str | None) -> None:
    engine = create_async_engine_from_settings(Settings())
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            users = [user_id] if user_id else list(await session.scalars(select(Conversation.user_id).distinct()))
        initialized = 0
        for owner in users:
            async with factory() as session, session.begin():
                initialized += int(await OrderRepository(session).seed_demo(owner))
        print(f"Checked {len(users)} users; initialized {initialized}; existing records unchanged.")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=UUID, help="Demo user UUID; default: all existing conversation owners")
    args = parser.parse_args()
    asyncio.run(seed(str(args.user_id) if args.user_id else None))


if __name__ == "__main__":
    main()
