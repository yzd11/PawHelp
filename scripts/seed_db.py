"""Idempotently insert the fixed development FAQ fixtures."""

import asyncio

from sqlalchemy import select

from app.config import get_settings
from app.database.models import FAQ
from app.database.session import create_async_engine_from_settings, create_session_factory


FAQ_FIXTURES = (
    {
        "id": 1,
        "question": "退货政策是什么？",
        "answer": "商品在符合条件时可申请退货，具体以售后页面说明为准。",
        "category": "退货政策",
    },
    {
        "id": 2,
        "question": "配送费用如何计算？",
        "answer": "配送费用会根据订单金额、收货地区和活动规则计算。",
        "category": "配送费用",
    },
)


async def main() -> None:
    """Insert missing stable FAQ rows and commit exactly one unit of work."""

    engine = create_async_engine_from_settings(get_settings())
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            fixture_ids = [fixture["id"] for fixture in FAQ_FIXTURES]
            existing_ids = set(
                (await session.scalars(select(FAQ.id).where(FAQ.id.in_(fixture_ids)))).all()
            )
            session.add_all(
                [FAQ(**fixture) for fixture in FAQ_FIXTURES if fixture["id"] not in existing_ids]
            )
            await session.commit()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
