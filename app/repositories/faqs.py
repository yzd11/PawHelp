"""Lexical FAQ lookup without user-controlled LIKE wildcards."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import FAQ


class FAQRepository:
    """Search the fixed FAQ corpus through one caller-owned session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search_contains(self, keyword: str, *, limit: int = 5) -> list[FAQ]:
        """Return a bounded literal containment match on FAQ questions."""

        rows = await self._session.scalars(
            select(FAQ)
            .where(FAQ.question.contains(keyword, autoescape=True))
            .order_by(FAQ.id)
            .limit(limit)
        )
        return list(rows)
