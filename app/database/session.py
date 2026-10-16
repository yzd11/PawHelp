"""Async SQLAlchemy engine and session-factory construction."""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings


def create_async_engine_from_settings(settings: Settings) -> AsyncEngine:
    """Create an async MySQL engine without exposing its credential."""

    return create_async_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
    )


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Return independent async sessions that retain committed ORM values."""

    return async_sessionmaker(engine, expire_on_commit=False)
