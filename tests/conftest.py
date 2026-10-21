"""Shared fixtures. Live-MySQL tests run only against an explicit test database.

Set TEST_DATABASE_URL to a disposable MySQL schema migrated to head; without it
those tests are skipped so a developer's real data is never touched.
"""

import os
from collections.abc import AsyncIterator

import pytest_asyncio
import pytest
from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, String, Table, Text, UniqueConstraint, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine




def _settings_env_names() -> set[str]:
    from pydantic import AliasChoices

    from app.config import Settings

    names = set()
    for name, field in Settings.model_fields.items():
        names.add(name.upper())
        if isinstance(field.validation_alias, AliasChoices):
            names.update(str(choice).upper() for choice in field.validation_alias.choices)
    return names


@pytest.fixture(autouse=True)
def isolate_test_settings(monkeypatch):
    """Tests see only the settings they pass, never the developer's .env.

    Some dependencies (pymilvus) load .env into os.environ at import time.
    Fake-model tests must also neither persist usage nor export to Langfuse.
    """
    for name in _settings_env_names():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")


def pytest_sessionstart(session):
    # Also cover module-level create_app()/cached Settings during collection,
    # before per-test fixtures start. This affects only the pytest process.
    # Fake required settings let collection pass without a .env (as in CI) and
    # override a developer's .env, so local runs see the same values.
    os.environ.update({
        "CHAT_BASE_URL": "http://localhost/v1",
        "OPENAI_MODEL": "fake",
        "OPENAI_API_KEY": "fake",
        "DATABASE_URL": "mysql+asyncmy://unused",
        "OBSERVABILITY_ENABLED": "false",
    })
    # Never read the developer's .env either: a local EVIDENCE_JUDGE=jev plus a real key
    # would send test questions to TypeSafe. An explicit _env_file still works.
    from app.config import Settings

    Settings.model_config["env_file"] = None


@pytest_asyncio.fixture
async def mysql_session() -> AsyncIterator[AsyncSession]:
    """Yield a session on the explicit test database; tests clean up their own rows."""

    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("set TEST_DATABASE_URL to run live-MySQL tests")
    engine = create_async_engine(url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            yield session
            await session.rollback()
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def context_sessions():
    """Ephemeral SQL adapter: exercises ORM transactions, not MySQL row-lock semantics.

    Production still uses MySQL. Live MySQL migration is a separate acceptance.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    meta = MetaData()
    Table("conversations", meta, Column("id", Integer, primary_key=True), Column("user_id", String), Column("status", String, server_default="进行中"),
          Column("summary", Text), Column("summary_upto_msg_id", Integer), Column("layer1_from_msg_id", Integer), Column("title", String),
          Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")), Column("updated_at", DateTime, server_default=text("CURRENT_TIMESTAMP")))
    Table("messages", meta, Column("id", Integer, primary_key=True), Column("conversation_id", Integer), Column("role", String), Column("content", Text),
          Column("tool_calls", JSON), Column("tool_call_id", String), Column("tool_trace", JSON), Column("suggested_actions", JSON), Column("source_refs", JSON),
          Column("related_order_id", String), Column("reply_to_message_id", Integer), Column("retrieved_chunks", JSON), Column("feedback", String),
          Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")))
    Table("conversation_summaries", meta, Column("id", Integer, primary_key=True), Column("conversation_id", Integer), Column("seq", Integer),
          Column("from_msg_id", Integer), Column("upto_msg_id", Integer), Column("content", Text), Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
          UniqueConstraint("conversation_id", "seq"))
    async with engine.begin() as connection:
        await connection.run_sync(meta.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()
