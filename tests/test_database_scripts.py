"""Executable database lifecycle scripts without a real networked MySQL server."""

from collections.abc import Callable
from typing import Any

import scripts.init_db as init_db
import scripts.seed_db as seed_db


class FakeConnection:
    def __init__(self) -> None:
        self.sync_functions: list[Callable[..., object]] = []

    async def run_sync(self, function: Callable[..., object]) -> None:
        self.sync_functions.append(function)


class FakeBegin:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, *args: object) -> None:
        return None


class FakeEngine:
    def __init__(self) -> None:
        self.connection = FakeConnection()
        self.dispose_calls = 0

    def begin(self) -> FakeBegin:
        return FakeBegin(self.connection)

    async def dispose(self) -> None:
        self.dispose_calls += 1


class FakeScalarResult:
    def __init__(self, values: set[int]) -> None:
        self.values = values

    def all(self) -> list[int]:
        return sorted(self.values)


class FakeSeedStore:
    def __init__(self) -> None:
        self.ids: set[int] = set()
        self.inserted_rows: list[tuple[int, str, str, str]] = []
        self.commit_calls = 0


class FakeSeedSession:
    def __init__(self, store: FakeSeedStore) -> None:
        self.store = store

    async def __aenter__(self) -> "FakeSeedSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def scalars(self, statement: object) -> FakeScalarResult:
        return FakeScalarResult(self.store.ids)

    def add_all(self, rows: list[Any]) -> None:
        for row in rows:
            self.store.ids.add(row.id)
            self.store.inserted_rows.append(
                (row.id, row.question, row.answer, row.category)
            )

    async def commit(self) -> None:
        self.store.commit_calls += 1


class FakeSessionFactory:
    def __init__(self, store: FakeSeedStore) -> None:
        self.store = store

    def __call__(self) -> FakeSeedSession:
        return FakeSeedSession(self.store)


def test_init_db_upgrades_the_database_to_head(monkeypatch: Any) -> None:
    """The established init command delegates schema evolution to Alembic."""

    calls: list[tuple[object, str]] = []
    monkeypatch.setattr(
        init_db.command,
        "upgrade",
        lambda config, revision: calls.append((config, revision)),
    )

    init_db.main()

    assert len(calls) == 1
    assert calls[0][1] == "head"
    assert calls[0][0].config_file_name.endswith("alembic.ini")


async def test_seed_db_commits_once_per_run_and_skips_existing_fixture_ids(
    monkeypatch: Any,
) -> None:
    """Catch duplicate FAQ fixtures or an uncommitted/disposed seed unit of work."""

    engine = FakeEngine()
    store = FakeSeedStore()
    session_factory = FakeSessionFactory(store)
    monkeypatch.setattr(seed_db, "get_settings", lambda: object())
    monkeypatch.setattr(
        seed_db,
        "create_async_engine_from_settings",
        lambda settings: engine,
    )
    monkeypatch.setattr(
        seed_db,
        "create_session_factory",
        lambda received_engine: session_factory,
    )

    await seed_db.main()
    await seed_db.main()

    assert store.inserted_rows == [
        (1, "退货政策是什么？", "商品在符合条件时可申请退货，具体以售后页面说明为准。", "退货政策"),
        (2, "配送费用如何计算？", "配送费用会根据订单金额、收货地区和活动规则计算。", "配送费用"),
    ]
    assert store.commit_calls == 2
    assert engine.dispose_calls == 2
