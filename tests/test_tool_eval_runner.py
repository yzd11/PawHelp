"""Deterministic and live-service contracts for labelled tool-call evaluation."""

from contextlib import AbstractAsyncContextManager
from random import Random
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.database.models import Conversation, Message, Ticket
from app.services.tool_chat import ToolAwareChatService
from scripts.run_tool_evals import EvalCase, load_cases, run_cases, score_case
import scripts.run_tool_evals as runner
from tests.fakes import FakeToolCallingChatModel


DATABASE_URL = "mysql+asyncmy://unused"


def make_settings() -> Settings:
    return Settings(
        chat_base_url="http://127.0.0.1:9/v1",
        openai_model="eval-test",
        openai_api_key="local-nonsecret-key",
        database_url=DATABASE_URL,
        shop_name="测试店",
        context_token_budget=4096,
        tool_timeout_seconds=1,
        tool_max_attempts=1,
    )


def logistics_call_chunk() -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "query_logistics",
                "args": '{"order_id":"1001"}',
                "id": "eval-call",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )


def logistics_case() -> EvalCase:
    return EvalCase(
        case_id="logistics",
        message="private prompt 订单 1001 的物流到哪了",
        expected_tool="query_logistics",
        expected_outcome="answer",
    )


class _FailingSession:
    """Fail one chosen persistence open without exposing its private text."""

    async def __aenter__(self) -> "_FailingSession":
        raise RuntimeError("private persistence exception secret=do-not-print")

    async def __aexit__(self, *args: object) -> None:
        return None


class FailingSessionFactory:
    """Delegate to real MySQL sessions except at one reproducible call boundary."""

    def __init__(
        self, inner: async_sessionmaker[AsyncSession], failing_calls: set[int]
    ) -> None:
        self.inner = inner
        self.failing_calls = failing_calls
        self.calls = 0

    def __call__(self) -> AbstractAsyncContextManager[Any]:
        self.calls += 1
        if self.calls in self.failing_calls:
            return _FailingSession()
        return self.inner()


async def table_counts(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[int, int, int]:
    async with session_factory() as session:
        return (
            await session.scalar(select(func.count()).select_from(Conversation)) or 0,
            await session.scalar(select(func.count()).select_from(Message)) or 0,
            await session.scalar(select(func.count()).select_from(Ticket)) or 0,
        )


def test_logistics_case_requires_done_succeeded_and_grounded_answer() -> None:
    """An error after grounded text must not satisfy a labelled answer case."""

    result = score_case(
        expected_tool="query_logistics",
        expected_outcome="answer",
        observed_tool="query_logistics",
        answer="订单 1001 正在运输，包裹位于配送中心",
        tool_result='{"order_id":"1001","status":"in_transit","location":"配送中心"}',
        terminal_event="error",
        tool_succeeded=True,
    )

    assert result.passed is False
    assert result.classification == "terminal_error"


def test_logistics_accepts_chinese_status_paraphrase_but_rejects_refund_claim() -> None:
    """The fixed logistics facts must be complete and free of contradictory claims."""

    common = {
        "expected_tool": "query_logistics",
        "expected_outcome": "answer",
        "observed_tool": "query_logistics",
        "tool_result": '{"order_id":"1001","status":"accepted","location":"配送中心"}',
        "terminal_event": "done",
        "tool_succeeded": True,
    }

    faithful = score_case(answer="订单 1001 已揽收，目前在配送中心", **common)
    invented = score_case(answer="订单 1001 已退款，记录位于配送中心", **common)

    assert faithful.passed is True
    assert invented == type(invented)(False, "ungrounded_answer")


def test_logistics_rejects_negated_status_and_invented_eta_but_allows_location_paraphrase() -> None:
    """Fixed logistics rules must preserve polarity and reject ETA claims absent from data."""

    common = {
        "expected_tool": "query_logistics",
        "expected_outcome": "answer",
        "observed_tool": "query_logistics",
        "terminal_event": "done",
        "tool_succeeded": True,
    }
    negated = score_case(
        answer="订单 1001 尚未派送，目前在配送中心",
        tool_result='{"order_id":"1001","status":"out_for_delivery","location":"配送中心"}',
        **common,
    )
    invented_eta = score_case(
        answer="订单 1001 正在运输，目前在配送中心，预计两小时后送达",
        tool_result='{"order_id":"1001","status":"in_transit","location":"配送中心"}',
        **common,
    )
    paraphrased_location = score_case(
        answer="订单 1001 正在运输，目前还在路上",
        tool_result='{"order_id":"1001","status":"in_transit","location":"运输途中"}',
        **common,
    )

    assert negated.passed is False
    assert invented_eta.passed is False
    assert paraphrased_location.passed is True


def test_logistics_fixed_case_handles_colloquial_negation_and_unknown_eta() -> None:
    common = {
        "expected_tool": "query_logistics",
        "expected_outcome": "answer",
        "observed_tool": "query_logistics",
        "tool_result": '{"order_id":"1001","status":"out_for_delivery","location":"配送中心"}',
        "terminal_event": "done",
        "tool_succeeded": True,
    }

    assert score_case(answer="订单 1001 还没派送，目前在配送中心", **common).passed is False
    assert score_case(answer="订单 1001 正在派送，大概两天能送到", **common).passed is False
    assert score_case(
        answer="订单 1001 正在派送，目前在配送中心，但无法预计送达时间", **common
    ).passed is True


def test_return_policy_accepts_paraphrase_but_rejects_added_refund_fact() -> None:
    """FAQ evaluation must validate policy facts rather than exact answer copying."""

    common = {
        "expected_tool": "query_faq",
        "expected_outcome": "answer",
        "observed_tool": "query_faq",
        "tool_result": (
            '{"matches":[{"question":"退货政策是什么？",'
            '"answer":"商品在符合条件时可申请退货，具体以售后页面说明为准。",'
            '"category":"退货政策"}]}'
        ),
        "terminal_event": "done",
        "tool_succeeded": True,
    }

    faithful = score_case(answer="符合条件的商品可以申请退货，具体规则以售后页面为准。", **common)
    invented = score_case(
        answer="商品在符合条件时可申请退货，具体以售后页面说明为准，并且会全额退款。",
        **common,
    )

    assert faithful.passed is True
    assert invented.passed is False
    assert invented.classification == "ungrounded_answer"


def test_return_policy_allows_conditions_wording_but_rejects_unsupported_benefits() -> None:
    """Policy-specific facts must not be confused with an FAQ miss or new benefits."""

    common = {
        "expected_tool": "query_faq",
        "expected_outcome": "answer",
        "observed_tool": "query_faq",
        "tool_result": (
            '{"matches":[{"question":"退货政策是什么？",'
            '"answer":"商品在符合条件时可申请退货，具体以售后页面说明为准。",'
            '"category":"退货政策"}]}'
        ),
        "terminal_event": "done",
        "tool_succeeded": True,
    }

    faithful = score_case(answer="退货条件请以售后页面说明为准，符合条件即可申请退货。", **common)
    invented = score_case(answer="符合条件可申请退货，规则见售后页面，并支持七天无理由换货。", **common)

    assert faithful.passed is True
    assert invented.passed is False


def test_return_policy_fixed_case_rejects_unconditional_or_seven_day_rights() -> None:
    common = {
        "expected_tool": "query_faq",
        "expected_outcome": "answer",
        "observed_tool": "query_faq",
        "tool_result": (
            '{"matches":[{"question":"退货政策是什么？",'
            '"answer":"商品在符合条件时可申请退货，具体以售后页面说明为准。",'
            '"category":"退货政策"}]}'
        ),
        "terminal_event": "done",
        "tool_succeeded": True,
    }

    assert score_case(
        answer="不符合条件也可以申请退货，具体规则见售后页面", **common
    ).passed is False
    assert score_case(
        answer="符合条件可申请退货，售后页面说明七日内都能办理", **common
    ).passed is False


def test_postage_case_requires_semantic_hit_and_grounded_answer() -> None:
    common = {
        "expected_tool": "query_faq",
        "expected_outcome": "postage_answer",
        "observed_tool": "query_faq",
        "tool_result": (
            '{"matches":[{"question":"配送费用如何计算？",'
            '"answer":"配送费用根据订单金额、收货地区和活动规则计算，结算页展示实际运费。",'
            '"category":"配送服务"}]}'
        ),
        "terminal_event": "done",
        "tool_succeeded": True,
    }

    faithful = score_case(
        answer="运费会根据订单金额和收货地区计算，请以结算页展示为准。",
        **common,
    )
    invented = score_case(answer="配送费固定收费 99 元。", **common)

    assert faithful.passed is True
    assert faithful.classification == "postage_answer"
    assert invented.passed is False


async def test_runner_rejects_partial_post_tool_answer_and_cleans_rows(
    mysql_session, capsys
) -> None:
    """A convergence error after a real tool result must exit nonzero and clean up."""

    session_factory = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    model = FakeToolCallingChatModel(
        selector_items=[logistics_call_chunk()],
        final_items=[
            "订单 1001 正在运输，包裹位于配送中心",
            RuntimeError("private upstream exception secret=do-not-print"),
        ],
    )
    service = ToolAwareChatService(
        settings=make_settings(),
        model=model,
        session_factory=session_factory,
        rng=Random(7),
    )
    before = await table_counts(session_factory)

    exit_code = await run_cases(
        [logistics_case()], make_settings(), service=service, session_factory=session_factory
    )

    stdout = capsys.readouterr().out
    assert exit_code == 1
    assert "terminal=error" in stdout
    assert "pass=False" in stdout
    assert "private prompt" not in stdout
    assert "正在运输" not in stdout
    assert "in_transit" not in stdout
    assert "secret=" not in stdout
    assert await table_counts(session_factory) == before


async def test_runner_rejects_full_answer_when_final_persistence_fails_and_cleans_rows(
    mysql_session, capsys
) -> None:
    """A final-row persistence error after complete text cannot become acceptance."""

    inner = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    session_factory = inner
    class FinalPersistenceFailure(ToolAwareChatService):
        async def _persist_assistant(self, conversation_id, content, *, tool_calls=None):
            if tool_calls is None:
                raise RuntimeError("private final persistence failure secret=do-not-print")
            return await super()._persist_assistant(conversation_id, content, tool_calls=tool_calls)
    model = FakeToolCallingChatModel(
        selector_items=[logistics_call_chunk()],
        final_items=["订单 1001 正在运输，包裹位于配送中心"],
    )
    service = FinalPersistenceFailure(
        settings=make_settings(),
        model=model,
        session_factory=session_factory,
        rng=Random(7),
    )
    before = await table_counts(inner)

    exit_code = await run_cases(
        [logistics_case()], make_settings(), service=service, session_factory=session_factory
    )

    stdout = capsys.readouterr().out
    assert exit_code == 1
    assert "terminal=error" in stdout
    assert "pass=False" in stdout
    assert "secret=" not in stdout
    assert await table_counts(inner) == before


async def test_runner_success_is_done_succeeded_safe_and_repeatable(
    mysql_session, capsys
) -> None:
    """Repeated real-service evaluation runs must pass cleanly without DB residue."""

    session_factory = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    before = await table_counts(session_factory)
    for _ in range(2):
        model = FakeToolCallingChatModel(
            selector_items=[logistics_call_chunk()],
            final_items=["订单 1001 已签收，包裹位于收货地址"],
        )
        service = ToolAwareChatService(
            settings=make_settings(),
            model=model,
            session_factory=session_factory,
            rng=Random(7),
        )
        assert await run_cases(
            [logistics_case()],
            make_settings(),
            service=service,
            session_factory=session_factory,
        ) == 0

    stdout = capsys.readouterr().out
    assert stdout.count("terminal=done") == 2
    assert stdout.count("pass=True") == 2
    assert "private prompt" not in stdout
    assert "订单 1001 正在运输" not in stdout
    assert "order_id" not in stdout
    assert await table_counts(session_factory) == before
    assert model.bound_tool_names == (
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    )


def test_main_hides_model_setup_exception_and_returns_stable_failure(
    monkeypatch, capsys
) -> None:
    """CLI startup errors must not leak a traceback or private provider failure."""

    async def fail_runner(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("private model setup failure secret=do-not-print")

    monkeypatch.setattr(runner, "Settings", lambda: make_settings())
    monkeypatch.setattr(runner, "run_cases", fail_runner)

    assert runner.main() == 1
    output = capsys.readouterr().out
    assert output == "UNAVAILABLE: evaluation runner could not start\n"
    assert "secret=" not in output