"""Safe validation, timeout, and retry contracts for tool execution."""

import asyncio

import pytest
from langchain_core.tools import tool

from app.tools.executor import (
    InvalidToolArguments,
    ToolExecutionFailed,
    ToolExecutor,
    TransientToolError,
    UnknownToolError,
)
from app.tools.registry import ToolRegistry
from app.tools.schemas import OrderArgs


@pytest.fixture
def success_tool():
    """A tool whose result is safe to return directly to the model loop."""

    @tool(args_schema=OrderArgs)
    async def query_order(order_id: str) -> str:
        """Return a controlled order result."""

        return f'{{"order_id":"{order_id}"}}'

    return query_order


@pytest.fixture
def executor(success_tool) -> ToolExecutor:
    return ToolExecutor(ToolRegistry([success_tool]))


@pytest.fixture
def flaky_tool_call():
    return {"id": "call-1", "name": "query_order", "args": {"order_id": "1001"}}


async def test_executor_returns_correlated_result_after_one_successful_attempt(
    executor: ToolExecutor,
    flaky_tool_call: dict[str, object],
) -> None:
    """Dropping a provider correlation ID or changing success attempt count must fail."""

    result = await executor.execute(flaky_tool_call)

    assert (result.tool_call_id, result.name, result.content, result.attempts) == (
        "call-1",
        "query_order",
        '{"order_id":"1001"}',
        1,
    )


async def test_transient_failure_retries_once(flaky_tool_call: dict[str, object]) -> None:
    """Removing the transient retry branch must leave this call failed after attempt one."""

    calls = 0

    @tool(args_schema=OrderArgs)
    async def query_order(order_id: str) -> str:
        """Fail once before returning a normal result."""

        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientToolError("upstream temporary details")
        return '{"status":"ok"}'

    result = await ToolExecutor(ToolRegistry([query_order])).execute(flaky_tool_call)

    assert (result.attempts, result.tool_call_id, result.content) == (
        2,
        "call-1",
        '{"status":"ok"}',
    )


async def test_default_executor_uses_three_seconds_and_retries_a_timeout_twice(
    flaky_tool_call: dict[str, object],
) -> None:
    """Changing default bounds or skipping a timeout retry must fail this boundary."""

    calls = 0
    never = asyncio.Event()

    @tool(args_schema=OrderArgs)
    async def query_order(order_id: str) -> str:
        """Remain pending until the executor deadline expires."""

        nonlocal calls
        calls += 1
        await never.wait()
        return '{"status":"late"}'

    executor = ToolExecutor(ToolRegistry([query_order]))
    assert (executor.timeout_seconds, executor.max_attempts) == (3, 2)

    result = await executor.execute(flaky_tool_call)

    assert calls == 2
    assert result.status == "超时" and result.code == "tool_timeout"
    assert result.attempts == 2
    assert "late" not in result.content


async def test_unknown_name_and_invalid_arguments_are_rejected_before_invocation(
    executor: ToolExecutor,
) -> None:
    """Allowing an unknown name or malformed args into a tool must fail closed."""

    unknown = await executor.execute({"id": "call-2", "name": "read_secrets", "args": {}})
    invalid = await executor.execute({"id": "call-3", "name": "query_order", "args": {"order_id": ""}})
    assert unknown.status == "权限拒绝" and unknown.code == "unknown_tool"
    assert invalid.status == "校验拦下" and invalid.code == "invalid_tool_arguments"
    assert "read_secrets" not in unknown.content
    assert "order_id" in invalid.content  # parameter path helps model correct, not offending value


async def test_permanent_failure_is_safe_and_not_retried(
    flaky_tool_call: dict[str, object],
) -> None:
    """Treating every exception as transient must cause this permanent failure to retry."""

    calls = 0

    @tool(args_schema=OrderArgs)
    async def query_order(order_id: str) -> str:
        """Raise an unsafe permanent business error."""

        nonlocal calls
        calls += 1
        raise RuntimeError("database secret=do-not-expose")

    result = await ToolExecutor(ToolRegistry([query_order])).execute(flaky_tool_call)

    assert calls == 1
    assert result.code == "tool_failed" and result.status == "失败"
    assert "secret" not in result.content


async def test_cancellation_is_never_converted_to_a_safe_error(
    flaky_tool_call: dict[str, object],
) -> None:
    """Catching cancellation as a retryable failure would prevent task cancellation."""

    @tool(args_schema=OrderArgs)
    async def query_order(order_id: str) -> str:
        """Propagate cooperative cancellation."""

        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await ToolExecutor(ToolRegistry([query_order])).execute(flaky_tool_call)
