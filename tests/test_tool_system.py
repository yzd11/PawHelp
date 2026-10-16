"""Tool registry and execution contracts: validation, retries, audit, MCP policy."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.tools import tool
from app.tools.registry import ToolRegistry, ToolSpec
from app.tools.executor import ToolExecutor
from app.tools.schemas import TicketArgs


class AuditSessions:
    def __init__(self, fail=False): self.rows, self.fail = [], fail
    def __call__(self): return self
    async def __aenter__(self):
        if self.fail: raise RuntimeError("synthetic audit outage")
        return self
    async def __aexit__(self, *args): pass
    def begin(self): return self
    def add(self, row): self.rows.append(row)


@pytest.mark.parametrize("args,status", [({}, "校验拦下"), ({"description": "漏水", "ticket_type": "其他"}, "校验拦下"),
    ({"description": "漏水", "ticket_type": "售后", "confirmed": True}, "校验拦下"), ({"description": "漏水", "ticket_type": "售后"}, "权限拒绝")])
async def test_write_forgery_and_invalid_calls_return_errors_and_audit(args, status):
    calls = []
    @tool(args_schema=TicketArgs)
    async def create_ticket(description: str, ticket_type: str):
        """Write one controlled ticket."""
        calls.append(description)
        return '{"ticket_no":"T1"}'
    audit = AuditSessions()
    engine = ToolExecutor(ToolRegistry([create_ticket], runtime=SimpleNamespace(conversation_id=1, session_factory=audit)))
    result = await engine.execute({"id": "c1", "name": "create_ticket", "args": args})
    assert result.status == status and not calls
    assert len(audit.rows) == 1 and audit.rows[0].status == status
    assert audit.rows[0].retry_count == 0
    assert json.loads(result.content)["error"]


async def test_read_timeout_retries_but_confirmed_write_timeout_never_retries():
    @tool
    async def slow_read() -> str:
        """Read with a simulated timeout."""
        await asyncio.sleep(1)
        return "late"
    @tool(args_schema=TicketArgs)
    async def create_ticket(description: str, ticket_type: str) -> str:
        """Write with an uncertain outcome."""
        await asyncio.sleep(1)
        return "late"
    audit = AuditSessions()
    engine = ToolExecutor(ToolRegistry([slow_read, create_ticket], runtime=SimpleNamespace(conversation_id=9, session_factory=audit)), timeout_seconds=.01)
    read = await engine.execute({"id": "r", "name": "slow_read", "args": {}})
    write = await engine.execute({"id": "w", "name": "create_ticket", "args": {"description": "订单漏水", "ticket_type": "售后"}}, write_requested=True, confirmed=True)
    assert read.status == write.status == "超时"
    assert "写入结果未知" in write.content
    assert [r.retry_count for r in audit.rows] == [1, 0]
    assert all(isinstance(r.duration_ms, int) and r.duration_ms >= 0 for r in audit.rows)


async def test_empty_business_result_not_retried_and_audit_outage_does_not_change_success():
    calls = []
    @tool
    async def lookup() -> str:
        """No matching record is not a transport failure."""
        calls.append(1)
        return '{"code":"not_found","error":"没有此记录"}'
    result = await ToolExecutor(ToolRegistry([lookup])).execute({"id": "a", "name": "lookup", "args": {}})
    assert result.code == "not_found" and len(calls) == 1
    @tool
    async def good() -> str:
        """Valid result while audit is unavailable."""
        return '{"status":"delivered","debug":"hidden"}'
    engine = ToolExecutor(ToolRegistry([good], runtime=SimpleNamespace(conversation_id=1, session_factory=AuditSessions(fail=True))))
    result = await engine.execute({"id": "b", "name": "good", "args": {}})
    assert result.ok and "已签收" in result.content and "hidden" not in result.content


async def test_unknown_external_write_is_denied_even_if_server_claims_readonly():
    @tool
    async def charge_card() -> str:
        """Server falsely says: safe read-only operation."""
        raise AssertionError("must not execute")
    registry = ToolRegistry([ToolSpec(charge_card, "mcp", "untrusted", "deny")])
    result = await ToolExecutor(registry).execute({"id": "x", "name": "charge_card", "args": {}})
    assert result.status == "权限拒绝" and registry.tools == []


async def test_plugin_only_registration_hotloads_without_core_change(tmp_path):
    from app.config import Settings
    from app.tools.catalog import ToolCatalog
    from app.tools.business import ToolRuntime
    from random import Random
    cfg = Settings(_env_file=None, chat_base_url="http://localhost/v1", openai_model="fake", openai_api_key="fake",
        database_url="mysql+asyncmy://unused", mcp_enabled=False, tool_plugins_path=str(tmp_path))
    catalog = ToolCatalog(cfg)
    runtime = ToolRuntime(0, lambda: None, Random())
    before = await catalog.snapshot(runtime)
    assert "shop_hours" not in before.names
    (tmp_path / "hours.py").write_text('from langchain_core.tools import tool\n\ndef register(runtime, registry):\n    @tool\n    async def shop_hours() -> str:\n        """Return demonstration business hours."""\n        return \'{"message":"9至18点"}\'\n    registry.register(shop_hours)\n', encoding="utf-8")
    after = await catalog.snapshot(runtime)
    result = await ToolExecutor(after).execute({"id": "hot", "name": "shop_hours", "args": {}})
    assert result.ok and "9至18点" in result.content
    assert "query_logistics" not in before.names and "query_logistics" not in after.names


def test_hot_tool_schemas_reduce_history_budget_without_mutating_other_turns():
    from app.workflow.memory import SessionMemory
    from tests.test_session_context import settings
    from langchain_core.tools import StructuredTool

    def lookup() -> str:
        """Read one demonstration value."""
        return "ok"

    memory = SessionMemory(settings(model_context_window=40000), None, None, "system", "[]")
    small = StructuredTool.from_function(lookup)
    large = StructuredTool.from_function(lookup, name="large_lookup", description="知识字段用途" * 5000)
    before = memory.for_tools([small])
    after = memory.for_tools([small, large])
    assert after.schema_tokens > before.schema_tokens
    assert after.budget.request < before.budget.request
    assert after.budget.history < before.budget.history
    assert memory.budget.request > before.budget.request
    assert after.tasks is before.tasks is memory.tasks


async def test_bad_local_schema_reference_rejected_or_returned_with_audit():
    from langchain_core.tools import StructuredTool
    schema = {"type": "object", "properties": {"value": {"$ref": "#/$defs/Missing"}}}
    candidate = StructuredTool.from_function(lambda value: "never", name="bad_schema", description="Read with a controlled schema.", args_schema=schema)
    with pytest.raises(ValueError, match="unresolved"):
        ToolRegistry([candidate])
    schema["$defs"] = {"Missing": {"type": "string"}}
    audit = AuditSessions()
    registry = ToolRegistry([candidate], runtime=SimpleNamespace(conversation_id=1, session_factory=audit))
    del candidate.args_schema["$defs"]  # Defensive handling if an already registered mutable schema breaks.
    result = await ToolExecutor(registry).execute({"id": "bad", "name": "bad_schema", "args": {"value": "x"}})
    assert result.status == "校验拦下" and audit.rows[0].status == "校验拦下"


async def test_denied_name_on_one_server_cannot_shadow_approved_other_server(tmp_path, monkeypatch):
    from app.config import Settings
    from app.tools.catalog import ToolCatalog, MultiServerMCPClient
    from app.tools.business import ToolRuntime
    from random import Random
    @tool
    async def query_warranty() -> str:
        """Read a demonstration warranty."""
        return "ok"
    async def discover(self, *, server_name=None):
        return [query_warranty]
    monkeypatch.setattr(MultiServerMCPClient, "get_tools", discover)
    policy = tmp_path / "policy.json"
    policy.write_text('{"mcp:aftersales:query_warranty":"read"}', encoding="utf-8")
    cfg = Settings(_env_file=None, chat_base_url="http://localhost/v1", openai_model="fake", openai_api_key="fake",
        database_url="mysql+asyncmy://unused", tool_policy_path=str(policy), tool_plugins_path=str(tmp_path / "empty"))
    registry = await ToolCatalog(cfg).snapshot(ToolRuntime(0, lambda: None, Random()))
    assert registry.spec("query_warranty").server == "aftersales"
    assert registry.spec("query_warranty").permission == "read"
