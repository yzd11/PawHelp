"""Real official SDK processes + official adapters, no LLM/network charges."""
import asyncio
import json
import os
from pathlib import Path
from random import Random
import socket
import subprocess
import sys

import httpx
import pytest

from app.config import Settings
from app.tools.business import ToolRuntime
from app.tools.catalog import ToolCatalog
from app.tools.executor import ToolExecutor
from tests.order_fakes import FakeOrderData

ROOT = Path(__file__).resolve().parents[1]


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(module, env, *, extended=False):
    # Controlled extension emulates adding a @tool on Server only, no app restart.
    code = "from " + module + " import mcp\n"
    if extended:
        code += 'from mcp.server.fastmcp import Context\nfrom mcp_servers.scenario import read_scenario\n@mcp.tool()\nasync def query_delivery_options(tracking_no: str, ctx: Context) -> str:\n    """Query demonstration delivery options for an owned tracking number."""\n    scenario = read_scenario(ctx)\n    if scenario["tracking_no"] != tracking_no: raise ValueError("wrong tracking")\n    return \'{"message":"演示支持配送中心自提","is_demo":true}\'\n'
    code += 'mcp.run(transport="streamable-http")'
    return subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


async def ready(url, process):
    async with httpx.AsyncClient() as client:
        for _ in range(120):
            if process.poll() is not None: raise AssertionError("MCP process failed at startup")
            try:
                await client.get(url, timeout=.3)
                return
            except httpx.TransportError: await asyncio.sleep(.1)
    raise AssertionError("MCP process readiness timeout")


def stop(process):
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=5)


async def test_two_mcp_processes_owned_data_dynamic_discovery_and_permission(tmp_path):
    logistics, aftersales = port(), port()
    env = {**os.environ, "MCP_LOGISTICS_PORT": str(logistics), "MCP_AFTERSALES_PORT": str(aftersales), "MCP_DEMO_SECRET": "test-local-fixture"}
    processes = [start("mcp_servers.logistics_server", env), start("mcp_servers.aftersales_server", env)]
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"mcp:logistics:query_logistics": "read", "mcp:aftersales:query_warranty": "read", "mcp:aftersales:query_return_progress": "read"}), encoding="utf-8")
    cfg = Settings(_env_file=None, chat_base_url="http://localhost/v1", openai_model="fake", openai_api_key="fake", database_url="mysql+asyncmy://unused",
        mcp_logistics_url=f"http://127.0.0.1:{logistics}/mcp", mcp_aftersales_url=f"http://127.0.0.1:{aftersales}/mcp", mcp_demo_secret="test-local-fixture", tool_policy_path=str(policy), tool_plugins_path=str(tmp_path / "empty"))
    runtime = ToolRuntime(0, lambda: None, Random(), user_id="test-user", order_data=FakeOrderData())
    try:
        await asyncio.gather(*(ready(url, proc) for url, proc in zip((cfg.mcp_logistics_url, cfg.mcp_aftersales_url), processes)))
        catalog = ToolCatalog(cfg)
        first = await catalog.snapshot(runtime)
        assert first.spec("query_logistics").source == "mcp"
        assert {"query_logistics", "query_warranty", "query_return_progress"} <= first.names
        engine = ToolExecutor(first)
        result = await engine.execute({"id": "logistics", "name": "query_logistics", "args": {"tracking_no": "DEMO1001"}})
        assert result.ok and json.loads(result.content)["status"] == "delivered"
        wrong = await engine.execute({"id": "wrong", "name": "query_logistics", "args": {"tracking_no": "other-user"}})
        assert wrong.code == "not_found" and wrong.attempts == 1
        for name in ("query_warranty", "query_return_progress"):
            result = await engine.execute({"id": name, "name": name, "args": {"order_id": "1001"}})
            assert result.ok and json.loads(result.content)["is_demo"]
        # The active Agent receives/exercises the same actual MCP tools, not a builtin substitute.
        from langchain_core.messages import HumanMessage
        from app.context import PreparedContext
        from app.services.tool_chat import PreparedTurn
        from tests.workflow_fakes import InMemoryWorkflow, LabelledModel, LabelledSearch, tool_call
        model = LabelledModel("物流", [
            tool_call("query_order", '{"order_id":"1001"}', "order-mcp"),
            tool_call("query_logistics", '{"tracking_no":"DEMO1001"}', "logistics-mcp"),
            "演示订单已签收，物流查询已完成。",
        ])
        service = InMemoryWorkflow(model, LabelledSearch(False))
        service._catalog = catalog
        turn = PreparedTurn(conversation_id=1, user_id="test-user", user_message_id=1,
            raw_question="先查订单1001，再查物流", prepared_context=PreparedContext((HumanMessage(content="先查订单1001，再查物流"),), 20, 0),
            tool_runtime=ToolRuntime(1, lambda: None, Random(), user_id="test-user", order_data=FakeOrderData()))
        stream = [event async for event in service.stream_turn(turn)]
        assert stream[-1].name == "done" and model.agent_calls == 3
        trace = stream[-1].data["tool_trace"]
        assert trace[1]["source"] == "mcp" and trace[1]["mcp_server"] == "logistics"
        assert "delivered" in model.agent_histories[-1][-1].content
        await service.close()
        stop(processes[0])
        processes[0] = start("mcp_servers.logistics_server", env, extended=True)
        await ready(cfg.mcp_logistics_url, processes[0])
        discovered = await catalog.snapshot(runtime)  # SAME Client catalog; only Server restarted.
        assert "query_delivery_options" in discovered.names and discovered.spec("query_delivery_options").permission == "deny"
        denied = await ToolExecutor(discovered).execute({"id": "denied", "name": "query_delivery_options", "args": {"tracking_no": "DEMO1001"}})
        assert denied.status == "权限拒绝"
        policy.write_text(json.dumps({"mcp:logistics:query_logistics": "read", "mcp:logistics:query_delivery_options": "read", "mcp:aftersales:query_warranty": "read"}), encoding="utf-8")
        approved = await catalog.snapshot(runtime)
        result = await ToolExecutor(approved).execute({"id": "hot", "name": "query_delivery_options", "args": {"tracking_no": "DEMO1001"}})
        assert result.ok and "自提" in result.content
        stop(processes[0])
        isolated = await catalog.snapshot(runtime)
        assert "query_warranty" in isolated.names and "query_logistics" not in isolated.names
    finally:
        for proc in processes: stop(proc)
