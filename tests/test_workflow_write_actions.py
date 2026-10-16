"""Live MySQL + graph/API confirmation contract, deterministic model only."""
import json
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import Settings
from app.database.models import Conversation, Message, Ticket, ToolAuditLog, RequestObservation
from app.main import create_app
from app.workflow.graph import WorkflowChatService
from tests.order_fakes import FakeOrderData, TestToolCatalog
from tests.workflow_fakes import LabelledModel, LabelledSearch, tool_call


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.mark.parametrize("confirmed,uncertain", [(True, False), (False, False), (True, True)])
async def test_preview_refresh_resume_owned_once_and_cancel_audit(mysql_session, tmp_path, monkeypatch, confirmed, uncertain):
    sessions = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    user = str(uuid4())
    class SwitchingModel(LabelledModel):
        def with_structured_output(self, schema, **kwargs):
            if schema.__name__ == "IntentDecision" and self.agent_calls:
                self.intent = "售后"  # Description follows an explicit write request.
            return super().with_structured_output(schema, **kwargs)

    model = SwitchingModel("人工", ["请描述需要登记的问题。", tool_call("create_ticket", '{"description":"订单1001漏水，承诺三天退款（编造草稿）","ticket_type":"售后"}', "ticket-proposal-1")])
    cfg = Settings(_env_file=None, chat_base_url="http://localhost/v1", openai_model="fake", openai_api_key="fake",
                   database_url="mysql+asyncmy://unused", model_context_window=100000, mcp_enabled=False,
                   tool_timeout_seconds=1 if uncertain else 3)
    if uncertain:
        import asyncio
        import app.workflow.tool_agent as agent_module
        original_builder = agent_module.build_business_tools
        def builder(runtime):
            tools = original_builder(runtime)
            write_tool = next(tool for tool in tools if tool.name == "create_ticket")
            original_write = write_tool.coroutine
            async def slow_write(description, ticket_type):
                await original_write(description, ticket_type)
                await asyncio.sleep(5)  # Commit succeeded, response disappeared: genuinely uncertain.
            write_tool.coroutine = slow_write
            return tools
        monkeypatch.setattr(agent_module, "build_business_tools", builder)
    checkpoint_path = str(tmp_path / "ticket-confirmation.sqlite3")
    saver_context = AsyncSqliteSaver.from_conn_string(checkpoint_path)
    checkpointer = await saver_context.__aenter__()
    service = WorkflowChatService(settings=cfg, model=model, session_factory=sessions, checkpointer=checkpointer,
        order_data=FakeOrderData(), catalog=TestToolCatalog(), knowledge_search=LabelledSearch(True))
    app = create_app(settings=cfg, model=model, session_factory=sessions, tool_chat_service=service)
    cid = None
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post("/v1/chat/stream", json={"user_id": user, "message": "帮我建个工单"})
            cid = next(e["conversation_id"] for e in events(first) if "conversation_id" in e)
            assert "请描述" in first.text and "ticket_confirmation" not in first.text
            second = await client.post("/v1/chat/stream", json={"user_id": user, "conversation_id": cid, "message": "订单1001饮水机漏水，请售后联系"})
            assert "ticket_confirmation" in second.text
            pending = await client.get(f"/v1/conversations/{cid}/pending-ticket", headers={"X-Demo-User-Id": user})
            preview = pending.json()["confirmation"]
            assert preview["description"] == "订单1001饮水机漏水，请售后联系"
            assert model.agent_calls == 2
            async with sessions() as session:
                assert not list(await session.scalars(select(Ticket).where(Ticket.conversation_id == cid)))
                assert not list(await session.scalars(select(ToolAuditLog).where(ToolAuditLog.conversation_id == cid)))
            blocked = await client.post("/v1/chat/stream", json={"user_id": user, "conversation_id": cid, "message": "你好"})
            assert blocked.status_code == 409
            payload = {"user_id": user, "message_id": preview["message_id"], "proposal_id": preview["proposal_id"], "confirmed": confirmed}
            path = f"/v1/conversations/{cid}/confirm-ticket"
            wrong = await client.post(path, json={**payload, "user_id": str(uuid4())})
            assert wrong.status_code == 404
            wrong_id = await client.post(path, json={**payload, "proposal_id": "forged"})
            assert wrong_id.status_code == 409
            # Close/reopen the actual SQLite file, not merely the service object.
            await service.close()
            await saver_context.__aexit__(None, None, None)
            saver_context = AsyncSqliteSaver.from_conn_string(checkpoint_path)
            checkpointer = await saver_context.__aenter__()
            restored = WorkflowChatService(settings=cfg, model=model, session_factory=sessions, checkpointer=checkpointer,
                order_data=FakeOrderData(), catalog=TestToolCatalog(), knowledge_search=LabelledSearch(True))
            restored_app = create_app(settings=cfg, model=model, session_factory=sessions, tool_chat_service=restored)
            async with AsyncClient(transport=ASGITransport(app=restored_app), base_url="http://test") as resumed_client:
                result = await resumed_client.post(path, json=payload)
                duplicate = await resumed_client.post(path, json=payload)
                assert duplicate.status_code == 409
                assert "done" in result.text
                assert ("结果暂时无法确认" if uncertain else "TICKET-" if confirmed else "已取消") in result.text
                assert "ticket_confirmation" not in result.text  # No automatic new proposal on uncertain write.
                assert model.agent_calls == 2  # interrupt did not replay selection
                state = await resumed_client.get(f"/v1/conversations/{cid}/pending-ticket", headers={"X-Demo-User-Id": user})
                assert state.json()["pending"] is False
            async with sessions() as session:
                tickets = list(await session.scalars(select(Ticket).where(Ticket.conversation_id == cid)))
                logs = list(await session.scalars(select(ToolAuditLog).where(ToolAuditLog.conversation_id == cid)))
                assert len(tickets) == int(confirmed)
                assert len(logs) == 1 and logs[0].status == ("超时" if uncertain else "成功" if confirmed else "权限拒绝")
                assert logs[0].retry_count == 0
                assert logs[0].tool_call_id == preview["proposal_id"]
                assert logs[0].arguments["description"] == preview["description"]
    finally:
        await service.close()
        await saver_context.__aexit__(None, None, None)
        if cid:
            async with sessions() as session, session.begin():
                for cls in (RequestObservation, ToolAuditLog, Ticket, Message): await session.execute(delete(cls).where(cls.conversation_id == cid))
                await session.execute(delete(Conversation).where(Conversation.id == cid))


async def test_invalid_tool_args_feed_back_then_agent_corrects_without_stream_failure():
    from tests.workflow_fakes import run_example
    _, model, _, stream = await run_example("订单", "查订单1001", [
        tool_call("query_order", '{"order_id":123}', "invalid"),
        tool_call("query_order", '{"order_id":"1001"}', "corrected"), "已查询订单1001。"])
    assert stream[-1].name == "done" and model.agent_calls == 3
    assert [t["status"] for t in stream[-1].data["tool_trace"]] == ["failed", "succeeded"]
    assert "校验拦下" in str(model.agent_histories[1][-1].content)


async def test_handoff_only_does_not_authorize_model_ticket():
    from tests.workflow_fakes import run_example
    _, model, _, stream = await run_example("人工", "转人工，不要建工单", [
        tool_call("create_ticket", '{"description":"用户未请求","ticket_type":"咨询"}', "unauthorized"),
        "您可以点击转人工。[建议选项:转人工]"])
    assert not any(e.name == "ticket_confirmation" for e in stream)
    assert stream[-1].name == "done" and stream[-1].data["suggested_actions"] == ["转人工"]
    assert stream[-1].data["tool_trace"][0]["status"] == "failed"


@pytest.mark.parametrize("question", ["查订单1001是否还在保", "订单1001在保吗", "订单1001退货进度到哪了"])
async def test_dynamic_aftersales_queries_bypass_policy_gate(question):
    from tests.workflow_fakes import run_example
    _, _, search, stream = await run_example("售后", question, ["需要使用演示售后查询工具。"], confident=False,
                                             decision={"task_kind": "progress_query"})
    assert stream[-1].name == "done" and not search.calls


@pytest.mark.parametrize("question,previous,expected", [
    ("帮我建个工单", False, True),
    ("转人工，不要建工单", True, False),
    ("如果我请你创建工单会怎样", False, False),
    ("工单是什么", False, False),
    ("订单1001饮水机漏水", True, True),
])
def test_write_request_is_explicit_not_a_hypothetical(question, previous, expected):
    from app.workflow.tool_agent import ticket_requested
    assert ticket_requested(question, previous) is expected


async def test_bare_ticket_request_cannot_preview_a_fabricated_description():
    from tests.workflow_fakes import run_example
    _, model, _, stream = await run_example("人工", "帮我建个工单", [
        tool_call("create_ticket", '{"description":"订单1001漏水，用户从未说过","ticket_type":"售后"}', "fabricated"),
        "请先描述您遇到的具体问题。",
    ])
    assert not any(event.name == "ticket_confirmation" for event in stream)
    assert stream[-1].name == "done" and model.agent_calls == 2
    assert "尚未提供具体问题" in model.agent_histories[-1][-1].content


async def test_execution_cap_rejects_and_audits_each_proposed_call(monkeypatch):
    from langchain_core.messages import AIMessageChunk
    from app.tools.executor import ToolExecutor
    from tests.workflow_fakes import run_example
    audit = []
    async def capture(self, result, spec, args, error, duration):
        audit.append(result)
    monkeypatch.setattr(ToolExecutor, "_audit", capture)
    proposals = AIMessageChunk(content="", tool_call_chunks=[
        {"name": "query_order", "args": '{"order_id":"1001"}', "id": f"cap-{i}", "index": i, "type": "tool_call_chunk"}
        for i in range(4)
    ])
    _, model, _, stream = await run_example("订单", "查订单1001", [proposals])
    assert stream[-1].name == "done" and model.agent_calls == 1
    assert len(audit) == 4 and all(result.status == "权限拒绝" and result.attempts == 0 for result in audit)
    assert len(stream[-1].data["tool_trace"]) == 4


async def test_failed_order_prefetch_is_not_success_or_refund_permission():
    from random import Random
    from langchain_core.messages import HumanMessage
    from app.context import PreparedContext
    from app.services.tool_chat import PreparedTurn
    from app.tools.business import ToolRuntime
    from tests.workflow_fakes import InMemoryWorkflow
    class UnavailableOrders(FakeOrderData):
        async def list_orders(self, user_id):
            return await FakeOrderData().list_orders(user_id)
        async def get_order(self, user_id, order_id):
            raise ConnectionError("controlled outage")
    model = LabelledModel("退款退货", ["目前无法确认订单情况，通用政策可参考证据。[1]"])
    service = InMemoryWorkflow(model, LabelledSearch(True))
    service._orders = UnavailableOrders()
    turn = PreparedTurn(conversation_id=1, user_id="test", user_message_id=1, raw_question="订单1001能退吗",
        prepared_context=PreparedContext((HumanMessage(content="订单1001能退吗"),), 20, 0),
        tool_runtime=ToolRuntime(1, lambda: None, Random(), user_id="test", order_data=service._orders))
    stream = [event async for event in service.stream_turn(turn)]
    assert stream[-1].name == "done"
    assert stream[-1].data["tool_trace"][0]["status"] == "failed"
    assert "申请退款" not in stream[-1].data["suggested_actions"]
    await service.close()
