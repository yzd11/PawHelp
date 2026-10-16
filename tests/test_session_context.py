"""Layered session context, summaries, checkpoints and history; no paid calls."""
import asyncio
import json
from uuid import uuid4

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import select, MetaData, Table, Column, Integer, String, Text, JSON, DateTime, UniqueConstraint, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.config import Settings
from app.database.models import Conversation, ConversationSummary, Message
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository
from app.schemas import ChatRequest
from app.services.history import to_langchain_messages, to_public_messages
from app.workflow.graph import WorkflowChatService
from app.workflow.memory import SessionMemory
from tests.fakes import fake_settings as settings
from tests.order_fakes import FakeOrderData, TestToolCatalog


class Model:
    def __init__(self):
        self.summary_wait = asyncio.Event()
        self.summary_wait.set()
        self.summary_started = asyncio.Event()
        self.histories = []
        self.summary_batches = []
        self.fail_summary = False

    def with_structured_output(self, schema, **kwargs):
        parent = self
        class Runnable:
            async def ainvoke(self, messages):
                if schema.__name__ == "SummaryResult":
                    parent.summary_batches.append(messages)
                    parent.summary_started.set()
                    await parent.summary_wait.wait()
                    if parent.fail_summary:
                        raise ValueError("synthetic summary failure")
                    payload = json.loads(messages[-1].content)
                    text = json.dumps(payload["new_interval"], ensure_ascii=False)
                    return {"content": "用户询问订单1001物流，随后希望申请退款，尚未处理。" if "1001" in text else ""}
                if schema.__name__ == "ResolvedQuestion":
                    question = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage) and not m.additional_kwargs.get("context_material")), messages[-1].content)
                    if "最开始" in question:
                        assert any("订单1001" in str(m.content) for m in messages if m.additional_kwargs.get("context_material"))
                        assert sum(bool(m.additional_kwargs.get("current_question")) for m in messages) == 1
                        return {"question": "订单1001当前情况"}
                    return {"question": question}
                return {"intent": "闲聊", "confidence": 0.99}
        return Runnable()

    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages):
        self.histories.append(messages)
        yield AIMessageChunk(content="订单1001尚需确认。")


def test_budget_and_copy_do_not_modify_raw_messages():
    memory = SessionMemory(settings(model_context_window=24000, model_max_output_tokens=2000, tool_result_max_tokens=1200), None, None, "系统规则", "工具定义")
    assert memory.budget.history == 24000 - memory.budget.fixed - memory.budget.peak
    assert memory.budget.layer1 + memory.budget.layer2 == memory.budget.history
    raw = [HumanMessage(content="用户原话" * 100), AIMessage(content="回答" * 100),
           AIMessage(content="", tool_calls=[{"name": "query_order", "id": "call-1", "args": {}}]),
           ToolMessage(content="x" * 4000, name="query_order", tool_call_id="call-1")]
    compressed = memory.compress(raw)
    assert compressed[0].content == raw[0].content
    assert len(compressed[1].content) == 60
    assert "历史工具结果" in compressed[3].content
    assert len(raw[1].content) == 200 and len(raw[3].content) == 4000
    observation = json.loads(memory.tool_observation(json.dumps({"tracking_no": "DEMO1001", "order_id": "1001", "detail": "x" * 20000})))
    assert observation["tracking_no"] == "DEMO1001" and observation["truncated"]


@pytest.mark.asyncio
async def test_24_turns_full_checkpoint_and_sidebar_history(context_sessions, caplog):
    sessions = context_sessions
    model = Model()
    service = WorkflowChatService(catalog=TestToolCatalog(), settings=settings(model_context_window=1_000_000), model=model,
                                  session_factory=sessions, order_data=FakeOrderData())
    user = str(uuid4())
    cid = None
    caplog.set_level("INFO", logger="app.workflow.memory")
    for index in range(24):
        request = ChatRequest(user_id=user, conversation_id=cid, message=f"第{index}轮问候" + "你好" * 50)
        turn = await service.prepare_turn(request)
        cid = turn.conversation_id
        events = [event async for event in service.stream_turn(turn)]
        assert events[-1].name == "done"
    checkpoint = await service._checkpointer.aget_tuple({"configurable": {"thread_id": f"{user}:{cid}"}})
    full = checkpoint.checkpoint["channel_values"]["messages"]
    assert len(full) == 48 and len({m.id for m in full}) == 48
    async with sessions() as session, session.begin():
        rows = await MessageRepository(session).list_ordered(cid)
        assert len(rows) == 48 and len(to_public_messages(rows, {})) == 48
        conversation = await ConversationRepository(session).get_owned(cid, user)
        assert conversation.layer1_from_msg_id is None and conversation.summary_upto_msg_id is None
        another = await ConversationRepository(session).create(user)
        listed = await ConversationRepository(session).list_owned(user)
        assert [row["id"] for row in listed] == [another.id, cid]
        assert listed[1]["preview"].startswith("第0轮")
        assert await ConversationRepository(session).list_owned(str(uuid4())) == []
    assert len(model.summary_batches) == 0
    assert caplog.text.count("history_ctx") == 24
    await service.close()


@pytest.mark.asyncio
async def test_async_summary_cascade_failure_and_immutable_segments(context_sessions, caplog):
    sessions = context_sessions
    model = Model()
    model.summary_wait.clear()
    memory = SessionMemory(settings(model_context_window=24000, max_user_input_tokens=600,
                                    model_max_output_tokens=500, tool_result_max_tokens=1200), sessions, model, "系统" * 4000, "工具")
    assert memory.budget.history >= memory.budget.steady
    user = str(uuid4())
    rows = []
    async with sessions() as session, session.begin():
        conversation = await ConversationRepository(session).create(user)
        repository = MessageRepository(session)
        for index in range(22):
            rows.append(await repository.add_user(conversation.id, f"订单1001第{index}轮物流之后退款" + "业务说明" * 50))
            rows.append(await repository.add_assistant(conversation.id, "订单1001待核实" + "补充说明" * 50))
        current = await repository.add_user(conversation.id, "最开始那个订单后来怎么说")
    full = to_langchain_messages([*rows, current])
    caplog.set_level("INFO", logger="app.workflow.memory")
    view = await memory.prepare(conversation.id, user, full, current.id)
    assert view.layer1_from > 0 and view.layer2_tokens > memory.budget.layer2
    memory.schedule(conversation.id, user, view)
    await asyncio.wait_for(model.summary_started.wait(), 2)
    # Caller proceeds while inference is deliberately blocked.
    async with sessions() as session:
        fresh = await session.get(Conversation, conversation.id)
        assert fresh.summary_upto_msg_id is None
    assert memory.count(view.messages) <= memory.budget.history
    model.summary_wait.set()
    await asyncio.gather(*list(memory.tasks.values()))
    async with sessions() as session:
        segments = list(await session.scalars(select(ConversationSummary).where(ConversationSummary.conversation_id == conversation.id)))
        fresh = await session.get(Conversation, conversation.id)
    assert len(segments) == 1 and segments[0].content.startswith("用户询问订单1001")
    assert fresh.summary_upto_msg_id == segments[0].upto_msg_id
    first = segments[0].content
    # A failed subsequent task cannot alter committed coverage or the first segment.
    model.fail_summary = True
    async with sessions() as session, session.begin():
        repository = MessageRepository(session)
        for index in range(22):
            rows.append(await repository.add_user(conversation.id, "订单1001新一轮" + "业务说明" * 50))
            rows.append(await repository.add_assistant(conversation.id, "补充说明" * 50))
        current2 = await repository.add_user(conversation.id, "最开始那个订单")
    view2 = await memory.prepare(conversation.id, user, to_langchain_messages([*rows, current2]), current2.id)
    before = fresh.summary_upto_msg_id
    memory.schedule(conversation.id, user, view2)
    await asyncio.gather(*list(memory.tasks.values()))
    async with sessions() as session:
        fresh = await session.get(Conversation, conversation.id)
        assert fresh.summary_upto_msg_id == before and segments[0].content == first
    assert "summary failed" in caplog.text and "summary done" in caplog.text
    model.fail_summary = False
    memory.schedule(conversation.id, user, view2)
    await asyncio.gather(*list(memory.tasks.values()))
    # Stale competing work cannot insert a duplicate or move an already advanced anchor.
    await memory._summarize(conversation.id, user, view2)
    async with sessions() as session:
        all_segments = list(await session.scalars(select(ConversationSummary).where(ConversationSummary.conversation_id == conversation.id).order_by(ConversationSummary.seq)))
        assert [s.seq for s in all_segments] == [1, 2]
        assert all_segments[0].content == first and all_segments[1].from_msg_id > all_segments[0].upto_msg_id
    assert "stale_boundary" in caplog.text
    await memory.close()


@pytest.mark.asyncio
async def test_mysql_summary_and_new_tool_trace_persistence(mysql_session):
    """Live MySQL acceptance after 0007 migration; test rows roll back."""
    user = str(uuid4())
    conversation = await ConversationRepository(mysql_session).create(user)
    row = await MessageRepository(mysql_session).add_user(conversation.id, "订单1001查询")
    trace = [{"call_id": "a", "name": "query_order", "label": "查询订单", "status": "succeeded"}]
    final = await MessageRepository(mysql_session).add_assistant(conversation.id, "订单待核实", tool_trace=trace)
    mysql_session.add(ConversationSummary(conversation_id=conversation.id, seq=1, from_msg_id=row.id, upto_msg_id=final.id, content="用户询问订单1001"))
    conversation.summary = "用户询问订单1001"
    conversation.summary_upto_msg_id = final.id
    conversation.layer1_from_msg_id = final.id
    await mysql_session.flush()
    assert len(to_public_messages([row, final], {"query_order": "查询订单"})[1].tool_trace) == 1
    listed = await ConversationRepository(mysql_session).list_owned(user)
    assert listed[0]["has_summary"] is True and listed[0]["preview"] == row.content
    assert len(await MessageRepository(mysql_session).list_ordered(conversation.id)) == 2


@pytest.mark.asyncio
async def test_owned_list_and_history_api_aliases(context_sessions):
    import httpx
    from fastapi import FastAPI
    from app.api import create_api_router
    user = str(uuid4())
    async with context_sessions() as session, session.begin():
        conversation = await ConversationRepository(session).create(user)
        await MessageRepository(session).add_user(conversation.id, "第一条原话")
    app = FastAPI()
    app.include_router(create_api_router(settings(), Model(), context_sessions, object(), object()))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        headers = {"X-Demo-User-Id": user}
        listing = await client.get("/api/conversations", headers=headers)
        assert listing.status_code == 200 and listing.json()[0]["preview"] == "第一条原话"
        for prefix in ("api", "v1"):
            response = await client.get(f"/{prefix}/conversations/{conversation.id}/messages", headers=headers)
            assert response.status_code == 200 and response.json()[0]["content"] == "第一条原话"
            foreign = await client.get(f"/{prefix}/conversations/{conversation.id}/messages", headers={"X-Demo-User-Id": str(uuid4())})
            assert foreign.status_code == 404


@pytest.mark.asyncio
async def test_same_session_serialization_and_summary_reference(context_sessions):
    model = Model()
    service = WorkflowChatService(catalog=TestToolCatalog(), settings=settings(model_context_window=1_000_000), model=model, session_factory=context_sessions, order_data=FakeOrderData())
    user = str(uuid4())
    first = await service.prepare_turn(ChatRequest(user_id=user, message="订单1001的物流"))
    competing = asyncio.create_task(service.prepare_turn(ChatRequest(user_id=user, conversation_id=first.conversation_id, message="最开始那个订单")))
    await asyncio.sleep(0)
    assert not competing.done()  # No user row can interleave before the prior assistant row.
    assert [event async for event in service.stream_turn(first)][-1].name == "done"
    second = await asyncio.wait_for(competing, 2)
    async with context_sessions() as session, session.begin():
        conversation = await session.get(Conversation, first.conversation_id)
        rows = await MessageRepository(session).list_ordered(first.conversation_id)
        assert [row.role for row in rows] == ["user", "assistant", "user"]
        # A committed early summary can resolve an identifier after raw history is covered.
        conversation.summary = "第1段：用户询问订单1001的物流，并想申请退款，尚未解决。"
        conversation.summary_upto_msg_id = rows[1].id
        conversation.layer1_from_msg_id = rows[1].id
    events = [event async for event in service.stream_turn(second)]
    assert events[-1].name == "done"
    snapshot = await service._checkpointer.aget_tuple({"configurable": {"thread_id": f"{user}:{first.conversation_id}"}})
    assert snapshot.checkpoint["channel_values"]["question"] == "订单1001当前情况"
    await service.close()


@pytest.mark.asyncio
async def test_partial_tool_failure_keeps_checkpoint_and_safe_trace(context_sessions, caplog):
    from tests.workflow_fakes import tool_call
    class PartialModel(Model):
        def __init__(self): super().__init__(); self.calls = 0
        def with_structured_output(self, schema, **kwargs):
            if schema.__name__ == "IntentDecision":
                class Runnable:
                    async def ainvoke(self, messages): return {"intent": "订单", "confidence": .99}
                return Runnable()
            return super().with_structured_output(schema, **kwargs)
        async def astream(self, messages):
            self.histories.append(messages)
            self.calls += 1
            if self.calls == 1:
                yield tool_call("query_order", '{"order_id":"1001"}', "tool-1")
            else:
                raise RuntimeError("synthetic later model error")
    model = PartialModel()
    service = WorkflowChatService(catalog=TestToolCatalog(), settings=settings(model_context_window=32768), model=model, session_factory=context_sessions, order_data=FakeOrderData())
    caplog.set_level("INFO", logger="app.workflow.memory")
    user = str(uuid4())
    turn = await service.prepare_turn(ChatRequest(user_id=user, message="查询订单1001"))
    events = [event async for event in service.stream_turn(turn)]
    assert events[-1].name == "done" and "部分查询" in events[-1].data["content"]
    checkpoint = await service._checkpointer.aget_tuple({"configurable": {"thread_id": f"{user}:{turn.conversation_id}"}})
    raw = checkpoint.checkpoint["channel_values"]["messages"]
    assert len([m for m in raw if isinstance(m, ToolMessage)]) == 1
    async with context_sessions() as session:
        rows = await MessageRepository(session).list_ordered(turn.conversation_id)
        assert [row.role for row in rows] == ["user", "assistant"]
        assert rows[-1].tool_trace[0]["call_id"] == "tool-1"
    assert sum(isinstance(m, SystemMessage) for m in model.histories[0]) == 1
    assert "call=1" in caplog.text and "call=2" in caplog.text and f"turn={turn.user_message_id}" in caplog.text
    await service.close()


def test_unknown_window_requires_explicit_configuration(caplog):
    memory = SessionMemory(settings(), None, None, "system", "tools")
    assert memory.budget.window == 8192 and not memory.window_known
    assert "未知模型窗口" in caplog.text


@pytest.mark.asyncio
async def test_24_agent_turns_mysql_and_sqlite_checkpoint_restart(mysql_session, tmp_path, caplog):
    """Real storage + fake inference. Only generated owned rows are cleaned up."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from sqlalchemy import delete
    from tests.workflow_fakes import tool_call
    class AgentModel(Model):
        def __init__(self): super().__init__(); self.calls = 0
        def with_structured_output(self, schema, **kwargs):
            if schema.__name__ == "IntentDecision":
                class Runnable:
                    async def ainvoke(self, messages): return {"intent":"订单", "confidence":.99}
                return Runnable()
            return super().with_structured_output(schema, **kwargs)
        async def astream(self, messages):
            self.histories.append(list(messages))
            phase = self.calls % 3
            self.calls += 1
            if phase == 0:
                yield tool_call("query_order", '{"order_id":"1001"}', f"call-{self.calls}")
            elif phase == 1:
                tracking = json.loads(messages[-1].content)["tracking_no"]
                yield tool_call("query_logistics", json.dumps({"tracking_no":tracking}), f"call-{self.calls}")
            else:
                yield AIMessageChunk(content="本轮订单与物流查询已完成。")
    sessions = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    model = AgentModel()
    cfg = settings(model_context_window=1_000_000)
    user = str(uuid4())
    cid = None
    checkpoint_path = str(tmp_path / "acceptance.sqlite3")
    caplog.set_level("INFO", logger="app.workflow.memory")
    try:
        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
            service = WorkflowChatService(catalog=TestToolCatalog(), settings=cfg, model=model, session_factory=sessions, checkpointer=saver, order_data=FakeOrderData())
            for index in range(24):
                turn = await service.prepare_turn(ChatRequest(user_id=user, conversation_id=cid, message=f"第{index}轮，先查订单1001，再查物流，分别告诉我。"))
                cid = turn.conversation_id
                events = [event async for event in service.stream_turn(turn)]
                assert events[-1].name == "done"
                assert len(events[-1].data["tool_trace"]) == 2
            state = (await saver.aget_tuple({"configurable":{"thread_id":f"{user}:{cid}"}})).checkpoint["channel_values"]
            assert len(state["messages"]) == 144
            assert sum(isinstance(m, ToolMessage) for m in state["messages"]) == 48
            assert all(service._memory.count(ctx) <= service._memory.budget.request for ctx in model.histories)
            assert all(sum(isinstance(m, SystemMessage) for m in ctx) == 1 for ctx in model.histories)
            assert len({ctx[0].content for ctx in model.histories}) == 1
            for ctx in model.histories:
                assert {call["id"] for m in ctx if isinstance(m, AIMessage) for call in m.tool_calls} == {m.tool_call_id for m in ctx if isinstance(m, ToolMessage)}
            await service.close()
        # Restart the saver and workflow, retaining DB raw history and the tool observations.
        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
            service = WorkflowChatService(catalog=TestToolCatalog(), settings=cfg, model=model, session_factory=sessions, checkpointer=saver, order_data=FakeOrderData())
            turn = await service.prepare_turn(ChatRequest(user_id=user, conversation_id=cid, message="再查订单1001和物流"))
            assert [event async for event in service.stream_turn(turn)][-1].name == "done"
            state = (await saver.aget_tuple({"configurable":{"thread_id":f"{user}:{cid}"}})).checkpoint["channel_values"]
            assert len(state["messages"]) == 150 and len({m.id for m in state["messages"]}) == 150
            await service.close()
        async with sessions() as session:
            conversation = await ConversationRepository(session).get_owned(cid, user)
            assert not conversation.layer1_from_msg_id and not conversation.summary_upto_msg_id
            rows = await MessageRepository(session).list_ordered(cid)
            assert len(rows) == 50 and all(row.role != "tool" for row in rows)
            assert len(to_public_messages(rows, {"query_order":"查询订单", "query_logistics":"查询物流"})[-1].tool_trace) == 2
        assert not model.summary_batches and caplog.text.count("model_ctx") == 75
    finally:
        if cid is not None:
            async with sessions() as session, session.begin():
                # Exact test conversation AND owner, not a broad user-data cleanup.
                await ConversationRepository(session).get_owned(cid, user)
                from app.database.models import ToolAuditLog
                await session.execute(delete(ToolAuditLog).where(ToolAuditLog.conversation_id == cid))
                await session.execute(delete(ConversationSummary).where(ConversationSummary.conversation_id == cid))
                await session.execute(delete(Message).where(Message.conversation_id == cid))
                await session.execute(delete(Conversation).where(Conversation.id == cid, Conversation.user_id == user))
