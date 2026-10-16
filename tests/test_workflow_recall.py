"""Current-conversation recall must not be discarded by business routing."""
from langchain_core.messages import AIMessage, HumanMessage
import pytest
from uuid import uuid4

from tests.workflow_fakes import InMemoryWorkflow, LabelledModel, LabelledSearch, tool_call, turn
from app.workflow.understanding import explicit_history_recall
from tests.fakes import fake_settings as settings


@pytest.mark.parametrize("question", [
    "我之前查询的是哪个订单", "我刚才查的是哪笔订单？", "我们之前聊了什么", "总结一下我们之前的对话",
])
async def test_previous_order_question_uses_history_without_a_fresh_lookup(question):
    # Replay the captured classifier verdict; the visible history contains the answer.
    model = LabelledModel("其他", ["您之前查询的是订单1001（猫爬架）。这是本会话的历史记录，未重新查询。"])
    search = LabelledSearch(True)
    service = InMemoryWorkflow(model, search)
    events = [event async for event in service.stream_turn(turn(
        question,
        [HumanMessage(content="先查订单1001当前状态，再查这个订单的物流"),
         AIMessage(content="订单1001，商品猫爬架，已签收。")],
    ))]
    assert events[-1].name == "done"
    assert "1001" in events[-1].data["content"]
    assert model.agent_calls == 1
    assert search.calls == []
    assert events[-1].data["tool_trace"] == []
    assert model.tool_names == []
    assert any("猫爬架" in str(m.content) for m in model.agent_histories[0])


@pytest.mark.parametrize("question", [
    "刚才那个订单的物流是什么", "我之前的订单现在能退吗", "我之前查询的是哪个订单，帮我重新查物流",
    "我刚才查的订单到哪了", "你好", "那个怎么弄", "我之前查的是哪个订单，帮我建工单",
])
def test_fresh_or_mixed_queries_are_not_captured_by_recall_rule(question):
    assert not explicit_history_recall(question)


async def test_missing_history_still_answers_recall_without_guessing_or_tools():
    model = LabelledModel("闲聊", ["当前会话没有此前的订单查询记录，无法确认您之前查询的订单。"])
    service = InMemoryWorkflow(model, LabelledSearch(True))
    events = [e async for e in service.stream_turn(turn("我之前查询的是哪个订单"))]
    assert "无法确认" in events[-1].data["content"]
    assert model.agent_calls == 1 and model.tool_names == []
    assert "不猜订单号" in str(model.agent_histories[0][-1].content)


async def test_model_classified_recall_is_supported_and_tools_never_execute():
    model = LabelledModel("会话回顾", [tool_call("query_order", '{"order_id":"1001"}', "unexpected")])
    service = InMemoryWorkflow(model, LabelledSearch(True))
    events = [e async for e in service.stream_turn(turn("请提醒我前面那次查询的结果"))]
    assert events[-1].name == "done"
    assert "不执行新的业务查询" in events[-1].data["content"]
    assert events[-1].data["tool_trace"] == []
    assert model.agent_calls == 1 and model.tool_names == []


async def test_fresh_logistics_question_keeps_tool_agent_path():
    model = LabelledModel("物流", [tool_call("query_order", '{"order_id":"1001"}', "fresh"), "已重新查询订单1001。"])
    service = InMemoryWorkflow(model, LabelledSearch(True))
    events = [e async for e in service.stream_turn(turn("刚才那个订单的物流是什么", [HumanMessage(content="查订单1001")]))]
    assert events[-1].name == "done"
    assert [item["name"] for item in events[-1].data["tool_trace"]] == ["query_order"]
    assert "query_order" in model.tool_names


async def test_recall_uses_owned_summary_and_does_not_cross_conversations(context_sessions, caplog):
    from app.repositories.conversations import ConversationRepository
    from app.repositories.messages import MessageRepository
    from app.schemas import ChatRequest
    from app.workflow.graph import WorkflowChatService
    from tests.order_fakes import FakeOrderData, TestToolCatalog

    user = str(uuid4())
    async with context_sessions() as session, session.begin():
        conversations = ConversationRepository(session)
        first = await conversations.create(user)
        old = await MessageRepository(session).add_user(first.id, "此前业务消息")
        first.summary = "用户曾查询订单1001猫爬架的物流，查询结果为已签收。"
        first.summary_upto_msg_id = old.id
        first.layer1_from_msg_id = old.id
        other = await conversations.create(user)
    model = LabelledModel("其他", ["您此前查询订单1001猫爬架。", "本会话没有此前订单记录，无法确认。"])
    service = WorkflowChatService(settings=settings(model_context_window=100000), model=model,
        session_factory=context_sessions, order_data=FakeOrderData(), catalog=TestToolCatalog(),
        knowledge_search=LabelledSearch(True))
    caplog.set_level("INFO", logger="app.workflow.memory")
    try:
        for cid in [first.id, other.id]:
            prepared = await service.prepare_turn(ChatRequest(user_id=user, conversation_id=cid, message="我之前查询的是哪个订单"))
            events = [e async for e in service.stream_turn(prepared)]
            assert events[-1].name == "done" and events[-1].data["tool_trace"] == []
        assert any("用户曾查询订单1001猫爬架" in str(m.content) for m in model.agent_histories[0])
        assert all("1001" not in str(m.content) for m in model.agent_histories[1])
        assert all("tool_schemas=[]" in r.message for r in caplog.records if r.message.startswith("model_ctx"))
    finally:
        await service.close()
