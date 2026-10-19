"""Workflow behaviour with scripted decisions; no claim of real-model accuracy."""
import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from app.workflow.understanding import (
    IntentDecision, ResolvedQuestion, classify_intent, plan_route, understanding_history,
)
from tests.workflow_fakes import (
    InMemoryWorkflow, LabelledModel, LabelledSearch, ScriptedUnderstandingModel, run_script, tool_call, turn,
)


async def test_unknown_product_policy_reclassifies_then_retrieves_and_records_gap(caplog):
    caplog.set_level("INFO", logger="app.workflow")
    service, model, search, events = await run_script(
        "猫抓板用旧了可以回收吗",
        [{"intent": "其他", "confidence": .9},
         {"intent": "商品咨询", "confidence": .95, "task_kind": "general_information"}],
        history=[HumanMessage(content="查订单1001物流"), AIMessage(content="请提供更多物流信息。")],
        confident=False,
    )
    assert events[-1].name == "done"
    assert search.calls and model.agent_calls == 0 and model.classifier_calls == 2
    assert service.low_confidence == ["retrieval_low_conf"]
    assert "intent=商品咨询" in caplog.text and "decision=execute" in caplog.text
    for _, messages in model.inputs:
        assert messages[-1].content == "猫抓板用旧了可以回收吗"
        assert sum(bool(m.additional_kwargs.get("current_question")) for m in messages) == 1


async def test_general_return_policy_does_not_select_or_query_an_order():
    _, model, search, events = await run_script(
        "退货政策是什么", [{"intent": "退款退货", "confidence": .95, "task_kind": "general_information"}],
        outputs=["退货条件以政策为准。[1]"],
    )
    assert events[-1].name == "done" and search.calls
    assert events[-1].data["tool_trace"] == []
    assert "申请退款" not in events[-1].data["suggested_actions"]
    assert model.agent_calls == 1 and search.calls[0]["content_types"] == ("policy",)


async def test_missing_logistics_order_card_resumes_without_reclassification():
    service, model, search, events = await run_script(
        "查询订单物流", [{"intent": "物流", "confidence": .95, "task_kind": "progress_query"}],
        outputs=[tool_call("query_logistics", '{"tracking_no":"DEMO1001"}', "logistics-1"), "订单1001已签收。"],
    )
    assert events[-1].name == "order_selection" and model.agent_calls == 0
    resumed = [e async for e in service.resume_order("demo", 1, "1001", 1)]
    assert resumed[-1].name == "done" and model.classifier_calls == 1
    assert [t["name"] for t in resumed[-1].data["tool_trace"]] == ["query_order", "query_logistics"]
    assert search.calls == []
    assert "订单 1001" in str(model.agent_histories[0][-1].content)


class PromptFollowingLogisticsModel(ScriptedUnderstandingModel):
    """Calls query_order first unless its result is already in context, as the system prompt instructs."""

    async def astream(self, messages):
        self.agent_calls += 1
        done = {m.name for m in messages if isinstance(m, ToolMessage)}
        if "query_order" not in done:
            yield tool_call("query_order", '{"order_id":"1001"}', f"order-{self.agent_calls}")
        elif "query_logistics" not in done:
            yield tool_call("query_logistics", '{"tracking_no":"DEMO1001"}', "logistics-1")
        else:
            yield AIMessageChunk(content="订单1001已签收。")


async def test_order_picked_from_card_is_queried_once_before_logistics():
    model = PromptFollowingLogisticsModel([{"intent": "物流", "confidence": .95, "task_kind": "progress_query"}])
    service = InMemoryWorkflow(model, LabelledSearch())
    events = [e async for e in service.stream_turn(turn("前几天买的东西到哪了"))]
    assert events[-1].name == "order_selection"
    resumed = [e async for e in service.resume_order("demo", 1, "1001", 1)]
    assert resumed[-1].name == "done"
    assert [t["name"] for t in resumed[-1].data["tool_trace"]] == ["query_order", "query_logistics"]
    assert model.agent_calls == 2


async def test_low_confidence_keeps_business_label_but_clarifies_after_one_review():
    decisions = [{"intent": "物流", "confidence": .4, "clarification_question": "您要查询物流位置还是退货进度？"}] * 2
    service, model, search, events = await run_script("那件到哪了", decisions)
    assert events[-1].name == "done" and "退货进度" in events[-1].data["content"]
    assert model.classifier_calls == 2 and model.agent_calls == 0 and search.calls == []
    snapshot = await service._checkpointer.aget_tuple({"configurable": {"thread_id": "demo:1"}})
    assert snapshot.checkpoint["channel_values"]["intent"] == "物流"
    assert snapshot.checkpoint["channel_values"]["routing_decision"] == "clarify"


async def test_invalid_first_output_gets_one_bounded_review():
    model = ScriptedUnderstandingModel([{"intent": "物流", "confidence": "high"}, {"intent": "物流", "confidence": .95}])
    decision = await classify_intent(model, "json_mode", "查订单1001物流")
    assert decision.intent == "物流" and model.classifier_calls == 2


async def test_ambiguous_product_asks_specific_question_without_retrieval():
    _, model, search, events = await run_script(
        "它可以洗吗", [{"intent": "商品咨询", "confidence": .95, "task_kind": "general_information"}],
        resolution={"question": "它可以洗吗", "reference_status": "ambiguous", "clarification_question": "您指猫窝还是猫抓板？"},
        history=[HumanMessage(content="猫窝和猫抓板分别是什么材质？")],
    )
    assert "猫窝还是猫抓板" in events[-1].data["content"]
    assert not search.calls and model.agent_calls == 0


@pytest.mark.parametrize("intent,question", [("售后", "它的保修政策是什么"), ("退款退货", "它的退货政策是什么")])
async def test_ambiguous_general_policy_clarifies_before_retrieval(intent, question):
    _, model, search, events = await run_script(
        question, [{"intent": intent, "confidence": .95, "task_kind": "general_information"}],
        resolution={"question": question, "reference_status": "ambiguous", "clarification_question": "您指饮水机还是猫窝？"},
        history=[HumanMessage(content="饮水机和猫窝分别有什么特点？")],
    )
    assert events[-1].name == "done" and "饮水机还是猫窝" in events[-1].data["content"]
    assert not search.calls and model.agent_calls == 0


@pytest.mark.parametrize("intent,task_kind,question", [
    ("物流", "progress_query", "订单1001和订单1002，那个订单物流到哪了"),
    ("退款退货", "order_specific", "订单1001和订单1002，那个订单能退吗"),
])
async def test_two_order_numbers_require_selection_and_resume_with_chosen_order(intent, task_kind, question):
    service, model, search, events = await run_script(
        question, [{"intent": intent, "confidence": .95, "task_kind": task_kind}],
        resolution={"question": question, "reference_status": "ambiguous", "clarification_question": "您指订单1001还是1002？"},
        outputs=["已核查所选订单1002。[1]" if intent == "退款退货" else "已核查所选订单1002。"],
    )
    assert events[-1].name == "order_selection" and model.agent_calls == 0
    assert not any(e.name == "tool_status" for e in events)
    resumed = [e async for e in service.resume_order("demo", 1, "1002", 1)]
    assert resumed[-1].name == "done" and model.classifier_calls == 1
    snapshot = await service._checkpointer.aget_tuple({"configurable": {"thread_id": "demo:1"}})
    state = snapshot.checkpoint["channel_values"]
    assert state["order_id"] == "1002"
    assert json.loads(state["order_observation"])["order_id"] == "1002"
    assert state["question"].startswith("针对用户本次明确选择的订单 1002")
    assert bool(search.calls) == (intent == "退款退货")


async def test_refund_progress_uses_business_not_policy_retrieval():
    service, model, search, events = await run_script(
        "我的退款处理进度到哪了", [{"intent": "退款退货", "confidence": .95, "task_kind": "progress_query"}],
        outputs=["请以实际进度查询结果为准。"],
    )
    assert events[-1].name == "order_selection"
    resumed = [e async for e in service.resume_order("demo", 1, "1001", 1)]
    assert resumed[-1].name == "done" and model.classifier_calls == 1
    assert not search.calls and model.agent_calls == 1
    snapshot = await service._checkpointer.aget_tuple({"configurable": {"thread_id": "demo:1"}})
    assert snapshot.checkpoint["channel_values"]["route"] == "business"


async def test_assistant_only_order_number_cannot_ground_resolver():
    _, model, _, events = await run_script(
        "它能退吗", [{"intent": "退款退货", "confidence": .95, "task_kind": "order_specific"}],
        resolution={"question": "订单1001能退吗", "reference_status": "resolved"},
        history=[AIMessage(content="我猜您指订单1001。")],
    )
    assert events[-1].name == "order_selection" and model.agent_calls == 0


def test_context_has_one_current_turn_and_only_tool_observations_prove_query():
    raw = "猫抓板用旧了可以回收吗"
    messages = [HumanMessage(content="查订单1001"), AIMessage(content="还查询过订单9999"),
                ToolMessage(content='{"order_id":"1001","product":"猫爬架"}', name="query_order", tool_call_id="order-1", id="obs-1"),
                HumanMessage(content=raw)]
    context = understanding_history(messages, raw, summary="用户曾询问物流。")
    material = json.loads(str(context[0].content).split("\n", 1)[1])
    assert [o["order_id"] for o in material["queried_orders"]] == ["1001"]
    assert material["queried_orders"][0]["message_id"] == "obs-1"
    assert context[-1].content == raw
    assert sum(m.content == raw for m in context) == 1


@pytest.mark.parametrize("question,intent,task_kind,expected", [
    ("猫抓板能回收吗", "商品咨询", "general_information", "knowledge"),
    ("退货政策是什么", "退款退货", "general_information", "knowledge"),
    ("查订单1001物流", "物流", "progress_query", "business"),
    ("饮水机保修政策是什么", "售后", "general_information", "knowledge"),
    ("查询订单1001退货进度", "售后", "progress_query", "business"),
    ("查询订单1001退款进度", "退款退货", "progress_query", "business"),
])
async def test_next_turn_clears_ticket_collection_when_topic_changes(question, intent, task_kind, expected):
    model = ScriptedUnderstandingModel([
        {"intent": "人工", "confidence": .95},
        {"intent": intent, "confidence": .95, "task_kind": task_kind},
    ], ["请描述工单问题。", "已处理。[1]"])
    service = InMemoryWorkflow(model, LabelledSearch(True))
    first = [e async for e in service.stream_turn(turn("帮我建个工单"))]
    assert first[-1].name == "done"
    second = [e async for e in service.stream_turn(turn(question))]
    assert second[-1].name == "done"
    snapshot = await service._checkpointer.aget_tuple({"configurable": {"thread_id": "demo:1"}})
    state = snapshot.checkpoint["channel_values"]
    assert state["route"] == expected and state["ticket_requested"] is False
    assert all(t["name"] != "create_ticket" for t in second[-1].data["tool_trace"])