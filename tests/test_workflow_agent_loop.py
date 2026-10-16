"""Workflow routing and bounded agent tool loop, with scripted models only."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from tests.workflow_fakes import LabelledModel, run_example, tool_call


@pytest.mark.parametrize("intent,route", [
    ("物流", "business"), ("订单", "business"),
    ("商品咨询", "knowledge"), ("退款退货", "order_knowledge"), ("售后", "order_knowledge"),
    ("投诉", "complaint"), ("闲聊", "smalltalk"),
])
async def test_seven_labelled_intents_use_the_required_four_routes(intent, route):
    outputs = ["有据可查。[1]"] if route in {"knowledge", "order_knowledge"} else ["您好"] if route == "business" else []
    question = f"{intent}订单1001" if route == "order_knowledge" else intent
    service, model, search, events = await run_example(intent, question, outputs)
    assert events[-1].name == "done"
    assert model.classifier_calls == 1
    assert len(search.calls) == (1 if route in {"knowledge", "order_knowledge"} else 0)
    assert model.agent_calls == (1 if route in {"business", "knowledge", "order_knowledge"} else 0)
    if model.agent_calls: assert "create_ticket" in model.tool_names
    if intent in {"退款退货", "售后"}:
        assert search.calls[0]["content_types"] == ("policy",)


async def test_weak_knowledge_is_logged_and_blocked_before_agent():
    service, model, search, events = await run_example("商品咨询", "商品有没有保修", confident=False)
    assert model.agent_calls == 0
    assert service.low_confidence == ["retrieval_low_conf"]
    assert "不会猜测" in events[-1].data["content"]


async def test_order_then_logistics_requires_two_agent_tool_steps():
    outputs = [
        tool_call("query_order", '{"order_id":"1001"}', "order-1"),
        tool_call("query_logistics", '{"order_id":"1001"}', "logistics-1"),
        "订单已查询，包裹正在运输中。",
    ]
    service, model, search, events = await run_example("物流", "先查订单1001再查物流", outputs)
    assert model.agent_calls == 3
    assert [event.data["status"] for event in events if event.name == "tool_status"] == [
        "running", "succeeded", "running", "succeeded"
    ]
    assert len(events[-1].data["tool_trace"]) == 2


async def test_two_tool_calls_in_one_model_response_are_executed_and_shown():
    simultaneous_calls = AIMessageChunk(content="", tool_call_chunks=[
        {"name": "query_order", "args": '{"order_id":"1001"}', "id": "order-1", "index": 0, "type": "tool_call_chunk"},
        {"name": "query_logistics", "args": '{"order_id":"1001"}', "id": "logistics-1", "index": 1, "type": "tool_call_chunk"},
    ])
    _, model, _, events = await run_example(
        "订单", "先查订单 1001 当前状态，再查这个订单的物流", [simultaneous_calls, "两项均已查询。"]
    )

    assert events[-1].name == "done"
    assert model.agent_calls == 2
    assert [item["name"] for item in events[-1].data["tool_trace"]] == [
        "query_order", "query_logistics",
    ]
    assert [event.data["status"] for event in events if event.name == "tool_status"] == [
        "running", "succeeded", "running", "succeeded",
    ]
    grouped, order_result, logistics_result = model.agent_histories[-1][-3:]
    assert isinstance(grouped, AIMessage)
    assert [call["id"] for call in grouped.tool_calls] == ["order-1", "logistics-1"]
    assert isinstance(order_result, ToolMessage) and order_result.tool_call_id == "order-1"
    assert isinstance(logistics_result, ToolMessage) and logistics_result.tool_call_id == "logistics-1"


async def test_over_limit_tool_batch_is_rejected_without_partial_execution():
    oversized_batch = AIMessageChunk(content="", tool_call_chunks=[
        {"name": "query_order", "args": '{"order_id":"1001"}', "id": f"order-{index}", "index": index, "type": "tool_call_chunk"}
        for index in range(1, 4)
    ])
    outputs = [tool_call("query_order", '{"order_id":"1001"}', "order-0"), oversized_batch]
    _, _, _, events = await run_example("订单", "核查订单 1001", outputs)

    assert events[-1].name == "done"
    assert [item["call_id"] for item in events[-1].data["tool_trace"] if item["status"] == "succeeded"] == ["order-0"]
    assert [item["status"] for item in events[-1].data["tool_trace"]] == ["succeeded", "failed", "failed", "failed"]
    assert "查询尚未完成" in events[-1].data["content"]


async def test_explicit_fresh_order_then_logistics_does_not_accept_stale_answer():
    outputs = [
        "根据此前查询，订单和物流都已有结果。",
        tool_call("query_order", '{"order_id":"1001"}', "order-fresh"),
        tool_call("query_logistics", '{"order_id":"1001"}', "logistics-fresh"),
        "本轮重新查询了订单和物流。",
    ]
    _, model, _, events = await run_example(
        "订单", "先查订单 1001 当前状态，再查这个订单的物流，最后分别告诉我两项结果。", outputs,
        decision={"task_kind": "progress_query", "requested_lookups": ["query_order", "query_logistics"]},
    )

    assert [item["name"] for item in events[-1].data["tool_trace"]] == [
        "query_order", "query_logistics"
    ]
    assert events[-1].data["content"] == "本轮重新查询了订单和物流。"
    assert model.agent_calls == 4


async def test_explicit_dual_lookup_fails_safely_if_model_never_calls_tools():
    outputs = ["根据旧结果，两项状态都已查询。"] * 3
    _, model, _, events = await run_example(
        "物流", "先查订单 1001 当前状态，再查这个订单的物流", outputs,
        decision={"task_kind": "progress_query", "requested_lookups": ["query_order", "query_logistics"]},
    )

    assert model.agent_calls == 3
    assert events[-1].data["tool_trace"] == []
    assert "本轮实时查询尚未完成" in events[-1].data["content"]
    assert "旧结果" not in events[-1].data["content"]


async def test_repeated_calls_to_the_same_tool_keep_distinct_executions():
    outputs = [
        tool_call("query_order", '{"order_id":"1001"}', "order-first"),
        tool_call("query_order", '{"order_id":"1001"}', "order-second"),
        "订单已重新查询。",
    ]
    _, _, _, events = await run_example("订单", "再查一次订单 1001", outputs)

    assert [item["call_id"] for item in events[-1].data["tool_trace"]] == [
        "order-first", "order-second"
    ]
    assert [item["name"] for item in events[-1].data["tool_trace"]] == [
        "query_order", "query_order"
    ]


async def test_complaint_only_suggests_independent_actions_without_creating_ticket():
    service, model, search, events = await run_example("投诉", "我要投诉")
    assert model.agent_calls == 0
    assert events[-1].data["suggested_actions"] == ["转人工", "建工单"]
    assert not any(item[0] == "tool_call" for item in service.saved)


async def test_agent_may_suggest_one_action_without_creating_ticket():
    service, model, search, events = await run_example(
        "人工", "请人工帮我核查订单", ["我可以帮您转人工。\n[建议选项:转人工]"]
    )
    assert events[-1].data["suggested_actions"] == ["转人工"]
    assert "建议选项" not in events[-1].data["content"]
    assert not any(item[0] == "tool_call" for item in service.saved)


async def test_bare_loop_stops_at_three_tool_calls_without_a_fourth_execution():
    outputs = [tool_call("query_order", '{"order_id":"1001"}', f"order-{i}") for i in range(4)]
    service, model, search, events = await run_example("订单", "继续核查订单1001", outputs)
    assert model.agent_calls == 4
    assert sum(item["status"] == "succeeded" for item in events[-1].data["tool_trace"]) == 3
    assert events[-1].data["tool_trace"][-1]["status"] == "failed"
    assert "查询尚未完成" in events[-1].data["content"]
