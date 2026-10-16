"""Multi-turn order binding, query fan-out and order-card resume."""

from langchain_core.messages import HumanMessage

from tests.workflow_fakes import InMemoryWorkflow, KeywordModel, LabelledSearch, turn


async def test_multiturn_logistics_refund_logistics_and_query_fanout_dedup():
    model, search = KeywordModel(), LabelledSearch()
    service = InMemoryWorkflow(model, search)
    first = [event async for event in service.stream_turn(turn("订单1001的物流到哪了"))]
    assert first[-1].name == "done"
    history = [HumanMessage(content="订单1001的物流到哪了")]
    second = [event async for event in service.stream_turn(turn("它能退吗", history))]
    assert second[-1].name == "done"
    assert any(kind == "IntentDecision" and q == "订单1001的猫爬架能退吗" for kind, q in model.structured)
    assert len(search.queries) == 3  # original + two unique expansions
    assert all(kwargs["content_types"] == ("policy",) for _, kwargs in search.queries)
    assert len(second[-1].data["sources"]) == 1  # same chunk across queries
    assert [item["name"] for item in second[-1].data["tool_trace"]] == ["query_order"]
    third = [event async for event in service.stream_turn(turn("再看订单1001的物流"))]
    assert third[-1].name == "done"
    assert len(search.queries) == 3
    graph, _ = service._build_graph(turn("再看订单1001的物流"))
    snapshot = await graph.aget_state(
        {"configurable": {"thread_id": "demo:1"}}
    )
    assert snapshot.values["order_observation"] == ""
    assert snapshot.values["refund_available"] is False


async def test_missing_order_interrupts_then_card_click_resumes_same_graph():
    model, search = KeywordModel(), LabelledSearch()
    service = InMemoryWorkflow(model, search)
    events = [event async for event in service.stream_turn(turn("这个能退吗"))]
    assert events[-1].name == "order_selection"
    assert "演示订单" in events[-1].data["message"]
    assert model.agent_calls == 0
    assert await service.pending_order("demo", 1) is not None
    resumed = [event async for event in service.resume_order("demo", 1, "1001", 1)]
    assert resumed[-1].name == "done"
    assert model.agent_calls == 1
    assert search.queries
    assert await service.pending_order("demo", 1) is None


async def test_unknown_intent_does_not_enter_agent_or_retrieve():
    model, search = KeywordModel(), LabelledSearch()
    service = InMemoryWorkflow(model, search)
    events = [event async for event in service.stream_turn(turn("那个怎么弄"))]
    assert events[-1].name == "done"
    assert "补充" in events[-1].data["content"]
    assert model.agent_calls == 0
    assert search.queries == []


async def test_resolver_cannot_invent_order_number():
    model, search = KeywordModel(), LabelledSearch()
    service = InMemoryWorkflow(model, search)
    events = [event async for event in service.stream_turn(turn("能退吗"))]
    assert events[-1].name == "order_selection"
    assert model.agent_calls == 0


async def test_invalid_intent_json_falls_back_to_other():
    model, search = KeywordModel(), LabelledSearch()
    service = InMemoryWorkflow(model, search)
    events = [event async for event in service.stream_turn(turn("怪问题"))]
    assert events[-1].name == "done"
    assert "补充" in events[-1].data["content"]
    assert model.agent_calls == 0
