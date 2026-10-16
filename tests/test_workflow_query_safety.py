"""User-visible guardrails when model understanding or queries do not converge."""
import pytest
from langchain_core.messages import AIMessageChunk

from tests.workflow_fakes import run_script
from tests.workflow_fakes import tool_call


async def test_missed_model_ambiguity_still_requires_order_selection():
    question = "订单1001和订单1002，那个订单物流到哪了"
    service, model, search, events = await run_script(
        question, [{"intent": "物流", "confidence": .96, "task_kind": "progress_query"}],
        resolution={"question": "订单1001的物流到哪了", "reference_status": "resolved"},
        outputs=["请补充更多信息。"],
    )
    assert events[-1].name == "order_selection"
    assert model.agent_calls == 0 and not search.calls
    assert not any(e.name == "tool_status" for e in events)
    resumed = [e async for e in service.resume_order("demo", 1, "1002", 1)]
    assert resumed[-1].name == "done" and model.classifier_calls == 1
    snapshot = await service._checkpointer.aget_tuple({"configurable": {"thread_id": "demo:1"}})
    assert snapshot.checkpoint["channel_values"]["order_id"] == "1002"


async def test_partial_queries_hit_limit_without_claiming_missing_order_id():
    batch = AIMessageChunk(content="", tool_call_chunks=[
        {"name": "query_logistics", "args": '{"tracking_no":"DEMO1001"}', "id": "logistics-1", "index": 0, "type": "tool_call_chunk"},
        {"name": "query_logistics", "args": '{"tracking_no":"DEMO1002"}', "id": "logistics-2", "index": 1, "type": "tool_call_chunk"},
    ])
    _, _, _, events = await run_script(
        "查询订单1001与订单1002的物流", [{"intent": "物流", "confidence": .96, "task_kind": "progress_query"}],
        outputs=[tool_call("query_order", '{"order_id":"1001"}', "order-1"),
                 tool_call("query_order", '{"order_id":"1002"}', "order-2"), batch],
    )
    done = events[-1]
    assert done.name == "done"
    assert [t["status"] for t in done.data["tool_trace"]] == ["succeeded", "succeeded", "failed", "failed"]
    assert "请提供订单号" not in done.data["content"]
    assert "查询尚未完成" in done.data["content"]
    assert done.data["suggested_actions"] == ["转人工"]


@pytest.mark.parametrize("question", [
    "分别查询订单1001和订单1002的物流", "查询订单1001和订单1001，那个订单到哪了",
    "订单1001和订单1002，那个订单，我指的是订单1002，到哪了", "订单1001的物流到哪了",
])
def test_explicit_batch_or_named_target_is_not_ambiguous(question):
    from app.workflow.understanding import explicit_order_ambiguity
    assert not explicit_order_ambiguity(question)
