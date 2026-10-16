"""Explicit current-turn order binding, driven through real graph events."""
import json

import pytest
from langchain_core.messages import HumanMessage

from app.workflow.understanding import explicit_order_id, explicit_order_ids, explicit_order_ambiguity
from tests.workflow_fakes import run_script


@pytest.mark.parametrize("question", [
    "我需要处理1001订单", "我需要处理订单1001", "订单号：1001需要处理",
    "我需要处理1001号订单", "单号 # 1001", "1001 订单需要处理",
])
def test_explicit_order_written_before_or_after_number(question):
    assert explicit_order_id(question)=="1001"


@pytest.mark.parametrize("question", ["售价1001元", "型号1001", "1001", "手机号13800138000",
    "订单123456789012345678901", "123456789012345678901订单"])
def test_numbers_without_a_valid_order_label_are_not_order_ids(question):
    assert explicit_order_id(question) is None


@pytest.mark.parametrize("detail", ["1002元", "13800138000是手机号", "1002是型号",
    "13800138000联系我", "20261013收到的", "13800138000", "20261013"])
def test_other_numeric_slots_do_not_create_a_second_order(detail):
    assert explicit_order_ids("订单1001，"+detail)==["1001"]


@pytest.mark.parametrize("detail", ["13800138000联系我", "20261013收到的"])
async def test_phone_or_date_does_not_repeat_order_selection(detail):
    question="我需要处理订单1001，"+detail
    service,_,_,events=await run_script(question,
        [{"intent":"订单","confidence":.95,"task_kind":"order_specific"}],
        resolution={"question":question,"reference_status":"ambiguous"},
        outputs=["请补充您希望对这笔订单办理的事项。"])
    assert events[-1].name=="done"
    assert not any(e.name=="order_selection" for e in events)
    snapshot=await service._checkpointer.aget_tuple({"configurable":{"thread_id":"demo:1"}})
    assert snapshot.checkpoint["channel_values"]["order_id"]=="1001"


@pytest.mark.parametrize("question", ["我需要处理1001订单", "我需要处理订单1001"])
async def test_unique_explicit_order_does_not_repeat_selection_when_model_says_ambiguous(question):
    service,model,_,events=await run_script(question,
        [{"intent":"订单","confidence":.95,"task_kind":"order_specific"}],
        resolution={"question":question,"reference_status":"ambiguous","clarification_question":"请选择订单"},
        outputs=["请补充您希望对这笔订单办理的事项。"],
        history=[HumanMessage(content="之前查询了订单1002")])
    assert events[-1].name=="done"
    assert not any(e.name=="order_selection" for e in events)
    snapshot=await service._checkpointer.aget_tuple({"configurable":{"thread_id":"demo:1"}})
    state=snapshot.checkpoint["channel_values"]
    assert state["order_id"]=="1001"
    assert json.loads(state["order_observation"])["order_id"]=="1001"
    assert [item["name"] for item in events[-1].data["tool_trace"]]==["query_order"]
    assert model.agent_calls==1


@pytest.mark.parametrize("rewritten", ["我需要处理订单1002", "我需要处理这笔订单"])
async def test_resolver_cannot_replace_or_drop_explicit_current_order(rewritten):
    service,_,_,events=await run_script("我需要处理订单1001",
        [{"intent":"售后","confidence":.95,"task_kind":"order_specific"}],
        resolution={"question":rewritten,"reference_status":"resolved"},
        confident=False,history=[HumanMessage(content="此前处理的是订单1002")])
    assert events[-1].name=="done" and not any(e.name=="order_selection" for e in events)
    snapshot=await service._checkpointer.aget_tuple({"configurable":{"thread_id":"demo:1"}})
    assert snapshot.checkpoint["channel_values"]["order_id"]=="1001"


async def test_after_sales_explicit_inverse_order_retrieves_without_reselection():
    _,model,search,events=await run_script("我需要处理1001订单",
        [{"intent":"售后","confidence":.95,"task_kind":"order_specific"}],
        resolution={"question":"我需要处理1001订单","reference_status":"ambiguous"},confident=False)
    assert events[-1].name=="done" and search.calls
    assert not any(e.name=="order_selection" for e in events)
    assert events[-1].data["tool_trace"][0]["name"]=="query_order"
    assert model.agent_calls==0  # Weak evidence must not start a write-capable agent.


@pytest.mark.parametrize("orders", ["1001订单和1002订单", "订单1001和1002", "1001和1002订单"])
async def test_two_inverse_orders_with_singular_reference_still_need_selection(orders):
    question=orders+"，那个订单物流到哪了"
    assert explicit_order_ambiguity(question)
    _,model,search,events=await run_script(question,
        [{"intent":"物流","confidence":.95,"task_kind":"progress_query"}],
        resolution={"question":question,"reference_status":"clear"})
    assert events[-1].name=="order_selection" and not search.calls and model.agent_calls==0
    assert not any(e.name=="tool_status" for e in events)


async def test_explicit_unknown_order_cannot_bypass_owned_choices():
    _,model,_,events=await run_script("我需要处理9999订单",
        [{"intent":"订单","confidence":.95,"task_kind":"order_specific"}],
        resolution={"question":"我需要处理9999订单","reference_status":"ambiguous"})
    assert events[-1].name=="order_selection" and model.agent_calls==0
    assert "9999" not in {str(order["order_id"]) for order in events[-1].data["orders"]}


def test_repeated_same_order_and_explicit_batch_are_not_ambiguous():
    assert explicit_order_ids("订单1001，订单1002")==["1001", "1002"]
    assert explicit_order_ids("订单1001和1002和1003的物流")==["1001", "1002", "1003"]
    assert not explicit_order_ambiguity("1001订单和订单1001，那个订单物流到哪了")
    assert not explicit_order_ambiguity("分别查询1001订单和1002订单的物流")
