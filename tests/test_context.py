import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.prompts import render_customer_service_messages
from app.context import ContextBudgetExceeded, prepare_context


def test_rendered_prompt_owns_system_message() -> None:
    rendered = render_customer_service_messages(
        shop_name="示例店铺",
        messages=[HumanMessage(content="你好")],
    )

    assert isinstance(rendered[0], SystemMessage)
    assert "示例店铺" in rendered[0].content
    assert isinstance(rendered[-1], HumanMessage)


def one_token_per_message(messages: list[BaseMessage]) -> int:
    return len(messages)


def test_prepare_context_keeps_latest_complete_turns() -> None:
    history = [
        HumanMessage(content="旧问题"),
        AIMessage(content="旧回答"),
        HumanMessage(content="新问题"),
        AIMessage(content="新回答"),
        HumanMessage(content="追问"),
    ]

    prepared = prepare_context(
        history,
        shop_name="本店",
        token_budget=4,
        token_counter=one_token_per_message,
    )

    assert [type(message) for message in prepared.messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert [message.content for message in prepared.messages[1:]] == [
        "新问题",
        "新回答",
        "追问",
    ]
    assert prepared.trimmed_messages == 2
    assert prepared.input_tokens_estimate == 4


def test_prepare_context_rejects_oversized_current_turn() -> None:
    with pytest.raises(ContextBudgetExceeded):
        prepare_context(
            [HumanMessage(content="很长的问题")],
            shop_name="本店",
            token_budget=1,
            token_counter=one_token_per_message,
        )


def test_prepare_context_never_orphans_a_tool_result_when_trimming() -> None:
    """The atomic tool pair is dropped together when only the current turn fits."""

    history = [
        HumanMessage(content="查订单"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "query_logistics", "args": {}}],
        ),
        ToolMessage(content='{"location": "Shanghai"}', tool_call_id="call-1"),
        AIMessage(content="订单在运输中"),
        HumanMessage(content="什么时候到？"),
    ]

    prepared = prepare_context(
        history,
        shop_name="本店",
        token_budget=2,
        token_counter=one_token_per_message,
    )

    assert [type(message) for message in prepared.messages] == [
        SystemMessage,
        HumanMessage,
    ]
    assert prepared.messages[-1].content == "什么时候到？"
    assert prepared.trimmed_messages == 4
