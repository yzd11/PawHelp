"""Conversation conversion and bounded prompt-context preparation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.prompts import render_customer_service_messages

TokenCounter = Callable[[list[BaseMessage]], int]


def bound_text(value: str, tokens: int, token_counter: TokenCounter) -> str:
    """Bound evidence using the same message counter as the conversation budget."""
    low, high = 0, len(value)
    while low < high:
        mid = (low + high + 1) // 2
        if token_counter([HumanMessage(content=value[:mid])]) <= tokens:
            low = mid
        else:
            high = mid - 1
    return value[:low]


class ContextBudgetExceeded(ValueError):
    """Raised when the server prompt and current user message do not fit."""


@dataclass(frozen=True)
class PreparedContext:
    """The immutable, budgeted messages supplied to the chat model."""

    messages: tuple[BaseMessage, ...]
    input_tokens_estimate: int
    trimmed_messages: int


def prepare_context(
    history: Sequence[BaseMessage],
    shop_name: str,
    token_budget: int,
    token_counter: TokenCounter = count_tokens_approximately,
) -> PreparedContext:
    """Render the newest valid server-owned history that fits the budget."""

    api_messages = list(history)
    if not api_messages or not isinstance(api_messages[-1], HumanMessage):
        raise ContextBudgetExceeded("current user message is required")
    rendered_current = render_customer_service_messages(shop_name, [api_messages[-1]])
    minimum_messages = [rendered_current[0], rendered_current[-1]]
    if token_counter(minimum_messages) > token_budget:
        raise ContextBudgetExceeded("current user message exceeds the context token budget")

    trimmed_api_messages = [api_messages[-1]]
    for start in range(len(api_messages)):
        if not isinstance(api_messages[start], HumanMessage):
            continue
        candidate = api_messages[start:]
        if not _has_complete_tool_pairs(candidate):
            continue
        rendered = render_customer_service_messages(shop_name, candidate)
        if token_counter(rendered) <= token_budget:
            trimmed_api_messages = candidate
            break
    trimmed = render_customer_service_messages(shop_name, trimmed_api_messages)
    return PreparedContext(
        messages=tuple(trimmed),
        input_tokens_estimate=token_counter(trimmed),
        trimmed_messages=len(api_messages) - len(trimmed_api_messages),
    )


def _has_complete_tool_pairs(messages: Sequence[BaseMessage]) -> bool:
    """Reject a candidate containing one side of a model/tool correlation pair."""

    requested_ids = {
        call.get("id")
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if isinstance(call, dict) and isinstance(call.get("id"), str)
    }
    tool_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage)
    }
    return requested_ids == tool_ids
