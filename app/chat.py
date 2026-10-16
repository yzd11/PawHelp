"""Logical, transport-independent chat streaming events."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import NamedTuple, Protocol

from langchain_core.messages import AIMessageChunk, BaseMessage

from app.context import PreparedContext


class StreamingChatModel(Protocol):
    """The small streaming-model boundary used by the chat service."""

    def astream(
        self, messages: Sequence[BaseMessage]
    ) -> AsyncIterator[AIMessageChunk]: ...


class ChatEvent(NamedTuple):
    """One transport-independent event emitted by a chat service."""

    name: str
    data: dict[str, object]


def text_from_chunk(chunk: AIMessageChunk) -> str:
    """Extract displayable text while preserving an upstream chunk boundary."""

    content = chunk.content
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            text_parts.append(block["text"])
    return "".join(text_parts)


async def stream_chat_events(
    model: StreamingChatModel, prepared: PreparedContext
) -> AsyncIterator[ChatEvent]:
    """Yield safe logical events from one model invocation."""

    response_parts: list[str] = []
    try:
        async for chunk in model.astream(list(prepared.messages)):
            content = text_from_chunk(chunk)
            if not content:
                continue
            response_parts.append(content)
            yield ChatEvent("token", {"content": content})
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        raise
    except Exception:
        yield ChatEvent(
            "error",
            {
                "code": "upstream_error",
                "message": "模型服务暂时不可用，请稍后重试",
            },
        )
        return

    yield ChatEvent(
        "done",
        {
            "content": "".join(response_parts),
            "input_tokens_estimate": prepared.input_tokens_estimate,
            "trimmed_messages": prepared.trimmed_messages,
        },
    )
