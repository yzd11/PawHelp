"""Server-sent event encoding."""

import json
from collections.abc import Mapping


SSE_EVENT_NAMES = frozenset(
    {"conversation", "tool_status", "token", "order_selection", "ticket_confirmation", "done", "error"}
)
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def encode_sse(event: str, data: Mapping[str, object]) -> bytes:
    """Encode one UTF-8 JSON server-sent event."""

    if event not in SSE_EVENT_NAMES:
        raise ValueError(f"unsupported SSE event: {event!r}")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")
