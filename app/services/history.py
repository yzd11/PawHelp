"""Convert private ORM history into model and browser-safe representations."""

from collections.abc import Mapping, Sequence
import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.database.models import Message
from app.schemas import PublicMessage, PublicSource, PublicToolTrace


def to_langchain_messages(rows: Sequence[Message]) -> list[BaseMessage]:
    """Rebuild the LangChain message types and tool-call correlation metadata."""

    result: list[BaseMessage] = []
    for row in rows:
        meta = {"id": f"db-{row.id}", "additional_kwargs": {"db_message_id": row.id}}
        if row.role == "user":
            result.append(HumanMessage(content=row.content or "", **meta))
        elif row.role == "tool":
            result.append(
                ToolMessage(content=row.content or "", tool_call_id=row.tool_call_id or "", **meta)
            )
        else:
            tool_calls = row.tool_calls if isinstance(row.tool_calls, list) else []
            result.append(AIMessage(content=row.content or "", tool_calls=tool_calls, **meta))
    return result


def _tool_names_by_id(row: Message) -> dict[str, str]:
    """Extract only call IDs and registry names from one private assistant row."""

    if not isinstance(row.tool_calls, list):
        return {}
    names: dict[str, str] = {}
    for call in row.tool_calls:
        if not isinstance(call, dict):
            continue
        call_id = call.get("id")
        name = call.get("name")
        if isinstance(call_id, str) and isinstance(name, str):
            names[call_id] = name
    return names


def _public_sources(row: Message, tool_name: str | None) -> list[PublicSource]:
    if tool_name != "query_faq" or not row.content:
        return []
    try:
        payload = json.loads(row.content)
    except (TypeError, ValueError):
        return []
    matches = payload.get("matches") if isinstance(payload, dict) else None
    if not isinstance(matches, list):
        return []
    sources: list[PublicSource] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        try:
            sources.append(
                PublicSource(
                    number=match["citation"],
                    chunk_id=match["chunk_id"],
                    category=match["category"],
                    section_path=match.get("section_path"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(sources, key=lambda item: item.number)


def to_public_messages(
    rows: Sequence[Message], safe_labels: Mapping[str, str]
) -> list[PublicMessage]:
    """Hydrate only customer text and registry-approved tool badges.

    Tool-call requests and raw tool results are private. A completed safe trace
    is held until the final customer-facing assistant row, where the browser can
    render it as a badge beside the resulting answer.
    """

    calls_by_id: dict[str, str] = {}
    pending_traces: list[PublicToolTrace] = []
    pending_sources: list[PublicSource] = []
    public: list[PublicMessage] = []
    for row in rows:
        if row.role == "user":
            calls_by_id = {}
            pending_traces = []
            pending_sources = []
            public.append(PublicMessage(id=row.id, role="user", content=row.content or ""))
        elif row.role == "assistant" and row.tool_calls:
            calls_by_id.update(_tool_names_by_id(row))
        elif row.role == "tool":
            name = calls_by_id.get(row.tool_call_id or "")
            label = safe_labels.get(name) if name is not None else None
            if label is not None and row.tool_call_id is not None:
                pending_traces.append(
                    PublicToolTrace(
                        tool_call_id=row.tool_call_id,
                        name=name,
                        label=label,
                        status="succeeded",
                    )
                )
            pending_sources = _public_sources(row, name)
        elif row.role == "assistant":
            public.append(
                PublicMessage(
                    id=row.id,
                    role="assistant",
                    content=row.content or "",
                    tool_trace=(
                        [PublicToolTrace(tool_call_id=str(item["call_id"]), name=str(item["name"]), label=safe_labels.get(str(item["name"]), str(item.get("label") or "工具调用"))[:80],
                            status=item["status"], source=item.get("source", "builtin"), mcp_server=item.get("mcp_server"))
                         for item in (row.tool_trace or []) if item.get("status") in {"succeeded", "failed"} and item.get("name") and item.get("call_id")]
                        if row.tool_trace is not None else pending_traces
                    ),
                    sources=(
                        [PublicSource.model_validate(item) for item in row.source_refs]
                        if isinstance(row.source_refs, list) else pending_sources
                    ),
                    suggested_actions=(
                        [item for item in row.suggested_actions if item in {"转人工", "建工单", "申请退款"}]
                        if isinstance(row.suggested_actions, list) else []
                    ),
                    related_order_id=row.related_order_id,
                    feedback=getattr(row, "feedback", None),
                )
            )
            pending_traces = []
            pending_sources = []
    return public
