"""Server-owned history reconstruction and public hydration contracts."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.database.models import Message
from app.schemas import ChatRequest
from app.services.history import to_langchain_messages, to_public_messages


def make_row(
    row_id: int,
    role: str,
    content: str,
    *,
    tool_calls: list[object] | None = None,
    tool_call_id: str | None = None,
) -> Message:
    """Build detached ORM rows without invoking a database session."""

    return Message(
        id=row_id,
        conversation_id=1,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


def test_tool_rows_reconstruct_matching_tool_message() -> None:
    """Stored tool metadata must rebuild LangChain's correlated message pair."""

    messages = to_langchain_messages(
        [
            make_row(
                1,
                "assistant",
                "",
                tool_calls=[
                    {"id": "call-1", "name": "query_logistics", "args": {"order_id": "1001"}}
                ],
            ),
            make_row(2, "tool", '{"location": "Shanghai"}', tool_call_id="call-1"),
        ]
    )

    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls[0]["id"] == "call-1"
    assert isinstance(messages[1], ToolMessage)
    assert messages[1].tool_call_id == "call-1"


def test_public_history_hides_tool_results_and_arguments_and_groups_safe_badge() -> None:
    """Hydration exposes only user/final assistant text and registry-approved labels."""

    public = to_public_messages(
        [
            make_row(1, "user", "订单到了吗"),
            make_row(
                2,
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "query_logistics",
                        "args": {"order_id": "secret-1001"},
                    }
                ],
            ),
            make_row(3, "tool", '{"internal_location": "warehouse-7"}', tool_call_id="call-1"),
            make_row(4, "assistant", "包裹正在配送中。"),
        ],
        {"query_logistics": "查询物流"},
    )

    assert [(message.role, message.content) for message in public] == [
        ("user", "订单到了吗"),
        ("assistant", "包裹正在配送中。"),
    ]
    assert public[-1].tool_trace[0].model_dump() == {
        "tool_call_id": "call-1",
        "name": "query_logistics",
        "label": "查询物流",
        "status": "succeeded",
        "source": "builtin",
        "mcp_server": None,
    }
    assert "secret-1001" not in str(public)
    assert "warehouse-7" not in str(public)


def test_public_history_keeps_every_tool_from_the_same_turn() -> None:
    public = to_public_messages(
        [
            make_row(1, "user", "先查订单 1001，再查物流"),
            make_row(2, "assistant", "", tool_calls=[
                {"id": "order-call", "name": "query_order", "args": {"order_id": "1001"}},
            ]),
            make_row(3, "tool", '{"status":"shipped"}', tool_call_id="order-call"),
            make_row(4, "assistant", "", tool_calls=[
                {"id": "logistics-call", "name": "query_logistics", "args": {"order_id": "1001"}},
            ]),
            make_row(5, "tool", '{"location":"配送中心"}', tool_call_id="logistics-call"),
            make_row(6, "assistant", "", tool_calls=[
                {"id": "order-retry", "name": "query_order", "args": {"order_id": "1001"}},
            ]),
            make_row(7, "tool", '{"status":"delivered"}', tool_call_id="order-retry"),
            make_row(8, "assistant", "订单和物流均已查询。"),
        ],
        {"query_order": "查询订单", "query_logistics": "查询物流"},
    )

    assert [item.label for item in public[-1].tool_trace] == ["查询订单", "查询物流", "查询订单"]
    assert [item.status for item in public[-1].tool_trace] == ["succeeded"] * 3
    assert "shipped" not in str(public)
    assert "配送中心" not in str(public)
    assert "delivered" not in str(public)


def test_workflow_suggestion_and_citations_survive_history_hydration() -> None:
    answer = make_row(2, "assistant", "可在签收后七天内申请[1]。")
    answer.suggested_actions = ["转人工", "建工单"]
    answer.source_refs = [
        {"number": 1, "chunk_id": 41, "category": "退款退货", "section_path": "退货政策"}
    ]
    public = to_public_messages([make_row(1, "user", "如何退货"), answer], {})
    assert public[-1].suggested_actions == ["转人工", "建工单"]
    assert public[-1].sources[0].chunk_id == 41


def test_public_history_omits_unknown_tool_names() -> None:
    """A persisted but unregistered tool never becomes a browser badge."""

    public = to_public_messages(
        [
            make_row(1, "user", "问题"),
            make_row(
                2,
                "assistant",
                "",
                tool_calls=[{"id": "call-unknown", "name": "admin_export", "args": {}}],
            ),
            make_row(3, "tool", "private export", tool_call_id="call-unknown"),
            make_row(4, "assistant", "我无法处理该请求。"),
        ],
        {"query_logistics": "查询物流"},
    )

    assert public[-1].tool_trace == []


def test_public_history_rebuilds_safe_faq_sources_from_private_tool_result() -> None:
    public = to_public_messages(
        [
            make_row(1, "user", "邮费是多少"),
            make_row(
                2,
                "assistant",
                "",
                tool_calls=[{"id": "call-faq", "name": "query_faq", "args": {}}],
            ),
            make_row(
                3,
                "tool",
                '{"matches":[{"citation":1,"chunk_id":7,"category":"配送服务",'
                '"section_path":"配送 / 费用","answer":"private evidence"}]}',
                tool_call_id="call-faq",
            ),
            make_row(4, "assistant", "请参考配送费用说明 [1]。"),
        ],
        {"query_faq": "查询常见问题"},
    )

    assert [source.model_dump() for source in public[-1].sources] == [
        {
            "number": 1,
            "chunk_id": 7,
            "category": "配送服务",
            "section_path": "配送 / 费用",
        }
    ]
    assert "private evidence" not in str(public)


def test_interrupted_tool_trace_does_not_cross_into_the_next_user_turn() -> None:
    """A later no-tool response must not inherit an unfinished turn's badge."""

    public = to_public_messages(
        [
            make_row(1, "user", "查订单"),
            make_row(
                2,
                "assistant",
                "",
                tool_calls=[{"id": "call-1", "name": "query_logistics", "args": {}}],
            ),
            make_row(3, "tool", "private logistics result", tool_call_id="call-1"),
            make_row(4, "user", "换个问题"),
            make_row(5, "assistant", "当然，请问。"),
        ],
        {"query_logistics": "查询物流"},
    )

    assert [(message.role, message.content) for message in public] == [
        ("user", "查订单"),
        ("user", "换个问题"),
        ("assistant", "当然，请问。"),
    ]
    assert public[-1].tool_trace == []


def test_chat_request_normalizes_one_user_message() -> None:
    """The new request cannot carry a transcript or a whitespace-only turn."""

    request = ChatRequest(
        user_id="00000000-0000-0000-0000-000000000001",
        message="  订单 1001 到哪了  ",
    )

    assert request.message == "订单 1001 到哪了"
    assert request.conversation_id is None
