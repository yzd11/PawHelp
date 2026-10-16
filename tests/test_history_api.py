"""Owned browser-safe conversation history endpoint behavior."""

from collections.abc import Iterable

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.database.models import Conversation, Message
from app.main import create_app
from tests.fakes import FakeCombinedModel


DATABASE_URL = "mysql+asyncmy://pawhelp:pawhelp@127.0.0.1:3306/pawhelp"
OWNER_ID = "00000000-0000-0000-0000-000000000001"
OTHER_USER_ID = "00000000-0000-0000-0000-000000000002"
CONVERSATION_ID = 3
MISSING_CONVERSATION_ID = 4


def make_settings() -> Settings:
    return Settings(
        chat_base_url="http://localhost:11434/v1",
        openai_model="qwen3",
        openai_api_key="ollama",
        database_url=DATABASE_URL,
    )


class _ScalarRows:
    def __init__(self, rows: Iterable[Message]) -> None:
        self._rows = tuple(rows)

    def __iter__(self):
        return iter(self._rows)


class FakeHistorySession:
    def __init__(
        self,
        conversation: Conversation | None,
        messages: Iterable[Message] = (),
        *,
        error: Exception | None = None,
    ) -> None:
        self.conversation = conversation
        self.messages = tuple(messages)
        self.error = error
        self.scalar_reads = 0

    async def __aenter__(self) -> "FakeHistorySession":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, model: object, identity: object) -> Conversation | None:
        if self.error is not None:
            raise self.error
        if self.conversation is None or self.conversation.id != identity:
            return None
        return self.conversation

    async def scalars(self, statement: object) -> _ScalarRows:
        self.scalar_reads += 1
        return _ScalarRows(self.messages)


class FakeSessionFactory:
    def __init__(self, session: FakeHistorySession) -> None:
        self.session = session
        self.calls = 0

    def __call__(self) -> FakeHistorySession:
        self.calls += 1
        return self.session


class _UnusedToolChatService:
    pass


def message(
    row_id: int,
    role: str,
    content: str,
    *,
    tool_calls: list[dict[str, object]] | None = None,
    tool_call_id: str | None = None,
) -> Message:
    return Message(
        id=row_id,
        conversation_id=CONVERSATION_ID,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


def get_history(factory: FakeSessionFactory, user_id: str, conversation_id: int):
    app = create_app(
        settings=make_settings(),
        model=FakeCombinedModel(),
        session_factory=factory,
        tool_chat_service=_UnusedToolChatService(),
    )
    with TestClient(app) as client:
        return client.get(
            f"/v1/conversations/{conversation_id}/messages",
            headers={"X-Demo-User-Id": user_id},
        )


def test_history_returns_ordered_public_rows_and_only_safe_tool_trace() -> None:
    conversation = Conversation(
        id=CONVERSATION_ID,
        user_id=OWNER_ID,
        status="进行中",
    )
    factory = FakeSessionFactory(
        FakeHistorySession(
            conversation,
            [
                message(1, "user", "订单到哪了"),
                message(
                    2,
                    "assistant",
                    "",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "query_logistics",
                            "args": {"order_id": "private-1001"},
                        }
                    ],
                ),
                message(
                    3,
                    "tool",
                    '{"internal_location":"warehouse-secret"}',
                    tool_call_id="call-1",
                ),
                message(4, "assistant", "包裹正在配送中。"),
            ],
        )
    )

    response = get_history(factory, OWNER_ID, CONVERSATION_ID)

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 1,
            "role": "user",
            "content": "订单到哪了",
            "tool_trace": [],
            "sources": [],
            "suggested_actions": [],
            "related_order_id": None,
            "feedback": None,
        },
        {
            "id": 4,
            "role": "assistant",
            "content": "包裹正在配送中。",
            "tool_trace": [
                {
                    "tool_call_id": "call-1",
                    "name": "query_logistics",
                    "label": "查询物流",
                    "status": "succeeded",
                    "source": "builtin",
                    "mcp_server": None,
                }
            ],
            "sources": [],
            "suggested_actions": [],
            "related_order_id": None,
            "feedback": None,
        },
    ]
    assert "private-1001" not in response.text
    assert "warehouse-secret" not in response.text
    assert factory.session.scalar_reads == 1


def test_missing_and_wrong_owner_have_the_same_opaque_404() -> None:
    conversation = Conversation(
        id=CONVERSATION_ID,
        user_id=OWNER_ID,
        status="进行中",
    )
    wrong_owner = get_history(
        FakeSessionFactory(FakeHistorySession(conversation)),
        OTHER_USER_ID,
        CONVERSATION_ID,
    )
    missing = get_history(
        FakeSessionFactory(FakeHistorySession(None)),
        OWNER_ID,
        MISSING_CONVERSATION_ID,
    )

    assert wrong_owner.status_code == missing.status_code == 404
    assert wrong_owner.json() == missing.json() == {
        "detail": {"code": "conversation_not_found", "message": "会话不存在"}
    }
    assert str(CONVERSATION_ID) not in wrong_owner.text


def test_history_requires_uuid_owner_header_and_positive_integer_path() -> None:
    factory = FakeSessionFactory(FakeHistorySession(None))
    app = create_app(
        settings=make_settings(),
        model=FakeCombinedModel(),
        session_factory=factory,
        tool_chat_service=_UnusedToolChatService(),
    )

    with TestClient(app) as client:
        missing_header = client.get(f"/v1/conversations/{CONVERSATION_ID}/messages")
        invalid_header = client.get(
            f"/v1/conversations/{CONVERSATION_ID}/messages",
            headers={"X-Demo-User-Id": "not-a-uuid"},
        )
        invalid_path = client.get(
            "/v1/conversations/not-a-uuid/messages",
            headers={"X-Demo-User-Id": OWNER_ID},
        )

    assert missing_header.status_code == 422
    assert invalid_header.status_code == 422
    assert invalid_path.status_code == 422
    assert factory.calls == 0


def test_history_database_failure_is_safe_http_503() -> None:
    factory = FakeSessionFactory(
        FakeHistorySession(
            None,
            error=OperationalError(
                "SELECT password FROM private_table",
                {},
                RuntimeError("database credential=secret"),
            ),
        )
    )

    response = get_history(factory, OWNER_ID, CONVERSATION_ID)

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "persistence_unavailable",
            "message": "消息服务暂时不可用，请稍后重试",
        }
    }
    assert "private_table" not in response.text
    assert "credential" not in response.text
