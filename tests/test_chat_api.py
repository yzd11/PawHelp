"""Public SSE chat endpoint behavior."""

import json
from collections.abc import AsyncIterator, Sequence

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.chat import ChatEvent
from app.config import Settings
from app.main import create_app
from app.repositories.conversations import ConversationNotFound
from app.schemas import ChatRequest
from tests.fakes import FakeCombinedModel


USER_ID = "00000000-0000-0000-0000-000000000001"


class _UnusedSessionFactory:
    def __call__(self) -> object:
        raise AssertionError("the injected service owns this test boundary")


class FakeToolChatService:
    def __init__(
        self,
        events: Sequence[ChatEvent] = (),
        *,
        prepare_error: Exception | None = None,
    ) -> None:
        self.events = tuple(events)
        self.prepare_error = prepare_error
        self.requests: list[ChatRequest] = []

    async def prepare_turn(self, request: ChatRequest) -> object:
        self.requests.append(request)
        if self.prepare_error is not None:
            raise self.prepare_error
        return object()

    async def stream_turn(self, turn: object) -> AsyncIterator[ChatEvent]:
        for event in self.events:
            yield event


def _settings() -> Settings:
    return Settings(
        chat_base_url="http://localhost:11434/v1",
        openai_model="qwen3",
        openai_api_key="ollama",
        database_url="mysql+asyncmy://pawhelp:pawhelp@127.0.0.1:3306/pawhelp",
    )


def _post(service: FakeToolChatService, body: object):
    app = create_app(
        settings=_settings(),
        model=FakeCombinedModel(),
        session_factory=_UnusedSessionFactory(),
        tool_chat_service=service,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        return client.post("/v1/chat/stream", json=body)


def _events(payload: str) -> list[tuple[str, dict[str, object]]]:
    result: list[tuple[str, dict[str, object]]] = []
    for block in payload.strip().split("\n\n"):
        lines = block.splitlines()
        result.append((lines[0][7:], json.loads(lines[1][6:])))
    return result


def test_first_chat_turn_accepts_no_conversation_id_and_returns_numeric_id() -> None:
    service = FakeToolChatService(
        [
            ChatEvent("conversation", {"conversation_id": 42}),
            ChatEvent("token", {"content": "您好"}),
            ChatEvent(
                "done",
                {
                    "content": "您好",
                    "conversation_id": 42,
                    "message_id": 7,
                    "input_tokens_estimate": 12,
                    "trimmed_messages": 0,
                    "tool_trace": [],
                },
            ),
        ]
    )

    response = _post(service, {"user_id": USER_ID, "message": "你好"})

    assert response.status_code == 200
    assert _events(response.text)[0] == ("conversation", {"conversation_id": 42})
    assert service.requests[0].conversation_id is None


def test_existing_chat_turn_accepts_a_positive_numeric_conversation_id() -> None:
    service = FakeToolChatService(
        [ChatEvent("conversation", {"conversation_id": 42})]
    )

    response = _post(
        service,
        {"user_id": USER_ID, "conversation_id": 42, "message": "继续说"},
    )

    assert response.status_code == 200
    assert service.requests[0].conversation_id == 42


def test_unknown_conversation_is_a_safe_404() -> None:
    response = _post(
        FakeToolChatService(prepare_error=ConversationNotFound(42)),
        {"user_id": USER_ID, "conversation_id": 42, "message": "继续说"},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "conversation_not_found"


def test_database_failure_is_a_safe_503() -> None:
    response = _post(
        FakeToolChatService(
            prepare_error=OperationalError(
                "SELECT password FROM private_table",
                {},
                RuntimeError("database credential=secret"),
            )
        ),
        {"user_id": USER_ID, "message": "你好"},
    )

    assert response.status_code == 503
    assert "secret" not in response.text
