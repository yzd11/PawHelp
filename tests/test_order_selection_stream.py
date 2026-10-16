"""Exercise order interruption and resumption through the public SSE transport."""

from dataclasses import replace
import json

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import create_api_router
from app.config import Settings
from app.repositories.conversations import ConversationRepository
from tests.workflow_fakes import InMemoryWorkflow, KeywordModel, LabelledSearch, turn


USER_ID = "00000000-0000-0000-0000-000000000001"


class OwnedSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


def events(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    return [
        (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
        for block in response.text.strip().split("\n\n")
        if (lines := block.splitlines())
    ]


async def test_missing_order_card_reaches_browser_and_click_resumes_workflow(monkeypatch):
    model, search = KeywordModel(), LabelledSearch()
    service = InMemoryWorkflow(model, search)

    async def prepare(request):
        return replace(turn(request.message), user_id=str(request.user_id))

    async def get_owned(self, conversation_id, user_id):
        assert conversation_id == 1 and user_id == USER_ID
        return object()

    monkeypatch.setattr(service, "prepare_turn", prepare)
    monkeypatch.setattr(ConversationRepository, "get_owned", get_owned)
    settings = Settings(
        chat_base_url="http://localhost:11434/v1", openai_model="fake",
        openai_api_key="fake", database_url="mysql+asyncmy://user:pass@localhost/db",
    )
    app = FastAPI()
    app.include_router(create_api_router(settings, model, OwnedSession, service, None))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/stream", json={"user_id": USER_ID, "message": "这个能退吗"},
        )
        initial = events(response)
        assert initial[0] == ("conversation", {"conversation_id": 1})
        assert initial[-1][0] == "order_selection"
        selection = initial[-1][1]
        assert "演示订单" in selection["message"]
        assert [item["order_id"] for item in selection["orders"]] == ["1001", "1002", "1003", "1004"]
        assert model.agent_calls == 0
        assert await service.pending_order(USER_ID, 1) is not None

        response = await client.post(
            "/v1/conversations/1/select-order",
            json={"user_id": USER_ID, "message_id": selection["message_id"], "order_id": "1001"},
        )
        resumed = events(response)
        assert resumed[-1][0] == "done"
        assert resumed[-1][1]["content"]
        assert [item["name"] for item in resumed[-1][1]["tool_trace"]] == ["query_order"]
        assert model.agent_calls == 1 and search.queries
        assert await service.pending_order(USER_ID, 1) is None
