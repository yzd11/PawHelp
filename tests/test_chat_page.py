"""Browser chat page delivery smoke test."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import CHAT_PAGE_PATH, create_app
from tests.fakes import FakeChatModel

DATABASE_URL = "mysql+asyncmy://pawhelp:pawhelp@127.0.0.1:3306/pawhelp"


def test_chat_page_is_served_from_app_static() -> None:
    settings = Settings(
        chat_base_url="http://localhost:11434/v1",
        openai_model="qwen3",
        openai_api_key="ollama",
        database_url=DATABASE_URL,
    )

    with TestClient(create_app(settings=settings, model=FakeChatModel())) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert CHAT_PAGE_PATH == Path(__file__).parents[1] / "app" / "static" / "index.html"
    assert response.content == CHAT_PAGE_PATH.read_bytes()
    # Page wiring that only greps the HTML source belongs in browser checks, not here.



def test_chat_markdown_assets_are_served_locally() -> None:
    settings = Settings(chat_base_url="http://localhost:11434/v1", openai_model="qwen3",
                        openai_api_key="ollama", database_url=DATABASE_URL)
    with TestClient(create_app(settings=settings, model=FakeChatModel())) as client:
        for name in ["chat-markdown.js", "vendor/marked-18.1.0.umd.js", "vendor/purify-3.4.16.min.js"]:
            response = client.get("/static/" + name)
            assert response.status_code == 200
            assert "javascript" in response.headers["content-type"]
            assert response.content == (CHAT_PAGE_PATH.parent / name).read_bytes()
        assert client.get("/static/missing-file.js").status_code == 404
