from pathlib import Path
import os
import subprocess
import sys
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings
from app.main import create_app

DATABASE_URL = "mysql+asyncmy://pawhelp:pawhelp@127.0.0.1:3306/pawhelp"


class NeverCalledModel:
    def __init__(self) -> None:
        self.calls = 0
        self.structured_bindings = 0

    def with_structured_output(
        self, schema: object, **kwargs: object
    ) -> "NeverInvokedStructuredRunnable":
        self.structured_bindings += 1
        return NeverInvokedStructuredRunnable()


class NeverInvokedStructuredRunnable:
    async def ainvoke(self, messages: object) -> object:
        raise AssertionError("health must not invoke the structured runnable")


class NeverCalledSessionFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        raise AssertionError("health must not open a database session")


class DisposableEngine:
    def __init__(self) -> None:
        self.dispose_calls = 0

    async def dispose(self) -> None:
        self.dispose_calls += 1


def make_settings() -> Settings:
    return Settings(
        chat_base_url="http://localhost:11434/v1",
        openai_model="qwen3",
        openai_api_key="ollama",
        database_url=DATABASE_URL,
    )


def test_health_does_not_call_model() -> None:
    model = NeverCalledModel()
    session_factory = NeverCalledSessionFactory()
    application = create_app(
        settings=make_settings(),
        model=model,
        session_factory=session_factory,
    )
    startup_bindings = model.structured_bindings

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert model.structured_bindings == startup_bindings
    assert model.calls == 0
    assert session_factory.calls == 0


def test_application_owned_engine_is_disposed_at_lifespan_shutdown(monkeypatch) -> None:
    engine = DisposableEngine()
    session_factory = NeverCalledSessionFactory()
    monkeypatch.setattr(
        main_module, "create_async_engine_from_settings", lambda settings: engine
    )
    monkeypatch.setattr(
        main_module, "create_session_factory", lambda supplied_engine: session_factory
    )

    app = create_app(settings=make_settings(), model=NeverCalledModel())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert engine.dispose_calls == 0

    assert engine.dispose_calls == 1


def test_supplied_session_factory_does_not_create_or_dispose_an_engine(
    monkeypatch,
) -> None:
    session_factory = NeverCalledSessionFactory()

    def unexpected_engine(settings: Settings) -> object:
        raise AssertionError("a supplied session factory must retain engine ownership")

    monkeypatch.setattr(main_module, "create_async_engine_from_settings", unexpected_engine)

    with TestClient(
        create_app(
            settings=make_settings(),
            model=NeverCalledModel(),
            session_factory=session_factory,
        )
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert session_factory.calls == 0


def test_uvicorn_start_up_does_not_import_sentence_transformers(tmp_path) -> None:
    # langchain_text_splitters' package import pulls in sentence-transformers (~16s on start-up).
    code = "import sys, app.main; sys.exit('sentence_transformers' in sys.modules)"
    # An empty working directory has no .env, as in CI; the fake settings stand in for it.
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1]), "CHAT_BASE_URL": "http://localhost/v1",
           "OPENAI_MODEL": "fake", "OPENAI_API_KEY": "fake", "DATABASE_URL": "mysql+asyncmy://unused"}
    assert subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env).returncode == 0


@pytest.mark.parametrize("fails", [False, True])
def test_preloaded_knowledge_models_report_loading_then_outcome(monkeypatch, fails) -> None:
    release, warmed = threading.Event(), []

    def embedder_warm(self) -> None:
        release.wait(5)
        if fails:
            raise RuntimeError("CUDA out of memory")
        warmed.append("embedder")

    monkeypatch.setattr(main_module.BgeM3Embedder, "warm", embedder_warm)
    monkeypatch.setattr(main_module.BgeReranker, "warm", lambda self: warmed.append("reranker"))
    app = create_app(settings=make_settings(), model=NeverCalledModel(),
                     session_factory=NeverCalledSessionFactory(), preload_models=True)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "knowledge_models": "loading"}
        release.set()
        for _ in range(100):
            body = client.get("/health").json()
            if body["knowledge_models"] != "loading":
                break
            time.sleep(0.05)
    assert body == {"status": "ok", "knowledge_models": "failed" if fails else "ready"}
    assert sorted(warmed) == (["reranker"] if fails else ["embedder", "reranker"])
