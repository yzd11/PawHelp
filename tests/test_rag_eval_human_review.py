"""Human review changes the ledger, never the model's evaluation report."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.api.evaluation as evaluation_module
from app.config import Settings
from app.main import RAG_EVAL_PAGE_PATH, create_app
from tests.fakes import FakeChatModel


class _Result:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.rows = rows

    def scalars(self) -> "_Result":
        return self

    def all(self) -> list[SimpleNamespace]:
        return self.rows

    def scalar_one_or_none(self) -> SimpleNamespace | None:
        return self.rows[0] if self.rows else None


class _Session:
    def __init__(self, row: SimpleNamespace | None) -> None:
        self.row = row
        self.fail = False
        self.flushed = 0

    async def __aenter__(self) -> "_Session":
        if self.fail:
            raise RuntimeError("database unavailable")
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    @asynccontextmanager
    async def begin(self):
        yield self

    async def execute(self, statement: object) -> _Result:
        return _Result([self.row] if self.row else [])

    async def flush(self) -> None:
        self.flushed += 1


def _app(session: _Session):
    settings = Settings(
        chat_base_url="http://localhost:11434/v1",
        openai_model="qwen3",
        openai_api_key="ollama",
        database_url="mysql+asyncmy://pawhelp:pawhelp@127.0.0.1:3306/pawhelp",
    )
    return create_app(
        settings=settings, model=FakeChatModel(), session_factory=lambda: session
    )


def test_only_flagged_production_answer_can_be_reviewed(monkeypatch) -> None:
    snapshot = {
        "case_id": "A17", "strategy": "hybrid_rerank", "faithful": False,
        "answer": "有争议的答案", "faithfulness_reason": "引用不足",
    }
    report = {"cases": [
        snapshot,
        {"case_id": "A18", "strategy": "hybrid_rerank", "faithful": True},
        {"case_id": "A19", "strategy": "dense", "faithful": False},
    ]}
    monkeypatch.setattr(evaluation_module, "_read_rag_eval_report", lambda: report)
    row = SimpleNamespace(
        eval_id="A17", answer="有争议的答案", reason="引用不足",
        status="未解决", resolution=None, resolved_at=None, seen_count=2,
    )
    session = _Session(row)
    with TestClient(_app(session)) as client:
        listing = client.get("/v1/rag-eval/faith-cases")
        assert listing.status_code == 200
        assert list(listing.json()["cases"]) == ["A17"]
        assert listing.json()["cases"]["A17"]["status"] == "未解决"

        url = "/v1/rag-eval/faith-cases/A17/status"
        request = {
            "status": "无需解决", "resolution": "  模型把有据的提醒误判为编造  ",
            "expected_seen_count": 2, "expected_status": "未解决",
            "expected_resolution": None,
        }
        assert client.post(url, json={**request, "resolution": "  "}).status_code == 400
        assert client.post(url, json={**request, "status": "别的状态"}).status_code == 422
        assert client.post("/v1/rag-eval/faith-cases/A18/status", json=request).status_code == 404
        saved = client.post(url, json=request)
        assert saved.status_code == 200
        assert saved.json()["status"] == "无需解决"
        assert saved.json()["resolution"] == "模型把有据的提醒误判为编造"
        assert session.flushed == 1
        assert client.get("/v1/rag-eval/faith-cases").json()["cases"]["A17"]["status"] == "无需解决"
        assert report["cases"][0]["faithful"] is False

        assert client.post(url, json=request).status_code == 409
        reopened = client.post(url, json={
            **request, "status": "未解决", "resolution": None,
            "expected_status": "无需解决",
            "expected_resolution": "模型把有据的提醒误判为编造",
        })
        assert reopened.status_code == 200
        assert reopened.json()["resolution"] is None
        assert reopened.json()["resolved_at"] is None


def test_changed_report_and_unavailable_database_are_read_only(monkeypatch) -> None:
    snapshot = {
        "case_id": "A17", "strategy": "hybrid_rerank", "faithful": False,
        "answer": "新答案", "faithfulness_reason": "新理由",
    }
    monkeypatch.setattr(evaluation_module, "_read_rag_eval_report", lambda: {"cases": [snapshot]})
    row = SimpleNamespace(
        eval_id="A17", answer="旧答案", reason="旧理由", status="未解决",
        resolution=None, resolved_at=None, seen_count=1,
    )
    session = _Session(row)
    request = {
        "status": "已解决", "resolution": "已经修复知识库",
        "expected_seen_count": 1, "expected_status": "未解决",
    }
    with TestClient(_app(session)) as client:
        assert client.get("/v1/rag-eval/faith-cases").json()["cases"]["A17"]["available"] is False
        assert client.post("/v1/rag-eval/faith-cases/A17/status", json=request).status_code == 409
        session.fail = True
        assert client.get("/v1/rag-eval/faith-cases").status_code == 503
        assert client.post("/v1/rag-eval/faith-cases/A17/status", json=request).status_code == 503
        assert row.status == "未解决"


def test_page_keeps_model_findings_separate_from_human_decisions() -> None:
    page = RAG_EVAL_PAGE_PATH.read_text(encoding="utf-8")
    assert 'if (row.faithful === false) body.appendChild(faithReviewBox(row));' in page
    assert 'if (status !== "未解决" && !resolution)' in page
    assert 'expected_resolution: review.resolution' in page
    assert "原始模型分数不变" in page
    assert "已解决：问题已实际修复" in page
