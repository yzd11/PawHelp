"""RAG evaluation console delivery and report API contracts."""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import RAG_EVAL_PAGE_PATH, create_app
from tests.fakes import FakeChatModel


DATABASE_URL = "mysql+asyncmy://pawhelp:pawhelp@127.0.0.1:3306/pawhelp"


def _settings() -> Settings:
    return Settings(
        chat_base_url="http://localhost:11434/v1",
        openai_model="qwen3",
        openai_api_key="ollama",
        database_url=DATABASE_URL,
    )


def test_rag_eval_page_and_committed_report_are_served() -> None:
    with TestClient(create_app(settings=_settings(), model=FakeChatModel())) as client:
        page = client.get("/rag-eval")
        report = client.get("/v1/rag-eval/report")

    assert page.status_code == 200
    assert page.content == RAG_EVAL_PAGE_PATH.read_bytes()
    assert "RAG 质量评估" in page.text
    assert "重新运行评估" in page.text
    assert "完整数据" in page.text
    assert "需人工复核的问题（仅混合 + Rerank）" in page.text
    assert "逐题证据审计" in page.text
    assert "当前评测题已修改，页面显示的是旧题目的历史分数" in page.text
    assert "知识块回答" in page.text
    assert 'document.createTextNode(String(source.answer || "").trim() || "该知识块没有回答正文")' in page.text
    assert "评估步骤报错" in page.text
    assert "生成答案：" in page.text
    assert "检查答案要点：" in page.text
    assert "核查事实依据：" in page.text
    assert "疑似编造回答" in page.text
    assert "保存失败" in page.text
    assert "模型调用失败" not in page.text
    assert "台账写入失败" not in page.text
    assert '["库内误拒答率"' in page.text
    assert '["口语类 MRR 提升"' not in page.text
    assert '["答案语义覆盖"' not in page.text
    assert 'id="fullGeneration" type="checkbox" checked' in page.text
    assert "旧口径报告，不含答案语义覆盖度" in page.text
    assert 'id="issuePagination"' in page.text
    assert "ISSUE_PAGE_SIZE = 10" in page.text
    assert 'let selectedIssueFilter = "all"' in page.text
    assert "召回材料要点命中（字面）" in page.text
    assert "最终答案要点覆盖（模型）" in page.text
    assert "回答事实有据（模型）" in page.text
    assert 'id="retrievalMetricHelp"' in page.text
    assert 'id="generationMetricHelp"' in page.text
    assert 'id="productionBreakdown" hidden' in page.text
    assert "productionBreakdown.hidden = !generationEnabled || showComparison" in page.text
    assert 'id="refusalCard"' not in page.text
    assert report.status_code == 200
    payload = report.json()
    assert payload["dataset_alignment"]["status"] in {"current", "stale", "unverified"}
    assert set(payload["strategies"]) == {
        "dense",
        "bm25",
        "hybrid",
        "hybrid_rerank",
    }
    assert len(payload["cases"]) == (
        payload["dataset"]["total_cases"] * len(payload["strategies"])
    )
    assert any(
        isinstance(citation.get("answer"), str) and citation["answer"]
        for row in payload["cases"]
        for citation in row.get("citations", [])
    )


def test_rag_eval_rerun_requires_confirmation_and_returns_runner_report() -> None:
    calls = 0

    async def fake_runner(full_generation: bool) -> dict[str, object]:
        nonlocal calls
        assert full_generation is True
        calls += 1
        return {"strategies": {"dense": {"mrr": 1.0}}, "cases": []}

    with TestClient(
        create_app(
            settings=_settings(),
            model=FakeChatModel(),
            rag_eval_runner=fake_runner,
        )
    ) as client:
        missing = client.post("/v1/rag-eval/run", json={})
        complete = client.post(
            "/v1/rag-eval/run",
            json={"confirm_external_model_usage": True},
        )

    assert missing.status_code == 422
    assert complete.status_code == 200
    assert complete.json()["strategies"]["dense"]["mrr"] == 1.0
    assert calls == 1


def test_rag_eval_can_request_four_strategy_answer_generation() -> None:
    requested: list[bool] = []

    async def fake_runner(full_generation: bool) -> dict[str, object]:
        requested.append(full_generation)
        return {"strategies": {}, "cases": []}

    with TestClient(
        create_app(
            settings=_settings(),
            model=FakeChatModel(),
            rag_eval_runner=fake_runner,
        )
    ) as client:
        response = client.post(
            "/v1/rag-eval/run",
            json={"confirm_external_model_usage": True, "full_generation": True},
        )

    assert response.status_code == 200
    assert requested == [True]


def test_rag_eval_can_limit_generation_to_production_route() -> None:
    requested: list[bool] = []

    async def fake_runner(full_generation: bool) -> dict[str, object]:
        requested.append(full_generation)
        return {"strategies": {}, "cases": []}

    with TestClient(
        create_app(settings=_settings(), model=FakeChatModel(), rag_eval_runner=fake_runner)
    ) as client:
        response = client.post(
            "/v1/rag-eval/run",
            json={"confirm_external_model_usage": True, "full_generation": False},
        )

    assert response.status_code == 200
    assert requested == [False]
