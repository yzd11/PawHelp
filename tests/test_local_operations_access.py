"""Operations HTTP seam: no database, no model, and no real mutations."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.api.operations import operations_router

LOCAL_HEADERS = {"X-PawHelp-Local-Admin": "1"}
ENDPOINTS = [
    ("GET", "/api/review", None),
    ("GET", "/api/review/1", None),
    ("POST", "/api/review/process", None),
    ("POST", "/api/review/1/approve", {"approved_answer": "审核答案"}),
    ("POST", "/api/review/1/reject", None),
    ("POST", "/api/review/1/publish", None),
    ("GET", "/api/observability", None),
    ("POST", "/api/observability/evaluate", None),
    ("POST", "/api/observability/calibration/analyze", None),
]


@pytest.fixture
def operations(monkeypatch):
    settings = SimpleNamespace(admin_token=SecretStr(""), langfuse_base_url="http://localhost:3000",
        flywheel_interval_seconds=0, evaluation_interval_seconds=0,
        evaluation_sample_per_bucket=2, evaluation_max_model_calls=25)
    flywheel = SimpleNamespace(**{name: AsyncMock(return_value={"ok": True})
        for name in ("list", "detail", "process", "review", "publish", "feedback")})
    pipeline = SimpleNamespace(history=AsyncMock(return_value=[]), run=AsyncMock(return_value={"ok": True}), last_error=None)
    monkeypatch.setattr("app.api.operations.cost_by_intent", AsyncMock(return_value={"items": []}))
    monkeypatch.setattr("app.api.operations.calibration_report", lambda *_: {"available": False})
    app = FastAPI()
    app.include_router(operations_router(settings, None, flywheel, pipeline))
    return app, settings, flywheel, pipeline


@pytest.mark.parametrize("peer,base_url", [("127.0.0.1", "http://localhost:8000"),
    ("::1", "http://[::1]:8000"), ("127.0.0.1", "http://127.0.0.1:8000")])
async def test_local_pages_read_and_mutate_without_admin_token(operations, peer, base_url):
    app, _, flywheel, pipeline = operations
    headers = {**LOCAL_HEADERS, "Origin": base_url, "Sec-Fetch-Site": "same-origin"}
    async with AsyncClient(transport=ASGITransport(app=app, client=(peer, 123)), base_url=base_url) as client:
        for method, path, body in ENDPOINTS:
            response = await client.request(method, path, headers=headers, json=body)
            assert response.status_code == 200, response.text
    assert flywheel.review.await_count == 2 and pipeline.run.await_count == 1


@pytest.mark.parametrize("peer,headers", [
    ("192.168.1.10", LOCAL_HEADERS),
    ("127.0.0.1", {**LOCAL_HEADERS, "Host": "attacker.example:8000"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "Origin": "https://attacker.example"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "Origin": "null"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "Origin": "http://localhost:9000"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "Origin": "http://127.0.0.1:8000"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "Sec-Fetch-Site": "cross-site"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "Sec-Fetch-Site": "same-site"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "Forwarded": "for=192.168.1.10"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "X-Forwarded-For": "127.0.0.1"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "X-Forwarded-Host": "localhost:8000"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "X-Forwarded-Proto": "http"}),
    ("127.0.0.1", {**LOCAL_HEADERS, "Host": "localhost:invalid"}),
])
async def test_local_bypass_rejects_remote_cross_origin_and_proxy_requests(operations, peer, headers):
    app, _, flywheel, pipeline = operations
    async with AsyncClient(transport=ASGITransport(app=app, client=(peer, 123)), base_url="http://localhost:8000") as client:
        for method, path, body in ENDPOINTS:
            response = await client.request(method, path, headers=headers, json=body)
            assert response.status_code == 403
    assert all(getattr(flywheel, name).await_count == 0 for name in ("list", "detail", "process", "review", "publish"))
    assert pipeline.run.await_count == pipeline.history.await_count == 0


async def test_no_custom_header_still_requires_token_and_legacy_clients_work(operations):
    app, settings, flywheel, _ = operations
    async with AsyncClient(transport=ASGITransport(app=app, client=("192.168.1.10", 123)), base_url="http://localhost:8000") as client:
        assert (await client.post("/api/review/process")).status_code == 503
        settings.admin_token = SecretStr("test-only-admin")
        assert (await client.post("/api/review/process")).status_code == 401
        assert (await client.post("/api/review/process", headers={"X-Admin-Token": "wrong"})).status_code == 401
        assert (await client.post("/api/review/process", headers={"X-Admin-Token": "test-only-admin"})).status_code == 200
    assert flywheel.process.await_count == 1


async def test_local_html_form_cannot_bypass_with_plain_post(operations):
    app, _, flywheel, _ = operations
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost:8000") as client:
        assert (await client.post("/api/review/process", headers={"Origin": "https://attacker.example"})).status_code == 503
    flywheel.process.assert_not_awaited()


def test_all_pages_use_shared_navigation_and_operations_have_no_token_form():
    root = Path(__file__).resolve().parents[1] / "app/static"
    for name in ("index", "kb", "rag-eval", "review", "observability"):
        page = (root / f"{name}.html").read_text(encoding="utf-8")
        assert "data-app-navigation" in page
        assert '/static/navigation.js" defer' in page
        assert '/static/navigation.css' in page
        if name in {"review", "observability"}:
            assert 'id="token"' not in page
            assert "管理令牌" not in page


@pytest.mark.parametrize("days", [0, 91, "bad"])
async def test_observability_rejects_invalid_window(operations, days):
    app, _, _, pipeline = operations
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost:8000") as client:
        response = await client.get(f"/api/observability?days={days}", headers=LOCAL_HEADERS)
        assert response.status_code == 422
    pipeline.history.assert_not_awaited()


async def test_offline_calibration_never_runs_paid_evaluation(operations):
    app, _, flywheel, pipeline = operations
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost:8000") as client:
        response = await client.post("/api/observability/calibration/analyze", headers=LOCAL_HEADERS)
        assert response.json() == {"available": False}
    pipeline.run.assert_not_awaited()
    flywheel.process.assert_not_awaited()
