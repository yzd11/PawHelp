"""Customer feedback and local-only, same-origin operations access."""
import asyncio
import hmac
from ipaddress import ip_address
import logging
from typing import Annotated, Literal
from uuid import UUID
from urllib.parse import urlsplit
from pathlib import Path
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import text
from app.repositories.conversations import ConversationNotFound
from app.evaluation.pipeline import cost_by_intent
from app.evaluation.dashboard import calibration_report, trend_report
from app.services.knowledge_admin import KnowledgeCheckUnavailable

logger = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    vote: Literal["up", "down"]


class ReviewRequest(BaseModel):
    approved_answer: str = Field(min_length=1, max_length=10000)


def _local_page_request(request: Request) -> bool:
    """Trust neither forwarded addresses nor a Host resolved through DNS."""
    try:
        if request.client is None or not ip_address(request.client.host).is_loopback:
            return False
        if any(name in request.headers for name in (
            "forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
        )):
            return False
        host = urlsplit("//" + request.headers.get("host", ""))
        if host.hostname not in {"localhost", "127.0.0.1", "::1"} or host.username or host.password or host.path:
            return False
        port = host.port or (443 if request.url.scheme == "https" else 80)
        origin = request.headers.get("origin")
        if origin is not None:
            parsed = urlsplit(origin)
            if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
                return False
            origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if (parsed.scheme, parsed.hostname, origin_port) != (request.url.scheme, host.hostname, port):
                return False
        if request.headers.get("sec-fetch-site", "none") not in {"same-origin", "none"}:
            return False
        return True
    except ValueError:
        return False


def operations_router(settings, sessions, flywheel, pipeline):
    router = APIRouter()
    def authorize(request: Request, x_admin_token: Annotated[str | None, Header()] = None):
        # A non-secret custom header cannot be sent by a cross-origin HTML form.
        # Browser preflight is NOT the sole guard: Origin and peer are checked too.
        if request.headers.get("x-pawhelp-local-admin") == "1":
            if not _local_page_request(request):
                raise HTTPException(403, detail="后台免令牌仅支持本机同源访问，请使用 localhost 或 127.0.0.1 打开页面")
            return
        expected = settings.admin_token.get_secret_value()
        if not expected: raise HTTPException(503, detail="未配置 ADMIN_TOKEN，后台接口未开放")
        if not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), expected.encode()): raise HTTPException(401, detail="管理令牌无效")

    admin = APIRouter(dependencies=[Depends(authorize)])

    @router.post("/v1/conversations/{conversation_id}/messages/{message_id}/feedback")
    async def feedback(conversation_id: int, message_id: int, request: FeedbackRequest, user_id: Annotated[UUID, Header(alias="X-Demo-User-Id")]):
        try: return await flywheel.feedback(conversation_id, message_id, str(user_id), request.vote)
        except (ConversationNotFound, LookupError) as error: raise HTTPException(404, detail="回答不存在") from error

    @admin.get("/api/review")
    async def list_reviews(status: Literal["全部", "待审", "通过", "驳回"] = "待审", offset: int = 0, sort: Literal["frequency", "newest"] = "frequency"):
        if offset < 0: raise HTTPException(422, detail="offset must be non-negative")
        return await flywheel.list(status, offset, sort)

    @admin.post("/api/review/process")
    async def process():
        try: return await flywheel.process()
        except ValueError as error: raise HTTPException(409, detail=str(error)) from error

    @admin.get("/api/review/{review_id}")
    async def detail(review_id: int):
        try: return await flywheel.detail(review_id)
        except LookupError as error: raise HTTPException(404, detail=str(error)) from error

    @admin.post("/api/review/{review_id}/approve")
    async def approve(review_id: int, request: ReviewRequest):
        try: return await flywheel.review(review_id, request.approved_answer)
        except LookupError as error: raise HTTPException(404, detail=str(error)) from error
        except ValueError as error: raise HTTPException(409, detail=str(error)) from error
        except KnowledgeCheckUnavailable as error: raise HTTPException(503, detail=str(error)) from error

    @admin.post("/api/review/{review_id}/reject")
    async def reject(review_id: int):
        try: return await flywheel.review(review_id, reject=True)
        except LookupError as error: raise HTTPException(404, detail=str(error)) from error
        except ValueError as error: raise HTTPException(409, detail=str(error)) from error

    @admin.post("/api/review/{review_id}/publish")
    async def publish(review_id: int):
        try: return await flywheel.publish(review_id)
        except ValueError as error: raise HTTPException(409, detail=str(error)) from error

    @admin.get("/api/observability")
    async def overview(days: Annotated[int, Query(ge=1, le=90)] = 7):
        history = await pipeline.history()
        calibration = await asyncio.to_thread(calibration_report, settings, getattr(pipeline, "root", Path(__file__).resolve().parents[2]))
        return {"costs": await cost_by_intent(sessions, settings, days=days), "eval_runs": history,
                "trend": trend_report(history), "calibration": calibration,
                "last_evaluation_error": pipeline.last_error,
                "configuration": {"langfuse_url": settings.langfuse_base_url, "flywheel_interval_seconds": settings.flywheel_interval_seconds,
                    "evaluation_interval_seconds": settings.evaluation_interval_seconds, "sample_per_bucket": settings.evaluation_sample_per_bucket,
                    "max_model_calls": settings.evaluation_max_model_calls}}

    @admin.post("/api/observability/calibration/analyze")
    async def analyze_calibration():
        return await asyncio.to_thread(calibration_report, settings, getattr(pipeline, "root", Path(__file__).resolve().parents[2]))

    @admin.post("/api/observability/evaluate")
    async def evaluate():
        try: return await pipeline.run()
        except ValueError as error: raise HTTPException(409, detail=str(error)) from error
        except Exception as error: raise HTTPException(503, detail="评估未完成，查看 log/app.log 与模型余额；没有生成虚假成功轮次") from error
    router.include_router(admin)
    return router


class ScheduledOperations:
    def __init__(self, settings, sessions, flywheel, pipeline):
        self.settings, self.sessions, self.flywheel, self.pipeline = settings, sessions, flywheel, pipeline
        self.tasks = []

    def start(self):
        for name, interval, work in (("pawhelp-flywheel", self.settings.flywheel_interval_seconds, self.flywheel.process),
                                     ("pawhelp-evaluation", self.settings.evaluation_interval_seconds, lambda: self.pipeline.run("定时"))):
            if interval:
                self.tasks.append(asyncio.create_task(self._loop(name, interval, work)))

    async def _loop(self, name, interval, work):
        while True:
            await asyncio.sleep(interval)
            try:
                # Connection-scoped lease prevents multiple Uvicorn workers running the same scheduled job.
                async with self.sessions() as lease:
                    if await lease.scalar(text("SELECT GET_LOCK(:name, 0)"), {"name": name}) != 1: continue
                    try: await work()
                    finally: await lease.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": name})
            except asyncio.CancelledError: raise
            except Exception: logger.exception("scheduled operation failed job=%s; next interval may retry", name)

    async def close(self):
        for task in self.tasks: task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
