"""Customer and operator page entry points, served by the same application."""
from pathlib import Path
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from app.api.operations import _local_page_request

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
CHAT_PAGE_PATH = STATIC_DIR / "index.html"
KNOWLEDGE_PAGE_PATH = STATIC_DIR / "kb.html"
RAG_EVAL_PAGE_PATH = STATIC_DIR / "rag-eval.html"


def page_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/client-storage-migration", include_in_schema=False)
    async def client_storage_migration(request: Request):
        """One-release local migration policy, never browser values or credentials."""
        if not _local_page_request(request):
            raise HTTPException(403, detail="本地会话迁移仅支持本机同源访问")
        path = STATIC_DIR.parents[1] / ".runtime" / "client-storage-migration.json"
        try:
            if path.stat().st_size > 4096:
                raise ValueError("policy too large")
            data = json.loads(path.read_text(encoding="utf-8"))
            allowed = {"pawhelp.userId", "pawhelp.conversationId", "pawhelp.messageFeedback.server"}
            if (not isinstance(data, dict) or not isinstance(data.get("id"), str)
                    or not isinstance(data.get("rules"), list) or len(data["rules"]) > 3
                    or any(not isinstance(row, dict) or row.get("target") not in allowed
                           or not isinstance(row.get("source"), str)
                           or row.get("kind") not in {"uuid", "positive_integer", "json_object"}
                           for row in data["rules"])):
                raise ValueError("invalid migration policy")
        except FileNotFoundError:
            data = {"id": None, "rules": []}
        except (OSError, UnicodeError, ValueError):
            raise HTTPException(503, detail="本地会话迁移配置无效，请联系维护人员") from None
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @router.get("/admin", include_in_schema=False)
    async def admin_home():
        return RedirectResponse("/review", status_code=307)

    @router.get("/", include_in_schema=False, response_class=FileResponse)
    async def chat_page() -> FileResponse:
        """Serve the browser chat client with bundled Markdown assets."""

        return FileResponse(
            CHAT_PAGE_PATH,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/kb", include_in_schema=False, response_class=FileResponse)
    async def knowledge_page() -> FileResponse:
        """Serve the dependency-free local knowledge console."""

        return FileResponse(
            KNOWLEDGE_PAGE_PATH,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/rag-eval", include_in_schema=False, response_class=FileResponse)
    async def rag_eval_page() -> FileResponse:
        return FileResponse(
            RAG_EVAL_PAGE_PATH,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/review", include_in_schema=False)
    async def review_page():
        return FileResponse(CHAT_PAGE_PATH.parent / "review.html", headers={"Cache-Control": "no-store"})

    @router.get("/observability", include_in_schema=False)
    async def observability_page():
        return FileResponse(CHAT_PAGE_PATH.parent / "observability.html", headers={"Cache-Control": "no-store"})

    return router
