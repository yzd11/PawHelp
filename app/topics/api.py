"""Local-only topic console: real persisted counts and fixed report artifacts."""
from pathlib import Path
from typing import Literal
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy.exc import SQLAlchemyError
from app.api.operations import _local_page_request
from app.topics.data import redact
from app.topics.pool import TopicPool
from app.topics.reports import acceptance_reports
from app.topics.taxonomy import TAXONOMY_HASH, TOPIC_NAMES, TOPICS


def topic_router(sessions, root: Path):
    router=APIRouter()
    pool=TopicPool(sessions)
    page=root/"app"/"static"/"topics.html"

    def authorize(request: Request):
        if request.headers.get("x-pawhelp-local-admin")!="1" or not _local_page_request(request):
            raise HTTPException(403,detail="分类器后台仅支持本机同源访问")

    admin=APIRouter(dependencies=[Depends(authorize)])

    @router.get("/topics",include_in_schema=False)
    @router.get("/topics/questions",include_in_schema=False)
    @router.get("/acceptance",include_in_schema=False)
    @router.get("/acceptance/eval",include_in_schema=False)
    @router.get("/acceptance/data",include_in_schema=False)
    @router.get("/acceptance/errors",include_in_schema=False)
    async def console():
        return FileResponse(page,headers={"Cache-Control":"no-store"})

    @admin.get("/api/topics/distribution")
    async def distribution():
        try:
            result=await pool.distribution()
        except SQLAlchemyError as exc:
            raise HTTPException(503,detail="主题数据暂不可用，请维护人员检查数据库连接与迁移状态") from exc
        result["taxonomy_hash"]=TAXONOMY_HASH
        result["taxonomy"]=[{"label":t.name,"boundary":t.boundary} for t in TOPICS]
        return result

    @admin.get("/api/topics/questions")
    async def questions(label: str,page: int=Query(1,ge=1),size: int=Query(10,ge=1,le=100),
                        q: str=Query("",max_length=200),
                        source: Literal["all","retrieval_low_conf","self_check","user_feedback"]="all",
                        review_status: Literal["all","待审","通过","驳回","unmerged"]="all"):
        if label not in TOPIC_NAMES: raise HTTPException(400,detail="未知主题类目")
        try:
            rows=await pool.questions()
        except SQLAlchemyError as exc:
            raise HTTPException(503,detail="主题表或数据库未就绪") from exc
        selected=[row for row in rows if row["labels"] and label in row["labels"]]
        # Filter the whole category before pagination, not only the visible page.
        needle=redact(q.strip()).casefold()
        if needle:
            selected=[row for row in selected if needle in redact(row["normalized_question"] or "").casefold()
                      or needle in redact(row["raw_question"]).casefold()]
        if source!="all":
            selected=[row for row in selected if row["source"]==source]
        if review_status!="all":
            selected=[row for row in selected if (row["review_status"] is None if review_status=="unmerged"
                                                 else row["review_status"]==review_status)]
        items=[]
        for row in selected[(page-1)*size:page*size]:
            items.append({"question_id":row["id"],"text":redact(row["normalized_question"] or row["raw_question"]),
                "raw_text":redact(row["raw_question"]),"normalized":bool(row["normalized_question"]),
                "source":row["source"],"labels":row["labels"],"occurrence_count":row["occurrence_count"],
                "review_status":row["review_status"],"classified_at":row["classified_at"].isoformat()+"Z"})
        return {"label":label,"total":len(selected),"page":page,"size":size,"pages":(len(selected)+size-1)//size,"items":items}

    @admin.get("/api/acceptance/reports")
    def reports():
        return acceptance_reports(root)

    @admin.get("/api/acceptance/service")
    async def service():
        try:
            async with httpx.AsyncClient(timeout=2,trust_env=False) as client:
                response=await client.get("http://127.0.0.1:8110/healthz")
                response.raise_for_status()
                data=response.json()
            return {"online":data.get("status")=="ok" and data.get("taxonomy_hash")==TAXONOMY_HASH,
                    "status":data.get("status"),"model_version":data.get("model_version"),
                    "taxonomy_hash":data.get("taxonomy_hash")}
        except (httpx.HTTPError,ValueError,AttributeError):
            return {"online":False,"status":"unavailable"}

    router.include_router(admin)
    return router
