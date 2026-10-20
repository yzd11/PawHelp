"""Knowledge administration HTTP interface; business operations live in services."""
from typing import Annotated, Literal
from fastapi import APIRouter, HTTPException, Query
from pydantic import PositiveInt
from sqlalchemy.exc import SQLAlchemyError
from app.schemas import (
    KnowledgeChunkPage,
    KnowledgeChunkView,
    KnowledgeDeleteResult,
    KnowledgeDraftView,
    KnowledgeMaterial,
    KnowledgeMiningRequest,
    KnowledgeMiningResult,
    KnowledgeMiningStats,
    KnowledgeOverview,
    KnowledgePreviewRequest,
    KnowledgeSearchRequest,
    KnowledgeSearchMatch,
    KnowledgeSearchResult,
    KnowledgeSyncResult,
    KnowledgeVectorizeResult,
    ManualKnowledgeRequest,
    MarkdownKnowledgeIngestResult,
    MarkdownKnowledgePreviewResult,
    MarkdownKnowledgeRequest,
)
from app.services.knowledge_admin import (
    KnowledgeAdminService, KnowledgeCheckUnavailable, KnowledgeChunkNotFound, KnowledgeConflictError,
    KnowledgeDuplicateError, KnowledgeMaterialNotFound,
)

SAFE_KNOWLEDGE_ERROR = {"code": "knowledge_unavailable", "message": "知识库服务暂时不可用，请稍后重试"}


def knowledge_router(knowledge_admin_service: KnowledgeAdminService) -> APIRouter:
    router = APIRouter(tags=["knowledge"])

    @router.get(
        "/v1/knowledge/materials",
        response_model=list[KnowledgeMaterial],
    )
    async def knowledge_materials() -> list[KnowledgeMaterial]:
        return knowledge_admin_service.list_materials()

    @router.get(
        "/v1/knowledge/overview",
        response_model=KnowledgeOverview,
    )
    async def knowledge_overview() -> KnowledgeOverview:
        try:
            return await knowledge_admin_service.overview()
        except SQLAlchemyError as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.post(
        "/v1/knowledge/preview",
        response_model=list[KnowledgeDraftView],
    )
    async def preview_knowledge(
        request: KnowledgePreviewRequest,
    ) -> list[KnowledgeDraftView]:
        try:
            return [
                KnowledgeDraftView.model_validate(row)
                for row in knowledge_admin_service.preview(request.path)
            ]
        except KnowledgeMaterialNotFound as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "knowledge_material_not_found"},
            ) from error

    @router.post(
        "/v1/knowledge/preview-markdown",
        response_model=MarkdownKnowledgePreviewResult,
    )
    async def preview_markdown(
        request: MarkdownKnowledgeRequest,
    ) -> MarkdownKnowledgePreviewResult:
        try:
            return await knowledge_admin_service.preview_markdown(request)
        except Exception as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.post(
        "/v1/knowledge/ingest-markdown",
        response_model=MarkdownKnowledgeIngestResult,
    )
    async def ingest_markdown(
        request: MarkdownKnowledgeRequest,
    ) -> MarkdownKnowledgeIngestResult:
        try:
            return await knowledge_admin_service.ingest_markdown(request)
        except KnowledgeCheckUnavailable as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "knowledge_check_unavailable", "message": str(error)},
            ) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.get(
        "/v1/knowledge/mining/stats",
        response_model=KnowledgeMiningStats,
    )
    async def knowledge_mining_stats() -> KnowledgeMiningStats:
        try:
            return await knowledge_admin_service.mining_stats()
        except SQLAlchemyError as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.post(
        "/v1/knowledge/mining/run",
        response_model=KnowledgeMiningResult,
    )
    async def run_knowledge_mining(
        request: KnowledgeMiningRequest,
    ) -> KnowledgeMiningResult:
        try:
            return await knowledge_admin_service.mine_conversations(
                vectorize_after=request.vectorize_after
            )
        except KnowledgeCheckUnavailable as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "knowledge_check_unavailable", "message": str(error)},
            ) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.get(
        "/v1/knowledge/chunks",
        response_model=KnowledgeChunkPage,
    )
    async def knowledge_chunks(
        status: Literal["pending", "done"] | None = None,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> KnowledgeChunkPage:
        try:
            rows, counts = await knowledge_admin_service.list_chunks(
                status=status,
                page=page,
                page_size=page_size,
            )
            all_total = sum(counts.values())
            total = all_total if status is None else counts.get(status, 0)
            return KnowledgeChunkPage(
                items=[KnowledgeChunkView.model_validate(row) for row in rows],
                page=page,
                page_size=page_size,
                total=total,
                total_pages=max(1, (total + page_size - 1) // page_size),
                all_total=all_total,
                pending_total=counts.get("pending", 0),
                done_total=counts.get("done", 0),
            )
        except SQLAlchemyError as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.get(
        "/v1/knowledge/chunks/{chunk_id}",
        response_model=KnowledgeChunkView,
    )
    async def knowledge_chunk(chunk_id: PositiveInt) -> KnowledgeChunkView:
        try:
            row = await knowledge_admin_service.get_chunk(int(chunk_id))
            return KnowledgeChunkView.model_validate(row)
        except KnowledgeChunkNotFound as error:
            raise HTTPException(status_code=404, detail={"code": "knowledge_not_found"}) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.post(
        "/v1/knowledge/chunks",
        response_model=KnowledgeChunkView,
        status_code=201,
    )
    async def create_knowledge(
        request: ManualKnowledgeRequest,
    ) -> KnowledgeChunkView:
        try:
            row = await knowledge_admin_service.create_manual(request)
            return KnowledgeChunkView.model_validate(row)
        except KnowledgeDuplicateError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "knowledge_duplicate", "message": str(error)},
            ) from error
        except KnowledgeConflictError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "knowledge_conflict", "message": str(error)},
            ) from error
        except KnowledgeCheckUnavailable as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "knowledge_check_unavailable", "message": str(error)},
            ) from error
        except SQLAlchemyError as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.put(
        "/v1/knowledge/chunks/{chunk_id}",
        response_model=KnowledgeChunkView,
    )
    async def update_knowledge(
        chunk_id: PositiveInt,
        request: ManualKnowledgeRequest,
    ) -> KnowledgeChunkView:
        try:
            row = await knowledge_admin_service.update_chunk(int(chunk_id), request)
            return KnowledgeChunkView.model_validate(row)
        except KnowledgeDuplicateError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "knowledge_duplicate", "message": str(error)},
            ) from error
        except KnowledgeConflictError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "knowledge_conflict", "message": str(error)},
            ) from error
        except KnowledgeCheckUnavailable as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "knowledge_check_unavailable", "message": str(error)},
            ) from error
        except KnowledgeChunkNotFound as error:
            raise HTTPException(
                status_code=404, detail={"code": "knowledge_not_found"}
            ) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.delete(
        "/v1/knowledge/chunks/{chunk_id}",
        response_model=KnowledgeDeleteResult,
    )
    async def delete_knowledge(chunk_id: PositiveInt) -> KnowledgeDeleteResult:
        try:
            await knowledge_admin_service.delete_chunk(int(chunk_id))
            return KnowledgeDeleteResult(deleted=int(chunk_id))
        except Exception as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.post(
        "/v1/knowledge/sync-materials",
        response_model=KnowledgeSyncResult,
    )
    async def sync_knowledge_materials() -> KnowledgeSyncResult:
        try:
            return await knowledge_admin_service.sync_materials()
        except Exception as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.post(
        "/v1/knowledge/vectorize-pending",
        response_model=KnowledgeVectorizeResult,
    )
    async def vectorize_pending_knowledge() -> KnowledgeVectorizeResult:
        try:
            completed = await knowledge_admin_service.vectorize_pending()
            return KnowledgeVectorizeResult(vectorized=completed)
        except Exception as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    @router.post(
        "/v1/knowledge/search",
        response_model=KnowledgeSearchResult,
    )
    async def search_knowledge(
        request: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResult:
        try:
            result = await knowledge_admin_service.search(
                request.query,
                strategy=request.strategy,
                limit=request.top_k,
                categories=request.categories,
                content_types=list(request.content_types),
            )
            return KnowledgeSearchResult(
                matches=[
                    KnowledgeSearchMatch(
                        **KnowledgeChunkView.model_validate(match.chunk).model_dump(),
                        score=match.score,
                    )
                    for match in result.matches
                ],
                strategy=result.strategy.value,
                confident=result.confident,
                normalized_query=result.query_plan.normalized_query,
                synonyms=list(result.query_plan.synonyms),
            )
        except Exception as error:
            raise HTTPException(status_code=503, detail=SAFE_KNOWLEDGE_ERROR) from error

    return router
