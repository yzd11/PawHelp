"""FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api import create_api_router
from app.config import Settings, get_settings
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.evaluation.runtime import (
    DEFAULT_GENERATION_STRATEGIES,
)
from app.knowledge.embedding import BgeM3Embedder
from app.knowledge.mining import ConversationKnowledgeExtractor
from app.knowledge.query_understanding import QueryUnderstandingService
from app.knowledge.reranking import BgeReranker
from app.knowledge.search import HybridKnowledgeSearch
from app.knowledge.vector_store import MilvusKnowledgeStore
from app.llm import create_chat_model
from app.services.knowledge_admin import KnowledgeAdminService
from app.services.tool_chat import ToolAwareChatService
from app.workflow.graph import WorkflowChatService
from app.workflow.context_logging import configure_context_logging
from app.flywheel import Flywheel
from app.evaluation.pipeline import EvaluationPipeline
from app.api.operations import operations_router, ScheduledOperations
from app.topics.api import topic_router


from app.api.pages import CHAT_PAGE_PATH, KNOWLEDGE_PAGE_PATH, RAG_EVAL_PAGE_PATH, page_router
from app.api.evaluation import (
    PROJECT_ROOT, RAG_EVAL_CASES_PATH, RAG_EVAL_REPORT_DIR, RAG_EVAL_REPORT_PATH,
    RagEvalRunner, evaluation_router,
)

KNOWLEDGE_DOCUMENTS_PATH = PROJECT_ROOT / "knowledge"

logger = logging.getLogger(__name__)


async def warm_knowledge_models(embedder: BgeM3Embedder, reranker: BgeReranker) -> None:
    """Load BGE weights in the background; a chat arriving earlier waits on the load lock."""

    started = time.perf_counter()
    try:
        await asyncio.gather(asyncio.to_thread(embedder.warm), asyncio.to_thread(reranker.warm))
    except Exception:
        logger.exception("knowledge model warm-up failed; models load on first query")
        raise
    logger.info("knowledge models ready in %.1fs", time.perf_counter() - started)


def create_app(
    settings: Settings | None = None,
    model: object | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    tool_chat_service: ToolAwareChatService | None = None,
    knowledge_admin_service: KnowledgeAdminService | None = None,
    rag_eval_runner: RagEvalRunner | None = None,
    preload_models: bool = False,
) -> FastAPI:
    """Create an initialized API application and its runtime dependencies."""

    active_settings = settings or get_settings()
    active_model = model or create_chat_model(active_settings)
    owned_engine = None
    if session_factory is None:
        owned_engine = create_async_engine_from_settings(active_settings)
        active_session_factory = create_session_factory(owned_engine)
    else:
        active_session_factory = session_factory
    embedder = BgeM3Embedder(
        active_settings.embedding_model,
        batch_size=active_settings.embedding_batch_size,
        max_length=active_settings.embedding_max_length,
    )
    vector_store = MilvusKnowledgeStore(
        active_settings.milvus_uri,
        active_settings.milvus_hybrid_collection,
    )
    reranker = BgeReranker(
        active_settings.reranker_model,
        batch_size=active_settings.reranker_batch_size,
        query_max_length=active_settings.reranker_query_max_length,
        passage_max_length=active_settings.reranker_passage_max_length,
    )
    query_planner = QueryUnderstandingService(
        active_model,  # type: ignore[arg-type]
        method=active_settings.structured_output_method,
    )
    knowledge_search = HybridKnowledgeSearch(
        active_session_factory,
        embedder,
        vector_store,
        reranker,
        query_planner,
        confidence_threshold=active_settings.knowledge_rerank_threshold,
        dense_limit=active_settings.dense_recall_limit,
        bm25_limit=active_settings.bm25_recall_limit,
        rerank_limit=active_settings.rerank_limit,
    )
    active_tool_chat = tool_chat_service or WorkflowChatService(
        settings=active_settings,
        model=active_model,  # type: ignore[arg-type]
        session_factory=active_session_factory,
        knowledge_search=knowledge_search,
    )
    active_knowledge_admin = knowledge_admin_service or KnowledgeAdminService(
        active_session_factory,
        embedder,
        vector_store,
        knowledge_search,
        documents_root=KNOWLEDGE_DOCUMENTS_PATH,
        vectorize_batch_size=active_settings.vectorize_batch_size,
        conversation_extractor=ConversationKnowledgeExtractor(
            active_model,  # type: ignore[arg-type]
            method=active_settings.structured_output_method,
        ),
        conversation_batch_size=active_settings.conversation_mining_batch_size,
    )
    if rag_eval_runner is None:
        async def default_rag_eval_runner(full_generation: bool) -> dict[str, object]:
            from app.evaluation.runtime import run_retrieval_evaluation

            report, _ = await run_retrieval_evaluation(
                active_settings,
                RAG_EVAL_CASES_PATH,
                RAG_EVAL_REPORT_DIR,
                generation_strategies=(
                    None if full_generation else DEFAULT_GENERATION_STRATEGIES
                ),
            )
            return report

        active_rag_eval_runner = default_rag_eval_runner
    else:
        active_rag_eval_runner = rag_eval_runner
    rag_eval_lock = asyncio.Lock()
    # Retry policy must be set before SDK clients are constructed, not mutated afterwards.
    flywheel_model = model or create_chat_model(active_settings, max_retries=0)
    flywheel = Flywheel(active_settings, active_session_factory, flywheel_model, active_knowledge_admin)
    pipeline = EvaluationPipeline(active_settings, active_session_factory, PROJECT_ROOT, rag_eval_lock)
    scheduled = ScheduledOperations(active_settings, active_session_factory, flywheel, pipeline)

    warmup: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        nonlocal warmup
        if preload_models:
            warmup = asyncio.create_task(warm_knowledge_models(embedder, reranker))
        try:
            scheduled.start()
            if isinstance(active_tool_chat, WorkflowChatService):
                configure_context_logging(PROJECT_ROOT)
                active_tool_chat._memory.startup_check()
                checkpoint_dir = PROJECT_ROOT / ".runtime"
                checkpoint_dir.mkdir(exist_ok=True)
                async with AsyncSqliteSaver.from_conn_string(
                    str(checkpoint_dir / "conversation-checkpoints.sqlite3")
                ) as checkpointer:
                    active_tool_chat.set_checkpointer(checkpointer)
                    try:
                        yield
                    finally:
                        await active_tool_chat.close()
            else:
                yield
        finally:
            if warmup is not None:
                warmup.cancel()
            await scheduled.close()
            if owned_engine is not None:
                await owned_engine.dispose()

    app = FastAPI(title="PawHelp", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=CHAT_PAGE_PATH.parent), name="static")
    app.state.settings = active_settings
    app.state.model = active_model
    app.state.engine = owned_engine
    app.state.session_factory = active_session_factory
    app.state.tool_chat_service = active_tool_chat
    app.state.knowledge_admin_service = active_knowledge_admin
    app.state.flywheel = flywheel
    app.state.evaluation_pipeline = pipeline
    app.include_router(operations_router(active_settings, active_session_factory, flywheel, pipeline))
    app.include_router(topic_router(active_session_factory, PROJECT_ROOT))
    app.include_router(
        create_api_router(
            active_settings,
            active_model,
            active_session_factory,
            active_tool_chat,
            active_knowledge_admin,
        )
    )

    app.include_router(page_router())
    app.include_router(evaluation_router(active_session_factory, pipeline, rag_eval_lock, active_rag_eval_runner))

    @app.get("/health")
    async def health() -> dict[str, str]:
        if warmup is None:
            return {"status": "ok"}
        if not warmup.done():
            return {"status": "ok", "knowledge_models": "loading"}
        failed = warmup.cancelled() or warmup.exception() is not None
        return {"status": "ok", "knowledge_models": "failed" if failed else "ready"}

    return app


# Module-level ASGI entry point used by ``uvicorn app.main:app``; tests build their own.
app = create_app(preload_models=True)
