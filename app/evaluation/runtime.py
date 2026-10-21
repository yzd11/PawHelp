"""Runtime composition shared by the evaluation CLI and local web console."""

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from langchain_core.messages.utils import count_tokens_approximately

from sqlalchemy import func, select

from app.config import Settings
from app.context import bound_text
from app.knowledge.confidence import EvidenceGate
from app.knowledge.jev_evidence import build_evidence_checker
from app.knowledge.pipeline import KnowledgeEvidencePipeline, QuestionSplitter
from app.database.models import KnowledgeChunk
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.evaluation.corpus import (
    EVALUATION_SOURCES,
    evaluation_corpus_sha256,
    validate_evaluation_cases,
    validate_evaluation_corpus,
)
from app.evaluation.retrieval_quality import (
    CachedQueryPlanner,
    EvalCase,
    RetrievalQualityEvaluator,
    load_cases,
    write_report,
)
from app.knowledge.embedding import BgeM3Embedder
from app.knowledge.query_understanding import (
    IdentityQueryUnderstanding,
    QueryUnderstandingService,
)
from app.knowledge.reranking import BgeReranker
from app.knowledge.search import HybridKnowledgeSearch, RetrievalStrategy
from app.knowledge.vector_store import MilvusKnowledgeStore
from app.llm import create_chat_model


DEFAULT_EVAL_CASES_PATH = Path("evals/customer_service_500.csv")
DEFAULT_GENERATION_STRATEGIES = (RetrievalStrategy.HYBRID_RERANK,)


def report_dataset_mismatch_count(
    report: dict[str, object], cases: Sequence[EvalCase]
) -> int:
    """Count saved case labels changed since a report was generated."""

    current = {case.case_id: case for case in cases}
    saved = {
        str(row.get("case_id")): row
        for row in report.get("cases", [])
        if isinstance(row, dict) and row.get("case_id")
    }
    return sum(
        case_id not in current
        or row.get("query") != current[case_id].query
        or row.get("expected_points") != current[case_id].expected_points
        or row.get("expected_answerable") != current[case_id].expected_answerable
        for case_id, row in saved.items()
    )


def report_dataset_alignment(
    report: dict[str, object], cases_path: Path
) -> dict[str, object]:
    """Annotate an in-memory report without changing the saved historical file."""

    current_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    run = report.get("run")
    saved_hash = run.get("cases_sha256") if isinstance(run, dict) else None
    mismatches = report_dataset_mismatch_count(report, load_cases(cases_path))
    if saved_hash:
        status = "current" if saved_hash == current_hash else "stale"
    else:
        status = "stale" if mismatches else "unverified"
    if cases_path.resolve() == DEFAULT_EVAL_CASES_PATH.resolve() and isinstance(run, dict):
        saved_corpus_hash = run.get("corpus_sha256")
        if saved_corpus_hash and saved_corpus_hash != evaluation_corpus_sha256():
            status = "stale"
        elif not saved_corpus_hash and status == "current":
            status = "unverified"
    return {"status": status, "changed_cases": mismatches}


def generation_strategy_names(
    generate_answers: bool,
    generation_strategies: Sequence[RetrievalStrategy] | None,
    strategies: Sequence[RetrievalStrategy],
) -> list[str]:
    """Record concrete generated routes so report consumers can compare them."""

    if not generate_answers:
        return []
    selected = generation_strategies if generation_strategies is not None else strategies
    return [item.value for item in selected if item in strategies]


async def run_retrieval_evaluation(
    settings: Settings,
    cases_path: Path,
    output_dir: Path,
    *,
    max_cases: int | None = None,
    sample_per_bucket: int | None = None,
    strategies: Sequence[RetrievalStrategy] | None = None,
    generate_answers: bool = True,
    generation_strategies: Sequence[RetrievalStrategy] | None = None,
    model_callbacks: list | None = None,
    normalize_queries: bool = True,
) -> tuple[dict[str, object], tuple[Path, Path]]:
    """Run the complete four-strategy evaluation and persist both reports."""

    cases = load_cases(cases_path)
    engine = create_async_engine_from_settings(settings)
    factory = create_session_factory(engine)
    benchmark_corpus_ids: list[int] | None = None
    if cases_path.resolve() == DEFAULT_EVAL_CASES_PATH.resolve():
        try:
            validate_evaluation_cases(cases)
            async with factory() as session:
                knowledge_rows = list(await session.scalars(select(KnowledgeChunk)))
            benchmark_corpus_ids = validate_evaluation_corpus(knowledge_rows)
        except Exception:
            await engine.dispose()
            raise
    model = create_chat_model(settings, max_retries=0) if model_callbacks is not None else create_chat_model(settings)
    if model_callbacks is not None:
        model.callbacks = model_callbacks
    planner = CachedQueryPlanner(
        QueryUnderstandingService(model, method=settings.structured_output_method)
    )
    if not normalize_queries:
        planner = IdentityQueryUnderstanding()
    search = HybridKnowledgeSearch(
        factory,
        BgeM3Embedder(
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
        ),
        MilvusKnowledgeStore(settings.milvus_uri, settings.milvus_hybrid_collection),
        BgeReranker(
            settings.reranker_model,
            batch_size=settings.reranker_batch_size,
            query_max_length=settings.reranker_query_max_length,
            passage_max_length=settings.reranker_passage_max_length,
        ),
        planner,
        confidence_threshold=settings.knowledge_rerank_threshold,
        dense_limit=settings.dense_recall_limit,
        bm25_limit=settings.bm25_recall_limit,
        rerank_limit=settings.rerank_limit,
    )
    pipeline = KnowledgeEvidencePipeline(
        search, EvidenceGate(settings, Path(__file__).resolve().parents[2] / "knowledge"),
        build_evidence_checker(settings, model),
        QuestionSplitter(model, method=settings.structured_output_method), top_k=settings.rerank_top_k,
        bound_text=lambda text: bound_text(text, settings.evidence_chunk_max_tokens,
            lambda messages: int(count_tokens_approximately(messages, chars_per_token=settings.context_chars_per_token))),
    ) if generate_answers else None
    evaluator = RetrievalQualityEvaluator(
        search=search,
        model=model,
        session_factory=factory,
        structured_output_method=settings.structured_output_method,
        judge_model=settings.openai_model,
        top_k=settings.rerank_limit,
        evaluation_source_paths=tuple(EVALUATION_SOURCES) if benchmark_corpus_ids is not None else None,
        evidence_pipeline=pipeline,
    )
    cases_sha256 = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    if sample_per_bucket is not None:
        counts: dict[str, int] = {}
        sampled = []
        for case in cases:
            used = counts.get(case.query_type, 0)
            if used < sample_per_bucket:
                sampled.append(case)
                counts[case.query_type] = used + 1
        cases = sampled
    if max_cases is not None:
        cases = cases[:max_cases]
    try:
        report = await evaluator.run(
            cases,
            strategies=strategies or tuple(RetrievalStrategy),
            generate_answers=generate_answers,
            generation_strategies=generation_strategies,
        )
        if benchmark_corpus_ids is not None:
            chunk_count = len(benchmark_corpus_ids)
        else:
            async with factory() as session:
                chunk_count = int(
                    await session.scalar(
                        select(func.count(KnowledgeChunk.id)).where(
                            KnowledgeChunk.vectorize_status == "done"
                        )
                    )
                    or 0
                )
        report["run"] = {
            "cases_file": cases_path.name,
            "cases_sha256": cases_sha256,
            "corpus_sha256": evaluation_corpus_sha256() if benchmark_corpus_ids is not None else None,
            "corpus_sources": list(EVALUATION_SOURCES) if benchmark_corpus_ids is not None else None,
            "knowledge_chunks": chunk_count,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "judge_model": settings.openai_model if generate_answers else None,
            "generation_strategies": generation_strategy_names(
                generate_answers,
                generation_strategies,
                strategies or tuple(RetrievalStrategy),
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generation_enabled": generate_answers,
            "normalize_queries": normalize_queries,
            "evidence_gate": {
                "formula_version": 2, "purpose": "relevance_prefilter",
                "complete_check": generate_answers, "top_k": settings.rerank_top_k,
                "chunk_max_tokens": settings.evidence_chunk_max_tokens,
                "calibration_sha256": hashlib.sha256(Path(settings.evidence_calibration_path).read_bytes()).hexdigest()
                    if Path(settings.evidence_calibration_path).is_file() else None,
                "judge": settings.evidence_judge,
                "jev_model": settings.jev_model if settings.evidence_judge != "llm" else None,
                "jev_calibration_sha256": hashlib.sha256(Path(settings.jev_calibration_path).read_bytes()).hexdigest()
                    if settings.evidence_judge != "llm" and Path(settings.jev_calibration_path).is_file() else None,
            },
            "retrieval_configuration": {key: getattr(settings, key) for key in (
                "dense_recall_limit", "bm25_recall_limit", "rerank_limit", "knowledge_rerank_threshold",
                "embedding_max_length", "reranker_query_max_length", "reranker_passage_max_length",
                "structured_output_method", "model_max_output_tokens")},
            "generation_failed": sum(
                row.get("generation_status") == "failed"
                for row in report.get("cases", [])
                if isinstance(row, dict)
            ),
            "coverage_failed": sum(
                row.get("coverage_status") == "failed"
                for row in report.get("cases", [])
                if isinstance(row, dict)
            ),
            "faithfulness_failed": sum(
                row.get("faithfulness_status") == "failed"
                for row in report.get("cases", [])
                if isinstance(row, dict)
            ),
            "ledger_failed": sum(
                bool(row.get("ledger_error"))
                for row in report.get("cases", [])
                if isinstance(row, dict)
            ),
        }
        return report, write_report(report, output_dir)
    finally:
        await engine.dispose()
