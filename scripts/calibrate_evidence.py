"""Collect the evidence the served gate scores, then publish a labelled calibration."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from app.config import get_settings
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.evaluation.runtime import DEFAULT_EVAL_CASES_PATH
from app.evaluation.retrieval_quality import load_cases
from app.knowledge.confidence import EvidenceGate, calibrate, kept_entries_sha256, scoring_fingerprint
from app.knowledge.embedding import BgeM3Embedder
from app.knowledge.pipeline import KnowledgeEvidencePipeline, QuestionSplitter
from app.knowledge.query_understanding import QueryUnderstandingService
from app.knowledge.reranking import BgeReranker
from app.knowledge.search import HybridKnowledgeSearch
from app.knowledge.vector_store import MilvusKnowledgeStore
from app.llm import create_chat_model
from app.repositories.knowledge import KnowledgeRepository

KNOWLEDGE_ROOT = Path("knowledge")
SERVED_COLLECTION = ("served path: model query rewrite + question split + whole knowledge base, knowledge intent; "
                     "policy-intent extra queries and content filter are not covered")


async def collect(output, case_ids):
    """Paid but bounded: one rewrite per query and one split per compound question, no knowledge writes."""
    settings = get_settings()
    engine = create_async_engine_from_settings(settings)
    factory = create_session_factory(engine)
    model = create_chat_model(settings)
    search = HybridKnowledgeSearch(
        factory,
        BgeM3Embedder(settings.embedding_model, batch_size=settings.embedding_batch_size,
                      max_length=settings.embedding_max_length),
        MilvusKnowledgeStore(settings.milvus_uri, settings.milvus_hybrid_collection),
        BgeReranker(settings.reranker_model, batch_size=settings.reranker_batch_size,
                    query_max_length=settings.reranker_query_max_length,
                    passage_max_length=settings.reranker_passage_max_length),
        QueryUnderstandingService(model, method=settings.structured_output_method),
        confidence_threshold=settings.knowledge_rerank_threshold, dense_limit=settings.dense_recall_limit,
        bm25_limit=settings.bm25_recall_limit, rerank_limit=settings.rerank_limit,
    )
    pipeline = KnowledgeEvidencePipeline(
        search, EvidenceGate(settings, KNOWLEDGE_ROOT), None,
        QuestionSplitter(model, method=settings.structured_output_method), top_k=settings.rerank_top_k)
    cases = [case for case in load_cases(DEFAULT_EVAL_CASES_PATH) if case_ids is None or case.case_id in case_ids]
    limit = asyncio.Semaphore(4)

    async def served(case):
        async with limit:
            _, matches, queries = await pipeline.gather(case.query)
        return {"case_id": case.case_id, "question": case.query, "queries": queries,
                "evidence": pipeline.evidence(matches), "expected_points": case.expected_points,
                "expected_answerable": case.expected_answerable, "sufficient": None}

    try:
        rows = await asyncio.gather(*(served(case) for case in cases))
        async with factory() as session:
            kept = await KnowledgeRepository(session).list_kept_entries()
    finally:
        await engine.dispose()
    payload = {"reranker_model": settings.reranker_model,
               "scoring_fingerprint": scoring_fingerprint(KNOWLEDGE_ROOT, settings),
               "dataset_sha256": hashlib.sha256(DEFAULT_EVAL_CASES_PATH.read_bytes()).hexdigest(),
               "kept_entries_sha256": kept_entries_sha256(kept), "kept_entries_count": len(kept),
               "collection_mode": SERVED_COLLECTION, "rows": rows}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evidence-labels.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"collected {len(rows)} served-path cases -> {path}; sufficient=null until reviewed")


def case_ids_from(paths):
    if not paths:
        return None
    return {row["case_id"] for path in paths for row in json.loads(path.read_text(encoding="utf-8"))["rows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--collect", action="store_true", help="served-path evidence; paid rewrite/split calls, no knowledge writes")
    actions.add_argument("--labels", type=Path, help="JSON with manually reviewed sufficient true/false labels, not suggested labels")
    parser.add_argument("--cases-from", type=Path, nargs="*", help="collect only the case IDs in these label files")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/confidence-calibration"))
    parser.add_argument("--validation-labels", type=Path, help="Frozen independent relevance-prefilter labels; never used to select threshold")
    parser.add_argument("--publish", type=Path, default=Path("config/evidence-calibration.json"))
    args = parser.parse_args()
    if args.collect:
        asyncio.run(collect(args.output_dir, case_ids_from(args.cases_from)))
        return
    data = json.loads(args.labels.read_text(encoding="utf-8"))
    rows = [row for row in data["rows"] if type(row.get("sufficient")) is bool]
    if data.get("label_scope") != "relevance_prefilter":
        raise ValueError("v2必须使用相关性粗筛标签；不得用完整性标签替代")
    settings = get_settings()
    if data["scoring_fingerprint"] != scoring_fingerprint(KNOWLEDGE_ROOT, settings):
        raise ValueError("标注时的评分输入（知识文档、检索代码、模型或检索参数）已变，禁止发布阈值")
    validation = None
    if args.validation_labels:
        validation_data = json.loads(args.validation_labels.read_text(encoding="utf-8"))
        for key in ("label_scope", "reranker_model", "scoring_fingerprint", "dataset_sha256", "collection_mode"):
            if validation_data.get(key) != data.get(key):
                raise ValueError(f"独立验证版本/标签用途不一致: {key}")
        validation = validation_data["rows"]
    artifact = calibrate(rows, reranker_model=data["reranker_model"], scoring_hash=data["scoring_fingerprint"],
                         dataset_hash=data["dataset_sha256"], validation_rows=validation)
    for key in ("collection_mode", "reviewer", "scope_limit", "kept_entries_sha256", "kept_entries_count"):
        artifact[key] = data.get(key)
    artifact["training_labels_sha256"] = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    artifact["validation_labels_sha256"] = hashlib.sha256(args.validation_labels.read_bytes()).hexdigest() if args.validation_labels else None
    args.publish.parent.mkdir(parents=True, exist_ok=True)
    args.publish.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"published {args.publish}; validation={artifact['validation']}")


if __name__ == "__main__": main()
