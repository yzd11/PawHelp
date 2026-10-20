"""Evaluate the write-time consistency check against labelled entries (paid; not run in CI).

Each case is checked against the live knowledge base with local retrieval and one model
call. Labels assume the documents in knowledge/ plus no added entry that repeats a case;
re-review a label when the knowledge base changes rather than editing it to match a run.
"""

import argparse
import asyncio
import json
from pathlib import Path
import re

from sqlalchemy import select

from app.config import get_settings
from app.database.models import KnowledgeChunk
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.knowledge.consistency import KnowledgeConsistencyChecker
from app.knowledge.embedding import BgeM3Embedder
from app.knowledge.query_understanding import IdentityQueryUnderstanding
from app.knowledge.reranking import BgeReranker
from app.knowledge.search import HybridKnowledgeSearch
from app.knowledge.vector_store import MilvusKnowledgeStore
from app.llm import create_chat_model

CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "knowledge_consistency_cases.jsonl"
POSITIONAL = re.compile(r"(?i)entry \d|第\s*\d+\s*条|条目\s*\d")


def reason_is_usable(result, chunk_ids) -> bool:
    """A reason the operator can act on: Chinese, citing only real chunk IDs, naming the blocking entry."""
    cited = {int(number) for number in re.findall(r"#(\d+)", result.reason)}
    return (bool(re.search(r"[一-鿿]", result.reason)) and cited <= chunk_ids
            and not POSITIONAL.search(result.reason) and (result.verdict == "new" or result.chunk_id is not None))


async def run(repeats: int) -> bool:
    settings = get_settings()
    engine = create_async_engine_from_settings(settings)
    factory = create_session_factory(engine)
    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    search = HybridKnowledgeSearch(
        factory,
        BgeM3Embedder(settings.embedding_model, batch_size=settings.embedding_batch_size,
                      max_length=settings.embedding_max_length),
        MilvusKnowledgeStore(settings.milvus_uri, settings.milvus_hybrid_collection),
        BgeReranker(settings.reranker_model, batch_size=settings.reranker_batch_size,
                    query_max_length=settings.reranker_query_max_length,
                    passage_max_length=settings.reranker_passage_max_length),
        IdentityQueryUnderstanding(), confidence_threshold=settings.knowledge_rerank_threshold,
        dense_limit=settings.dense_recall_limit, bm25_limit=settings.bm25_recall_limit,
        rerank_limit=settings.rerank_limit,
    )
    checker = KnowledgeConsistencyChecker(search, create_chat_model(settings), method=settings.structured_output_method)
    try:
        async with factory() as session:
            chunk_ids = set(await session.scalars(select(KnowledgeChunk.id)))
        wrong = unusable = 0
        for case in cases:
            for _ in range(repeats):
                result = await checker.check(case["question"], case["answer"])
                blocked = result.verdict in ("duplicate", "conflict")
                verdict_ok = result.verdict != "unavailable" and blocked == (case["expect"] == "block")
                reason_ok = reason_is_usable(result, chunk_ids)
                wrong += not verdict_ok
                unusable += not reason_ok
                print(json.dumps({"id": case["id"], "expect": case["expect"], "verdict": result.verdict,
                                  "passed": verdict_ok and reason_ok, "message": result.message}, ensure_ascii=False))
    finally:
        await engine.dispose()
    total = len(cases) * repeats
    print(json.dumps({"runs": total, "wrong_verdicts": wrong, "unusable_reasons": unusable,
                      "all_passed": not wrong and not unusable}, ensure_ascii=False))
    return not wrong and not unusable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2, help="runs per case; the model is not deterministic")
    raise SystemExit(0 if asyncio.run(run(parser.parse_args().repeats)) else 1)
