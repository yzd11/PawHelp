"""Collect real retrieval scores without paid calls, then publish a labelled calibration."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from app.config import get_settings
from app.evaluation.runtime import DEFAULT_EVAL_CASES_PATH, run_retrieval_evaluation
from app.knowledge.search import RetrievalStrategy
from app.knowledge.confidence import calibrate, corpus_fingerprint


async def collect(output):
    settings = get_settings()
    report, paths = await run_retrieval_evaluation(settings, DEFAULT_EVAL_CASES_PATH, output,
        strategies=[RetrievalStrategy.HYBRID_RERANK], generate_answers=False, normalize_queries=False)
    candidates = []
    for row in report["cases"]:
        evidence = row["citations"][:settings.rerank_top_k]
        retrieved = {item["chunk_id"] for item in evidence}
        groups = row.get("relevant_chunk_groups") or [[item] for item in row["relevant_chunk_ids"]]
        gt_supported = row["expected_answerable"] and bool(groups) and all(set(group).intersection(retrieved) for group in groups)
        candidates.append({"case_id": row["case_id"], "question": row["query"], "evidence": evidence,
                           "suggested_sufficient": bool(gt_supported), "sufficient": None,
                           "label_basis": "expected section groups (not retriever-generated GT)", "expected_points": row.get("expected_points", [])})
    payload = {"reranker_model": settings.reranker_model, "corpus_fingerprint": corpus_fingerprint(Path("knowledge")),
               "dataset_sha256": hashlib.sha256(DEFAULT_EVAL_CASES_PATH.read_bytes()).hexdigest(),
               "collection_mode": "raw query, local retrieval only; serving rewrite distribution must be audited",
               "rows": candidates}
    path = output / "evidence-labels.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"scored report={paths[0]} labels={path}; sufficient=null until human verification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--collect", action="store_true", help="300 local retrieval cases, no provider calls, no knowledge writes")
    actions.add_argument("--labels", type=Path, help="JSON with manually reviewed sufficient true/false labels, not suggested labels")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/confidence-calibration"))
    parser.add_argument("--validation-labels", type=Path, help="Frozen independent relevance-prefilter labels; never used to select threshold")
    parser.add_argument("--publish", type=Path, default=Path("config/evidence-calibration.json"))
    args = parser.parse_args()
    if args.collect:
        asyncio.run(collect(args.output_dir))
    else:
        data = json.loads(args.labels.read_text(encoding="utf-8"))
        rows = [row for row in data["rows"] if type(row.get("sufficient")) is bool]
        if data.get("label_scope") != "relevance_prefilter":
            raise ValueError("v2必须使用相关性粗筛标签；不得用完整性标签替代")
        if data["corpus_fingerprint"] != corpus_fingerprint(Path("knowledge")):
            raise ValueError("标注的知识文档版本已过期，禁止发布阈值")
        settings = get_settings()
        if data["reranker_model"] != settings.reranker_model:
            raise ValueError("重排模型版本不一致，禁止发布阈值")
        validation = None
        if args.validation_labels:
            validation_data = json.loads(args.validation_labels.read_text(encoding="utf-8"))
            for key in ("label_scope", "reranker_model", "corpus_fingerprint", "dataset_sha256"):
                if validation_data.get(key) != data.get(key):
                    raise ValueError(f"独立验证版本/标签用途不一致: {key}")
            validation = validation_data["rows"]
        artifact = calibrate(rows, reranker_model=data["reranker_model"], corpus_hash=data["corpus_fingerprint"], dataset_hash=data["dataset_sha256"], validation_rows=validation)
        artifact["collection_mode"] = data.get("collection_mode")
        artifact["reviewer"] = data.get("reviewer")
        artifact["scope_limit"] = data.get("scope_limit")
        artifact["training_labels_sha256"] = hashlib.sha256(args.labels.read_bytes()).hexdigest()
        artifact["validation_labels_sha256"] = hashlib.sha256(args.validation_labels.read_bytes()).hexdigest() if args.validation_labels else None
        args.publish.parent.mkdir(parents=True, exist_ok=True)
        args.publish.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"published {args.publish}; validation={artifact['validation']}")


if __name__ == "__main__": main()
