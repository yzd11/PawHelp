"""Evidence signals and reproducible threshold calibration, not probabilities."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import math

from app.prompts import QUERY_UNDERSTANDING_SYSTEM_TEMPLATE

_KNOWLEDGE_CODE = Path(__file__).resolve().parent
# Everything that changes the evidence or scores the gate sees on the served path.
SCORING_SOURCES = ("chunking.py", "embedding.py", "reranking.py", "search.py", "query_understanding.py", "pipeline.py")
SCORING_SETTINGS = ("embedding_model", "embedding_max_length", "reranker_model", "reranker_query_max_length",
                    "reranker_passage_max_length", "dense_recall_limit", "bm25_recall_limit", "rerank_limit",
                    "rerank_top_k", "knowledge_rerank_threshold", "openai_model", "structured_output_method")


def evidence_signals(evidence):
    scores = sorted([float(x.get("score", 0)) for x in evidence if x.get("answer") and math.isfinite(float(x.get("score", 0)))], reverse=True)
    top1 = max(0., min(1., scores[0])) if scores else 0.
    top2 = max(0., min(1., scores[1])) if len(scores) > 1 else 0.
    # Count is diagnostic only: weak neighbours must not create a fixed bonus.
    valid = sum(s >= top1 * .5 and s > 0 for s in scores)
    weighted_count = sum(max(0., min(1., s)) for s in scores)
    margin = max(0., top1 - top2)
    return {"top1": top1, "effective_count": valid, "weighted_count": weighted_count, "margin": margin,
            "score": .6 * top1 + .25 * min(weighted_count / 3., 1.) + .15 * margin}


def _text_bytes(path):
    # A checkout with autocrlf must hash the same as the committed LF text.
    return path.read_bytes().replace(b"\r\n", b"\n")


def scoring_fingerprint(root, settings):
    """Bind a calibration to the documents, scoring code, rewrite prompt, models and limits."""
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*.md")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(_text_bytes(path))
    for name in SCORING_SOURCES:
        digest.update(name.encode())
        digest.update(_text_bytes(_KNOWLEDGE_CODE / name))
    digest.update(QUERY_UNDERSTANDING_SYSTEM_TEMPLATE.encode())
    digest.update(json.dumps({key: getattr(settings, key, None) for key in SCORING_SETTINGS},
                             sort_keys=True, default=str).encode())
    return digest.hexdigest()


def kept_entries_sha256(rows):
    """Hash the non-document entries served alongside the documents; they change at runtime."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda row: row.id):
        digest.update(json.dumps([row.id, row.source_kind, row.questions, row.answer], ensure_ascii=False).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ConfidenceDecision:
    accepted: bool
    calibrated: bool
    signals: dict
    threshold: float | None
    reason: str


class EvidenceGate:
    def __init__(self, settings, root):
        self.path = Path(settings.evidence_calibration_path)
        self.fingerprint = scoring_fingerprint(root, settings)

    def check(self, evidence, *, legacy_confident):
        signals = evidence_signals(evidence)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data["formula_version"] != 2 or data.get("purpose") != "relevance_prefilter":
                raise ValueError("校准公式版本不符")
            if data.get("scoring_fingerprint") != self.fingerprint:
                raise ValueError("校准与当前评分输入不一致（知识文档、检索代码、改写提示词、模型或检索参数已变），需重新校准")
            threshold = float(data["threshold"])
            if not 0 <= threshold <= 1 or not data["validation"]["passed"]:
                raise ValueError("阈值或留出集验证无效")
            accepted = bool(evidence) and signals["score"] >= threshold
            return ConfidenceDecision(accepted, True, signals, threshold, "已校准证据评分" if accepted else "证据评分未达到校准阈值")
        except (OSError, ValueError, KeyError, TypeError) as error:
            return ConfidenceDecision(bool(evidence) and legacy_confident, False, signals, None,
                                      "未校准：保留原闸行为；" + str(error))


def calibrate(rows, *, reranker_model, scoring_hash, dataset_hash, validation_rows=None):
    """Calibrate relevance only; completeness is a separate mandatory runtime check."""
    ordered = sorted(rows, key=lambda r: hashlib.sha256(r["case_id"].encode()).hexdigest())
    if len(ordered) < 30 or any(type(r.get("sufficient")) is not bool for r in ordered):
        raise ValueError("校准至少需要 30 条独立标注的充分/不足样例")
    if validation_rows is None:
        validation, training = ordered[::4], [r for i, r in enumerate(ordered) if i % 4]
    else:
        validation, training = list(validation_rows), ordered
        if len(validation) < 10 or any(type(r.get("sufficient")) is not bool for r in validation):
            raise ValueError("独立验证至少10条充分/不足样例")
        if {r["case_id"] for r in training}.intersection(r["case_id"] for r in validation):
            raise ValueError("训练与独立验证样例不可重叠")
    if any(not any(r["sufficient"] == label for r in part) for part in (training, validation) for label in (False, True)):
        raise ValueError("训练与留出集都必须含充分、不足样例")
    def stats(part, threshold):
        accept = [r for r in part if r["evidence"] and evidence_signals(r["evidence"])["score"] >= threshold]
        unsafe = sum(not r["sufficient"] for r in accept)
        positives = sum(r["sufficient"] for r in part)
        return {"size": len(part), "accepted": len(accept), "unsafe_accepted": unsafe,
                "unsafe_rate": unsafe / max(len(accept), 1), "useful_recall": sum(r["sufficient"] for r in accept) / max(positives, 1)}
    candidates = [(stats(training, t / 1000), t / 1000) for t in range(1001)]
    eligible = [(s, t) for s, t in candidates if s["accepted"] and s["unsafe_rate"] <= .05]
    if not eligible:
        raise ValueError("无可用阈值达到训练集安全条件；必须改进检索或标注")
    best = max((s["useful_recall"], -s["unsafe_rate"]) for s, _ in eligible)
    tied = [t for s, t in eligible if (s["useful_recall"], -s["unsafe_rate"]) == best]
    threshold = round((min(tied) + max(tied)) / 2, 3)
    held = stats(validation, threshold)
    held["passed"] = held["accepted"] > 0 and held["unsafe_rate"] <= .05 and held["useful_recall"] >= .5
    if not held["passed"]:
        raise ValueError(f"留出集验证失败，不发布阈值：{held}")
    return {"formula_version": 2, "purpose": "relevance_prefilter", "threshold": threshold,
            "training": stats(training, threshold), "training_optimal_range": [min(tied), max(tied)],
            "validation": held, "validation_method": "independent" if validation_rows is not None else "hash_split",
            "reranker_model": reranker_model, "scoring_fingerprint": scoring_hash,
            "dataset_sha256": dataset_hash, "sample_count": len(rows)}
