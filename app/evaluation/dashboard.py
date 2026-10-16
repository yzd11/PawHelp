"""Read-only dashboard projections. No retrieval, model calls or publication."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from app.knowledge.confidence import EvidenceGate


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _label_snapshot(directory, digest, artifact):
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("校准文件未绑定标注快照 SHA256，不能重现扫描")
    for path in sorted(directory.glob("*.json")):
        # Only fixed local report files; never accept a caller-supplied path.
        if path.is_symlink() or path.stat().st_size > 10_000_000:
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            continue
        data = json.loads(raw)
        if any(data.get(key) != artifact.get(key) for key in (
            "reranker_model", "corpus_fingerprint", "dataset_sha256",
        )):
            raise ValueError("标注快照的模型 / 知识文档 / 数据集与校准记录不一致")
        rows = data["rows"]
        if not isinstance(rows, list) or not rows or any(type(r.get("sufficient")) is not bool for r in rows):
            raise ValueError("标注快照缺少独立布尔标签")
        if data.get("label_scope") != "relevance_prefilter":
            raise ValueError("标注不是相关性标签，不能混入相关性阈值扫描")
        return rows
    raise ValueError("找不到校准记录绑定的标注快照；不会用其他评估结果替代")


def _score_rows(rows):
    # Replay the frozen v2 artifact explicitly, independent of the serving gate
    # version. A historical report must not silently acquire a future formula.
    scored = []
    for row in rows:
        scores = sorted((max(0., min(1., float(chunk.get("score", 0))))
            for chunk in row["evidence"] if chunk.get("answer") and math.isfinite(float(chunk.get("score", 0)))), reverse=True)
        top1, top2 = (scores[0] if scores else 0.), (scores[1] if len(scores) > 1 else 0.)
        score = .6 * top1 + .25 * min(sum(scores) / 3., 1.) + .15 * max(0., top1 - top2)
        scored.append((score, row["sufficient"], bool(row["evidence"])))
    return scored


def _recommend(training, validation):
    # Same frozen v2 selection policy as the published artifact. The validation
    # set verifies a selection; it never influences selection or the tie break.
    if len(training) < 30 or len(validation) < 10 or any(
        not any(label == expected for _, label, _ in part)
        for part in (training, validation) for expected in (True, False)
    ):
        raise ValueError("训练至少30条、独立验证至少10条，且都需包含两类标签")
    candidates = [_rates(training, t / 1000) for t in range(1001)]
    eligible = [r for r in candidates if r["accepted"] and r["unsafe_share"] <= .05]
    if not eligible:
        raise ValueError("训练集没有满足放行中不相关占比 ≤ 5% 的候选阈值")
    best = max((r["relevant_pass_rate"], -r["unsafe_share"]) for r in eligible)
    tied = [r["threshold"] for r in eligible if (r["relevant_pass_rate"], -r["unsafe_share"]) == best]
    threshold = round((min(tied) + max(tied)) / 2, 3)
    held = _rates(validation, threshold)
    passed = held["accepted"] > 0 and held["unsafe_share"] <= .05 and held["relevant_pass_rate"] >= .5
    return threshold, [min(tied), max(tied)], passed


def _rates(scored, threshold):
    positives = sum(label for _, label, _ in scored)
    negatives = len(scored) - positives
    tp = sum(label and retrieved and score >= threshold for score, label, retrieved in scored)
    fp = sum(not label and retrieved and score >= threshold for score, label, retrieved in scored)
    return {"threshold": threshold, "relevant_pass_rate": tp / positives if positives else None,
            "irrelevant_pass_rate": fp / negatives if negatives else None,
            "unsafe_share": fp / (tp + fp) if tp + fp else None,
            "accepted": tp + fp, "unsafe_accepted": fp, "size": len(scored)}


def calibration_report(settings, root):
    """Rescan frozen labels; keep current serving configuration completely untouched."""
    root = Path(root)
    report = {"available": False, "active": False, "active_threshold": None,
              "recommended_threshold": None, "generated_at": datetime.now(timezone.utc).isoformat(),
              "message": "尚无可用校准报告", "distributions": [], "scan": []}
    try:
        path = Path(getattr(settings, "evidence_calibration_path", "config/evidence-calibration.json"))
        artifact = json.loads(path.read_text(encoding="utf-8"))
        gate = EvidenceGate(settings, root / "knowledge")
        decision = gate.check([], legacy_confident=False)
        report.update(active=decision.calibrated, active_threshold=decision.threshold,
                      published_threshold=artifact.get("threshold"), active_reason=decision.reason)
        if artifact.get("formula_version") != 2 or artifact.get("purpose") != "relevance_prefilter":
            raise ValueError("该历史校准不是当前相关性预筛公式，不能混合分析")
        directory = root / "reports" / "confidence-calibration"
        training = _label_snapshot(directory, artifact.get("training_labels_sha256"), artifact)
        validation = _label_snapshot(directory, artifact.get("validation_labels_sha256"), artifact)
        ids = [{r["case_id"] for r in part} for part in (training, validation)]
        if ids[0] & ids[1] or any(len(unique) != len(part) for unique, part in zip(ids, (training, validation))):
            raise ValueError("训练与独立验证必须使用不重叠且无重复的题目")
        train_scores, held_scores = _score_rows(training), _score_rows(validation)
        threshold, optimal_range, passed = _recommend(train_scores, held_scores)
        active = decision.threshold
        thresholds = sorted({i / 100 for i in range(101)} | {threshold} | ({active} if active is not None else set()))
        distributions = []
        for name, scored in (("训练", train_scores), ("独立验证", held_scores)):
            for label, title in ((True, "相关证据"), (False, "不相关证据")):
                values = [s for s, relevant, _ in scored if relevant == label]
                distributions.append({"group": f"{name} · {title}", "count": len(values),
                    **{key: _quantile(values, q) for key, q in (("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("max", 1))}})
        report.update(available=True, recommended_threshold=threshold,
            optimal_range=optimal_range, distributions=distributions,
            scan=[_rates(train_scores, t) for t in thresholds],
            training_at_recommended=_rates(train_scores, threshold), validation_at_recommended=_rates(held_scores, threshold),
            training_at_active=_rates(train_scores, active) if active is not None else None,
            validation_at_active=_rates(held_scores, active) if active is not None else None,
            training_size=len(training), validation_size=len(validation), reviewer=artifact.get("reviewer"),
            scope_limit=artifact.get("scope_limit"), validation_passed=passed,
            formula="0.60 × Top1 + 0.25 × min(证据得分总和 / 3, 1) + 0.15 × (Top1 − Top2)",
            message="离线扫描完成；推荐仅由训练集选取，独立验证集只验证。未调用模型，也未发布阈值。" if passed else
                "离线扫描完成，但独立验证未通过；此建议不可发布。未修改当前阈值。")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        report["message"] = f"校准分析不可用：{error}"
    return report


TREND_METRICS = ("recall_at_k", "mrr", "faithfulness", "refusal_accuracy")


def trend_report(runs):
    """Compute deltas only against a previous complete run of the same key."""
    groups = {}
    for run in sorted(runs, key=lambda r: r["id"]):
        metrics = run.get("metrics") or {}
        key = metrics.get("comparison_key")
        group = groups.setdefault(key or "unclassified", {"key": key or "unclassified", "rows": [], "complete_count": 0})
        previous = next((r for r in reversed(group["rows"]) if r["comparable"]), None)
        values = {name: metrics.get(name) for name in TREND_METRICS}
        values["refusal_accuracy"] = (metrics.get("strategies", {}).get("hybrid_rerank", {}).get("refusal_accuracy"))
        values = {name: value if _finite(value) and 0 <= value <= 1 else None for name, value in values.items()}
        comparable = bool(key) and metrics.get("complete") is True
        deltas = {name: values[name] - previous["values"][name]
            if comparable and previous and values[name] is not None and previous["values"][name] is not None else None
            for name in TREND_METRICS}
        row = {**run, "values": values, "deltas": deltas, "comparable": comparable,
               "previous_id": previous["id"] if comparable and previous else None}
        group["rows"].append(row)
        group["complete_count"] += comparable
    for group in groups.values():
        complete = [r for r in group["rows"] if r["comparable"]]
        latest = complete[-1] if complete else None
        group["latest"] = latest
        group["declining"] = [key for key, delta in latest["deltas"].items() if delta is not None and delta < -1e-9] if latest else []
        group["rows"].reverse()
    return {"total_runs": len(runs), "groups": sorted(groups.values(), key=lambda g: g["rows"][0]["id"], reverse=True)}
