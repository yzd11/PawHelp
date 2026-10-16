"""Read-only report projection. Never loads weights or starts classifier jobs."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from app.topics.data import digest, redact
from app.topics.taxonomy import TAXONOMY_HASH, TOPIC_NAMES


REPORT_FILES = {
    "golden": "golden-predictions.report.json",
    "legacy": "reports/legacy-audit.json",
    "environment": "reports/environment-witness.json",
    "dataset": "dataset/manifest.json",
    "model": "model/threshold.json",
    "training": "model/training-history.json",
    "threshold": "model/threshold-scan.json",
    "evaluation": "reports/eval-report.json",
    "errors": "reports/error-samples.json",
}
ARTIFACT_GROUPS = {
    "corpus": ["pool-20261013.jsonl", "pool-prelabels.jsonl", "synthetic-v1.jsonl",
               "sample-review-approved.json", "corpus-approved-v1.jsonl", "corpus-grouped-v2.jsonl"],
    "dataset": ["dataset/train.jsonl", "dataset/val.jsonl", "dataset/test.jsonl", "dataset/manifest.json"],
    "model": ["model/config.json", "model/model.safetensors", "model/tokenizer.json",
              "model/tokenizer_config.json", "model/threshold.json", "model/training-history.json"],
    "reports": list(REPORT_FILES.values()),
}


def _number(value, *, probability=False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (not probability or 0 <= value <= 1))


def _count(value) -> bool:
    return type(value) is int and value >= 0


def _labels(value, *, empty=False) -> bool:
    return (isinstance(value, list) and (empty or bool(value))
            and all(isinstance(x, str) and x in TOPIC_NAMES for x in value)
            and len(value) == len(set(value)))


def _valid(key: str, data) -> bool:
    """Validate the nested fields consumed by the projection and browser."""
    if key == "errors":
        return isinstance(data, list) and all(
            isinstance(row, dict) and isinstance(row.get("text"), str)
            and all(_labels(row.get(k), empty=k in ("missed", "extra")) for k in ("expected", "predicted", "missed", "extra"))
            and set(row["missed"]) == set(row["expected"]) - set(row["predicted"])
            and set(row["extra"]) == set(row["predicted"]) - set(row["expected"])
            and isinstance(row.get("scores", {}), dict) for row in data)
    if not isinstance(data, dict):
        return False
    if key == "model":
        return (isinstance(data.get("model_version"), str) and bool(data["model_version"])
                and _number(data.get("threshold"), probability=True) and 0 < data["threshold"] < 1
                and data.get("labels") == list(TOPIC_NAMES) and isinstance(data.get("dataset_hashes"), dict))
    if key == "dataset":
        return all(isinstance(data.get(stage), dict) and all(
            isinstance(data[stage].get(split), dict) and _count(data[stage][split].get("size"))
            for split in ("train", "val", "test")) for stage in ("before_augmentation", "after_augmentation")) and (
                isinstance(data.get("overlaps", {}), dict) and all(isinstance(v, dict) for v in data.get("overlaps", {}).values()))
    if key == "threshold":
        trials = data.get("trials")
        chosen = data.get("chosen")
        valid_trial = lambda row: (isinstance(row, dict) and _number(row.get("threshold"), probability=True)
                                  and _number(row.get("micro_f1"), probability=True)
                                  and _number(row.get("macro_f1"), probability=True)
                                  and _count(row.get("fp")) and _count(row.get("fn")))
        return (data.get("selection_split") == "validation_only"
                and isinstance(trials, list) and bool(trials) and all(valid_trial(row) for row in trials)
                and valid_trial(chosen) and chosen in trials
                and isinstance(data.get("validation_scores", []), list))
    if key == "training":
        return isinstance(data.get("history"), list) and all(isinstance(row, dict) for row in data["history"])
    if key == "golden":
        return _count(data.get("size")) and _number(data.get("exact_match"), probability=True) and type(data.get("gate_passed")) is bool
    if key == "evaluation":
        rows = data.get("per_class")
        if not (_count(data.get("size")) and data["size"] > 0 and isinstance(rows, list) and len(rows) == len(TOPIC_NAMES)
                and all(isinstance(row, dict) for row in rows)
                and [row.get("label") for row in rows] == list(TOPIC_NAMES)
                and isinstance(data.get("errors"), dict) and isinstance(data.get("source_counts", {}), dict)
                and all(_count(data["errors"].get(k)) for k in ("wrong_samples", "false_positive_labels", "false_negative_labels"))
                and all(isinstance(data.get(k), dict) and all(_number(data[k].get(m), probability=True)
                    for m in ("precision", "recall", "f1")) for k in ("micro", "macro"))):
            return False
        for row in rows:
            m = row.get("matrix")
            if not (_count(row.get("support")) and all(_number(row.get(k), probability=True) for k in ("precision", "recall", "f1"))
                    and isinstance(m, list) and len(m) == 2 and all(isinstance(r, list) and len(r) == 2 for r in m)
                    and all(_count(n) for r in m for n in r) and sum(sum(r) for r in m) == data["size"]
                    and row["support"] == m[1][0] + m[1][1]):
                return False
        return (sum(row["matrix"][0][1] for row in rows) == data["errors"]["false_positive_labels"]
                and sum(row["matrix"][1][0] for row in rows) == data["errors"]["false_negative_labels"])
    return True


def _safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise OSError("artifact outside report root")
    return path


def read_report(root: Path, relative: str) -> dict:
    try:
        path = _safe_path(root, relative)
        stat = path.stat()
        if stat.st_size > 8 * 1024 * 1024:
            raise ValueError("report too large")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, (dict, list)):
            raise ValueError("expected report object or list")
        return {"present": True, "path": relative, "data": data,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
    except FileNotFoundError:
        return {"present": False, "path": relative}
    except (OSError, ValueError):
        return {"present": False, "path": relative, "error": "产物读取失败或格式无效"}


def _inventory(root: Path) -> dict:
    result = {}
    for group, paths in ARTIFACT_GROUPS.items():
        result[group] = []
        for relative in dict.fromkeys(paths):
            item = {"path": relative, "present": False, "rows": None}
            try:
                path = _safe_path(root, relative)
                stat = path.stat()
                if not path.is_file():
                    raise OSError("not a file")
                item.update(present=True, size_bytes=stat.st_size,
                            modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())
                if path.suffix == ".jsonl" and stat.st_size <= 16 * 1024 * 1024:
                    with path.open(encoding="utf-8-sig") as stream:
                        item["rows"] = sum(1 for line in stream if line.strip())
            except FileNotFoundError:
                pass
            except (OSError, UnicodeError):
                item["error"] = "文件无法读取"
            result[group].append(item)
    return result


def _latest_batch(root: Path) -> dict:
    try:
        folder = _safe_path(root, "reports/batches")
        paths = sorted(folder.glob("classify-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not paths:
            return {"present": False}
        report = read_report(root, paths[0].relative_to(root).as_posix())
        if not report["present"] or not isinstance(report["data"], dict):
            return {"present": False, "error": "最近批次记录不可读"}
        data = report["data"]
        # Deliberately omit per-question scores, text, and input hashes.
        return {"present": True, "status": data.get("status"), "started_at": data.get("started_at"),
                "finished_at": data.get("finished_at"), "model_version": data.get("model_version"),
                "pending": data.get("pending"), "saved": sum(row.get("status") == "saved" for row in data.get("classified", [])),
                "failed": len(data.get("failed", [])), "path": report["path"]}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"present": False, "error": "最近批次记录不可读"}


def acceptance_reports(project: Path) -> dict:
    root = project / "data" / "topic-classifier"
    result = {key: read_report(root, path) for key, path in REPORT_FILES.items()}
    # Shape checks turn incomplete files into local empty states, not whole-page crashes.
    for key, report in result.items():
        if report["present"] and not _valid(key, report["data"]):
            report.update(present=False, error="报告结构无效")
            report.pop("data", None)
    model = result["model"].get("data", {})
    evaluation = result["evaluation"].get("data", {})
    dataset = result["dataset"].get("data", {})
    reasons = []
    if evaluation:
        if not model.get("model_version") or evaluation.get("model_version") != model.get("model_version"):
            reasons.append("评测模型版本与当前模型不一致")
        if evaluation.get("threshold") != model.get("threshold"):
            reasons.append("评测阈值与当前模型不一致")
        if evaluation.get("taxonomy_hash") != TAXONOMY_HASH or model.get("taxonomy_hash") != TAXONOMY_HASH:
            reasons.append("评测或模型类目版本已变化")
        if not dataset or model.get("manifest_hash") != digest(dataset):
            reasons.append("当前数据集清单与训练记录不一致")
        if evaluation.get("test_sha256") != model.get("dataset_hashes", {}).get("test"):
            reasons.append("测试集摘要与训练记录不一致")
    result["evaluation"].update(stale=bool(reasons), stale_reasons=reasons)
    if result["golden"]["present"]:
        result["golden"]["stale"] = result["golden"]["data"].get("taxonomy_hash") != TAXONOMY_HASH
    # Error samples have no version field: only trust a companion evaluation and matching totals.
    rows = result["errors"].get("data", [])
    compatible_errors = (bool(evaluation) and not reasons
                         and len(rows) == evaluation.get("errors", {}).get("wrong_samples")
                         and sum(len(r.get("missed", [])) for r in rows) == evaluation.get("errors", {}).get("false_negative_labels")
                         and sum(len(r.get("extra", [])) for r in rows) == evaluation.get("errors", {}).get("false_positive_labels"))
    result["errors"]["stale"] = not compatible_errors
    result["errors"]["provenance"] = "unverified_no_model_version"
    for row in rows:
        row["text"] = redact(str(row.get("text", "")))
    scan = result["threshold"].get("data", {})
    result["threshold"]["validation_size"] = len(scan.get("validation_scores", []))
    # The browser needs aggregate trials, not thousands of raw probabilities.
    scan.pop("validation_scores", None)
    scan.pop("validation_labels", None)
    scan["matches_model"] = bool(scan) and scan.get("chosen", {}).get("threshold") == model.get("threshold")
    policy = read_report(project / "config", "topic-acceptance.json")
    if policy["present"]:
        data = policy["data"]
        valid = isinstance(data, dict) and all(isinstance(data.get(k), dict)
                and _labels(data[k].get("labels"))
                and (data[k].get("minimum_f1") is None or _number(data[k]["minimum_f1"], probability=True))
                for k in ("strict", "standard", "observe"))
        if valid:
            labels = [label for k in ("strict", "standard", "observe") for label in data[k]["labels"]]
            valid = len(labels) == len(TOPIC_NAMES) and set(labels) == set(TOPIC_NAMES)
        if not valid:
            policy = {"present": False, "path": "topic-acceptance.json", "error": "验收配置无效"}
    result["acceptance_policy"] = policy
    result["inventory"] = _inventory(root)
    result["batch"] = _latest_batch(root)
    review = read_report(root, "sample-review-approved.json")
    approved = review.get("data", {})
    if not isinstance(approved, dict) or not isinstance(approved.get("items", []), list) or not isinstance(approved.get("corpus_report", {}), dict):
        approved = {}
        review["present"] = False
    result["review_summary"] = {"present": review["present"], "approved": approved.get("approved") is True,
                                "samples": len(approved.get("items", [])), "approved_at": approved.get("approved_at"),
                                "candidates": approved.get("corpus_report", {}).get("size")}
    result["taxonomy_hash"] = TAXONOMY_HASH
    result["topics"] = list(TOPIC_NAMES)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result
