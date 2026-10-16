"""Multilabel metrics and threshold policy, shared by evaluation and serving."""
import math

from app.topics.taxonomy import TOPIC_NAMES


def sigmoid(x: float) -> float:
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def labels_from_scores(scores: list[float], threshold: float) -> list[str]:
    if len(scores) != len(TOPIC_NAMES) or not 0 < threshold < 1 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("invalid probability vector or threshold")
    labels = [label for label, score in zip(TOPIC_NAMES, scores) if score >= threshold and label != "其他"]
    # Other is exclusive. No specific class over threshold is an explicit
    # fallback, not evidence that the model confidently recognized 'other'.
    return labels or ["其他"]


def evaluate(expected: list[list[str]], predicted: list[list[str]]) -> dict:
    if not expected or len(expected) != len(predicted):
        raise ValueError("equal, nonempty expected and prediction arrays required")
    per_class = []
    total_tp = total_fp = total_fn = 0
    for label in TOPIC_NAMES:
        tp = sum(label in e and label in p for e,p in zip(expected,predicted))
        fp = sum(label not in e and label in p for e,p in zip(expected,predicted))
        fn = sum(label in e and label not in p for e,p in zip(expected,predicted))
        tn = len(expected)-tp-fp-fn
        precision, recall = tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        per_class.append({"label": label, "precision": precision, "recall": recall, "f1": f1,
            "support": tp+fn, "matrix": [[tn,fp],[fn,tp]]})
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return {"size": len(expected), "per_class": per_class,
        "micro": {"precision": total_tp/(total_tp+total_fp) if total_tp+total_fp else 0,
            "recall": total_tp/(total_tp+total_fn) if total_tp+total_fn else 0,
            "f1": 2*total_tp/(2*total_tp+total_fp+total_fn) if 2*total_tp+total_fp+total_fn else 0},
        "macro": {key: sum(c[key] for c in per_class)/len(per_class) for key in ("precision","recall","f1")},
        "exact_match": sum(set(e)==set(p) for e,p in zip(expected,predicted))/len(expected),
        "errors": {"false_positive_labels": total_fp, "false_negative_labels": total_fn,
            "wrong_samples": sum(set(e)!=set(p) for e,p in zip(expected,predicted))}}


def evaluate_sources(expected, predicted, sources):
    """Do not hide a tiny real holdout inside a mostly synthetic aggregate."""
    if len(sources)!=len(expected) or len(predicted)!=len(expected):
        raise ValueError('source/prediction/label arrays must align')
    return {source:evaluate([e for e,s in zip(expected,sources) if s==source],
                           [p for p,s in zip(predicted,sources) if s==source])
            for source in sorted(set(sources))}


def scan_threshold(expected: list[list[str]], scores: list[list[float]]) -> dict:
    trials = []
    for n in range(30,71,5):
        threshold = n/100
        report = evaluate(expected, [labels_from_scores(row, threshold) for row in scores])
        trials.append({"threshold": threshold, "micro_f1": report["micro"]["f1"], "macro_f1": report["macro"]["f1"],
            "fp": report["errors"]["false_positive_labels"], "fn": report["errors"]["false_negative_labels"]})
    # Deterministic ties: fewer false positives, then higher threshold.
    chosen = max(trials, key=lambda row: (row["micro_f1"], -row["fp"], row["threshold"]))
    return {"selection_split": "validation_only", "tie_rule": "micro_f1, fewer_fp, higher_threshold", "chosen": chosen, "trials": trials}
