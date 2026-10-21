"""Compare Jev with the LLM evidence check on the archived 500-question run (paid; not run in CI).

Every hybrid_rerank case already records the exact evidence the served gate read and, when
the relevance prefilter passed, the LLM's verdict. Jev reads the same payload, so the LLM
baseline needs no new model calls. Labels are expected_answerable. Case IDs ending in 0 or 5
are the validation split, fixed before any Jev score was seen; thresholds use the rest only.
"""

import argparse
import asyncio
from datetime import date
import hashlib
import json
from pathlib import Path
import statistics

from app.config import get_settings
from app.knowledge.jev_evidence import INSTRUCTIONS_SHA256, JevEvidenceJudge

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "retrieval-evaluation.json"
OUT_DIR = ROOT / "reports" / "jev-evidence"
CALIBRATION = ROOT / "config" / "jev-evidence-calibration.json"
PRICE_PER_MILLION = 0.042
MIN_COVERAGE = 0.5
EPSILON = 1e-6


def load_cases(report: dict) -> list[dict]:
    cases = []
    for case in report["cases"]:
        if case["strategy"] != "hybrid_rerank" or case["generation_status"] == "failed":
            continue
        confidence = case.get("evidence_confidence") or {}
        served = bool(confidence.get("relevance_accepted"))
        cases.append({
            "case_id": case["case_id"], "query": case["query"], "evidence": case["citations"],
            "label": bool(case["expected_answerable"]), "served": served,
            "llm": confidence.get("complete") if served else None,
            "validation": case["case_id"][-1] in "05",
        })
    return cases


def choose_bounds(train: list[dict]) -> tuple[float, float]:
    """Jev accepts above every training negative and refuses below every training positive."""
    negatives = [c["p_yes"] for c in train if not c["label"]]
    positives = [c["p_yes"] for c in train if c["label"]]
    if not negatives or not positives:
        raise SystemExit("训练集缺少可答题或库外题的 Jev 分数，无法选阈值")
    upper = min(max(negatives) + EPSILON, 1.0)
    lower = max(min(positives) - EPSILON, 0.0)
    return min(lower, upper - EPSILON), upper


def policy(case: dict, bounds: tuple[float, float]) -> tuple[bool | None, str]:
    """What jev mode serves for this case: Jev outside the band, else the recorded LLM verdict."""
    lower, upper = bounds
    if case["p_yes"] >= upper:
        return True, "jev"
    if case["p_yes"] <= lower:
        return False, "jev"
    return case["llm"], "llm"


def gate_metrics(cases: list[dict], bounds: tuple[float, float]) -> dict:
    """Served-path comparison: only cases whose relevance prefilter passed reach either judge."""
    served = [c for c in cases if c["served"]]
    decided = [(c, *policy(c, bounds)) for c in served]
    return {
        "cases": len(served),
        "positives": sum(c["label"] for c in served),
        "negatives": sum(not c["label"] for c in served),
        "llm_false_refusals": sum(c["label"] and c["llm"] is False for c in served),
        "llm_false_accepts": sum(not c["label"] and c["llm"] is True for c in served),
        "policy_false_refusals": sum(c["label"] and d is False for c, d, _ in decided),
        "policy_false_accepts": sum(not c["label"] and d is True for c, d, _ in decided),
        "jev_coverage": sum(by == "jev" for *_, by in decided) / max(len(served), 1),
    }


def probability_metrics(cases: list[dict]) -> dict:
    p = [c["p_yes"] for c in cases]
    y = [c["label"] for c in cases]
    positives = [pi for pi, yi in zip(p, y) if yi]
    negatives = [pi for pi, yi in zip(p, y) if not yi]
    pairs = sum((a > b) + 0.5 * (a == b) for a in positives for b in negatives)
    bins = []
    for index in range(10):
        members = [(pi, yi) for pi, yi in zip(p, y) if index / 10 <= pi < (index + 1) / 10 or (index == 9 and pi == 1)]
        if members:
            bins.append({"bin": f"{index / 10:.1f}-{(index + 1) / 10:.1f}", "n": len(members),
                         "mean_p": round(statistics.fmean(m[0] for m in members), 3),
                         "answerable_rate": round(statistics.fmean(m[1] for m in members), 3)})
    return {
        "cases": len(cases), "positives": len(positives), "negatives": len(negatives),
        "accuracy_at_0_5": sum((pi >= 0.5) == yi for pi, yi in zip(p, y)) / max(len(p), 1),
        "brier": statistics.fmean((pi - yi) ** 2 for pi, yi in zip(p, y)),
        "auc": pairs / max(len(positives) * len(negatives), 1),
        "reliability": bins,
    }


def dataset_stamp(cases: list[dict], report: dict) -> dict:
    return {"cases_sha256": report["run"]["cases_sha256"], "report_generated_at": report["run"]["generated_at"],
            "evidence_sha256": hashlib.sha256(json.dumps([[c["case_id"], c["evidence"]] for c in cases],
                                                         ensure_ascii=False).encode()).hexdigest()}


def summarize(cases: list[dict], *, model: str, dataset: dict) -> dict:
    scored = [c for c in cases if "p_yes" in c]
    train = [c for c in scored if not c["validation"]]
    validation = [c for c in scored if c["validation"]]
    bounds = choose_bounds(train)
    gate = gate_metrics(validation, bounds)
    passed = (gate["policy_false_accepts"] <= gate["llm_false_accepts"]
              and gate["policy_false_refusals"] <= gate["llm_false_refusals"]
              and gate["jev_coverage"] >= MIN_COVERAGE)
    latencies = sorted(c["latency_ms"] for c in scored)
    tokens = sum(c.get("input_tokens") or 0 for c in scored)
    return {
        "purpose": "jev_evidence_sufficiency", "model": model, "instructions_sha256": INSTRUCTIONS_SHA256,
        "dataset": dataset,
        "lower": bounds[0], "upper": bounds[1],
        "errors": sorted(c["case_id"] for c in cases if "p_yes" not in c),
        "train": {"probability": probability_metrics(train), "served_gate": gate_metrics(train, bounds)},
        "validation": {"probability": probability_metrics(validation), "served_gate": gate, "passed": passed,
                       "rule": f"served validation: false accepts and false refusals <= LLM, Jev coverage >= {MIN_COVERAGE}"},
        "all_cases": probability_metrics(scored),
        "latency_ms": {"p50": latencies[len(latencies) // 2], "p95": latencies[int(len(latencies) * 0.95)]} if latencies else {},
        "input_tokens": tokens, "estimated_cost_usd": round(tokens * PRICE_PER_MILLION / 1e6, 4),
    }


def markdown(summary: dict) -> str:
    v, t = summary["validation"], summary["train"]
    lines = [f"# Jev 证据充分性评测（{summary['model']}）", "",
             f"阈值（只用训练集选）：P ≤ {summary['lower']:.4f} 直接拒答，P ≥ {summary['upper']:.4f} 直接放行，中间交给大模型。", "",
             "| 集合 | 题数 | 可答/库外 | 准确率@0.5 | Brier | AUC |", "|---|---:|---:|---:|---:|---:|"]
    for name, part in (("训练", t["probability"]), ("验证", v["probability"]), ("全部", summary["all_cases"])):
        lines.append(f"| {name} | {part['cases']} | {part['positives']}/{part['negatives']} | {part['accuracy_at_0_5']:.3f} "
                     f"| {part['brier']:.3f} | {part['auc']:.3f} |")
    lines += ["", "线上路径（相关性粗筛通过的题）：", "",
              "| 集合 | 题数 | 误拒 大模型→Jev 模式 | 误放 大模型→Jev 模式 | Jev 直接决定 |", "|---|---:|---:|---:|---:|"]
    for name, gate in (("训练", t["served_gate"]), ("验证", v["served_gate"])):
        lines.append(f"| {name} | {gate['cases']} | {gate['llm_false_refusals']} → {gate['policy_false_refusals']} "
                     f"| {gate['llm_false_accepts']} → {gate['policy_false_accepts']} | {gate['jev_coverage']:.1%} |")
    lines += ["", f"上线门槛：{v['rule']}。结果：{'通过' if v['passed'] else '未通过'}。", "",
              "可靠性（全部题）：", "", "| P 区间 | 题数 | 平均 P | 实际可答比例 |", "|---|---:|---:|---:|"]
    lines += [f"| {b['bin']} | {b['n']} | {b['mean_p']} | {b['answerable_rate']} |" for b in summary["all_cases"]["reliability"]]
    lines += ["", f"延迟 P50 {summary['latency_ms'].get('p50')} ms，P95 {summary['latency_ms'].get('p95')} ms；"
                  f"输入 {summary['input_tokens']} token，约 ${summary['estimated_cost_usd']}。调用失败：{summary['errors'] or '无'}。"]
    return "\n".join(lines) + "\n"


async def score(cases: list[dict], judge: JevEvidenceJudge, concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case):
        async with semaphore:
            case.update(await judge.judge(case["query"], case["evidence"]))

    await asyncio.gather(*(one(case) for case in cases))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="score only the first N cases (smoke run)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--from-results", type=Path, help="re-summarize a saved run instead of calling Jev")
    parser.add_argument("--publish", action="store_true", help="write the calibration file when the gate passes")
    args = parser.parse_args()
    settings = get_settings()
    if args.from_results:
        saved = json.loads(args.from_results.read_text(encoding="utf-8"))
        cases, dataset = saved["cases"], saved["summary"]["dataset"]  # the run's own data, not today's report
    else:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        from typesafe_sdk import AsyncTypeSafeClient

        if not settings.typesafe_api_key:
            raise SystemExit("需要在 .env 设置 TYPESAFE_API_KEY")
        cases = load_cases(report)[:args.limit]
        dataset = dataset_stamp(cases, report)
        async with AsyncTypeSafeClient(api_key=settings.typesafe_api_key.get_secret_value(),
                                       model=settings.jev_model, timeout=settings.jev_timeout_seconds) as client:
            await score(cases, JevEvidenceJudge(client, model=settings.jev_model,
                                                timeout=settings.jev_timeout_seconds), args.concurrency)
    if args.limit:
        for c in cases:
            print(c["case_id"], c["label"], {k: c.get(k) for k in ("p_yes", "model", "latency_ms", "input_tokens", "error")})
        return 0
    models = {c["model"] for c in cases if "model" in c}
    if len(models) != 1:
        raise SystemExit(f"Jev 返回的模型版本不唯一：{models or '全部调用失败'}")
    summary = summarize(cases, model=models.pop(), dataset=dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"jev-evidence-{date.today():%Y%m%d}"
    stem.with_suffix(".json").write_text(json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    if args.publish and summary["validation"]["passed"]:
        CALIBRATION.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已发布 {CALIBRATION.relative_to(ROOT)}")
    return 0 if summary["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
