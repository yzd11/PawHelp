"""Does Jev flag wrong intent decisions better than the LLM's self-reported confidence? (paid; not run in CI)

The LLM classifies every case in evals/intent_jev_cases.jsonl several times through the served
classify_intent. Jev answers one Choice question per case over the ten intents. A review rule
asks the customer to clarify when the intent is 其他 or a score falls below a threshold; the
score is the LLM's confidence or Jev's probability for the LLM's chosen intent. Thresholds are
chosen on the train split only and compared with the current rule on the fixed validation split.
"""

import argparse
import asyncio
from datetime import date
import hashlib
import json
from pathlib import Path
import time

from app.config import get_settings
from app.llm import create_chat_model
from app.workflow.understanding import INTENT_REVIEW_THRESHOLD, ResolvedQuestion, classify_intent, plan_route
from scripts.run_intent_evals import mismatches
from scripts.run_jev_evidence_evals import PRICE_PER_MILLION, probability_metrics

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evals" / "intent_jev_cases.jsonl"
OUT_DIR = ROOT / "reports" / "jev-intent"
JEV_QUESTION = "intent"
# English: Jev is trained mostly on English. Keys are mapped back to the served Chinese labels.
JEV_INTENTS = {
    "logistics": ("物流", "Where a parcel is or how its delivery is progressing, or general carrier and delivery-time questions."),
    "order": ("订单", "Placing, checking, changing or cancelling an order, or an order's payment status."),
    "product_or_rules": ("商品咨询", "Questions about products or the shop's rules and services: specs, price, usage, cleaning, "
                         "safety, fault codes, recycling, invoices, points, coupons, accounts, service hours or contacts."),
    "refund_return": ("退款退货", "Asking for a refund or a return, the return and refund policy, or the progress of a refund."),
    "after_sales": ("售后", "Repair, exchange, damaged or faulty goods, missing parts, warranty, or where a returned or "
                    "repaired item sent back now is."),
    "complaint": ("投诉", "Explicitly complaining about or expressing dissatisfaction with the shop or its service."),
    "human_agent": ("人工", "Explicitly asking for a human agent, a support ticket, or a person to follow up; "
                    "this takes precedence over a complaint."),
    "conversation_recall": ("会话回顾", "Asking what was said, asked or found earlier in this same conversation, "
                            "without asking for fresh data."),
    "small_talk": ("闲聊", "Greetings, thanks, acknowledgements or other social talk that a fixed reply can answer."),
    "other": ("其他", "Unrelated to the shop's business, or too vague to tell what the customer wants."),
}
JEV_INSTRUCTIONS = ("A customer of an online pet-supplies shop sent this single message, with no earlier conversation. "
                    "Which intent best describes what the customer wants the shop to do now?")
INSTRUCTIONS_SHA256 = hashlib.sha256(json.dumps([JEV_INSTRUCTIONS, JEV_INTENTS], ensure_ascii=False).encode()).hexdigest()


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def outcome(case: dict, intent: str, flagged: bool) -> str:
    """What the customer gets: clarify is right only for 其他 cases, serving is right only with an accepted intent."""
    if flagged:
        return "ok" if "其他" in case["intents"] else "clarified_clear"
    return "ok" if intent in case["intents"] else "served_wrong"


def errors(runs: list[dict], score: str, threshold: float) -> dict:
    counts = {"served_wrong": 0, "clarified_clear": 0}
    for run in runs:
        result = outcome(run["case"], run["intent"], run["intent"] == "其他" or run[score] < threshold)
        if result != "ok":
            counts[result] += 1
    return counts


def choose_threshold(runs: list[dict], score: str) -> float:
    """Fewest train errors; ties prefer fewer wrong answers served, then the lower threshold."""
    candidates = sorted({0.0, 1.000001, *(run[score] for run in runs)})
    return min(candidates, key=lambda t: (sum(errors(runs, score, t).values()), errors(runs, score, t)["served_wrong"], t))


def signal_auc(runs: list[dict], score: str) -> dict:
    """How well a score separates right from wrong intents among decisions the rule could still flag."""
    rows = [{"p_yes": run[score], "label": run["intent"] in run["case"]["intents"]} for run in runs if run["intent"] != "其他"]
    metrics = probability_metrics(rows)
    return {"runs": metrics["cases"], "wrong": metrics["negatives"], "auc": metrics["auc"] if metrics["negatives"] else None}


def summarize(cases: list[dict], *, model: str, jev_model: str, runs_per_case: int) -> dict:
    scored = [c for c in cases if "jev" in c and "probabilities" in c["jev"] and c["llm"]]
    runs = [{**run, "case": c, "jev_p": c["jev"]["probabilities"].get(run["intent"], 0.0)}
            for c in scored for run in c["llm"] if "intent" in run]
    train = [r for r in runs if r["case"]["split"] == "train"]
    validation = [r for r in runs if r["case"]["split"] == "validation"]
    rules = {"current": ("confidence", INTENT_REVIEW_THRESHOLD),
             "llm_tuned": ("confidence", choose_threshold(train, "confidence")),
             "jev_tuned": ("jev_p", choose_threshold(train, "jev_p"))}
    split = {}
    for name, part in (("train", train), ("validation", validation)):
        split[name] = {"runs": len(part),
                       "llm_intent_accuracy": sum(r["intent"] in r["case"]["intents"] for r in part) / max(len(part), 1),
                       "llm_all_fields": sum(not r["mismatches"] for r in part) / max(len(part), 1),
                       "auc": {"confidence": signal_auc(part, "confidence"), "jev_p": signal_auc(part, "jev_p")},
                       "rules": {rule: {"score": score, "threshold": t, **errors(part, score, t)}
                                 for rule, (score, t) in rules.items()}}
    for name in split:
        part_cases = [c for c in scored if c["split"] == name]
        split[name]["jev_top1_accuracy"] = sum(c["jev"]["top"] in c["intents"] for c in part_cases) / max(len(part_cases), 1)
    v = split["validation"]
    jev_auc, llm_auc = v["auc"]["jev_p"]["auc"], v["auc"]["confidence"]["auc"]
    passed = (v["rules"]["jev_tuned"]["served_wrong"] <= v["rules"]["current"]["served_wrong"]
              and v["rules"]["jev_tuned"]["clarified_clear"] <= v["rules"]["current"]["clarified_clear"]
              and jev_auc is not None and llm_auc is not None and jev_auc > llm_auc)
    latencies = sorted(c["jev"]["latency_ms"] for c in scored)
    tokens = sum(c["jev"].get("input_tokens") or 0 for c in scored)
    return {
        "purpose": "jev_intent_review", "model": model, "jev_model": jev_model, "instructions_sha256": INSTRUCTIONS_SHA256,
        "cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(), "runs_per_case": runs_per_case,
        "errors": sorted(c["id"] for c in cases if c not in scored),
        "train": split["train"], "validation": v, "passed": passed,
        "rule": "validation: Jev rule serves no more wrong intents and asks no more clear questions to clarify than the "
                f"current {INTENT_REVIEW_THRESHOLD} rule, and its AUC is higher than the LLM confidence's",
        "jev_latency_ms": {"p50": latencies[len(latencies) // 2], "p95": latencies[int(len(latencies) * 0.95)]} if latencies else {},
        "jev_input_tokens": tokens, "jev_estimated_cost_usd": round(tokens * PRICE_PER_MILLION / 1e6, 4),
    }


def markdown(summary: dict) -> str:
    lines = [f"# Jev 意图复核评测（大模型 {summary['model']}，Jev {summary['jev_model']}）", "",
             f"每题大模型调用 {summary['runs_per_case']} 次，Jev 1 次。阈值只用训练集选。", "",
             "| 集合 | 次数 | 大模型意图准确率 | 大模型全字段通过 | Jev top-1 准确率 | AUC 大模型自报 | AUC Jev | 可被复核的错判 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    for name, label in (("train", "训练"), ("validation", "验证")):
        p = summary[name]
        lines.append(f"| {label} | {p['runs']} | {p['llm_intent_accuracy']:.3f} | {p['llm_all_fields']:.3f} | {p['jev_top1_accuracy']:.3f} "
                     f"| {fmt(p['auc']['confidence']['auc'])} | {fmt(p['auc']['jev_p']['auc'])} | {p['auc']['confidence']['wrong']} |")
    lines += ["", "复核规则（意图为“其他”或分数低于阈值就请用户澄清）：", "",
              "| 规则 | 分数 | 阈值 | 训练 错答/多问 | 验证 错答/多问 |", "|---|---|---:|---:|---:|"]
    for rule, label in (("current", "现行"), ("llm_tuned", "大模型自报重选阈值"), ("jev_tuned", "Jev")):
        t, v = summary["train"]["rules"][rule], summary["validation"]["rules"][rule]
        lines.append(f"| {label} | {t['score']} | {t['threshold']:.3f} | {t['served_wrong']}/{t['clarified_clear']} "
                     f"| {v['served_wrong']}/{v['clarified_clear']} |")
    lines += ["", "错答=没有澄清就按错误意图处理；多问=用户意图清楚却被要求澄清。", "",
              f"门槛：{summary['rule']}。结果：{'通过' if summary['passed'] else '未通过'}。", "",
              f"Jev 延迟 P50 {summary['jev_latency_ms'].get('p50')} ms，P95 {summary['jev_latency_ms'].get('p95')} ms；"
              f"输入 {summary['jev_input_tokens']} token，约 ${summary['jev_estimated_cost_usd']}。调用失败：{summary['errors'] or '无'}。"]
    return "\n".join(lines) + "\n"


async def jev_choice(client, question: str, *, model: str, timeout: float) -> dict:
    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(client.system_one(
            json.dumps({"customer_message": question}, ensure_ascii=False),
            {JEV_QUESTION: {"type": "choice", "instructions": JEV_INSTRUCTIONS,
                            "criteria": {key: text for key, (_, text) in JEV_INTENTS.items()}}},
            model=model, timeout=timeout), timeout + 1)
        answer = response.choices[JEV_QUESTION]
        return {"top": JEV_INTENTS[answer.choice][0], "model": response.model,
                "probabilities": {JEV_INTENTS[key][0]: p for key, p in answer.probabilities.items()},
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "input_tokens": getattr(response.usage, "input_tokens", None)}
    except Exception as error:  # one failed call is a missing score, not a crashed run
        return {"error": type(error).__name__, "latency_ms": round((time.perf_counter() - started) * 1000)}


async def llm_runs(model, method: str, case: dict, runs: int, gate: asyncio.Semaphore) -> list[dict]:
    results = []
    for _ in range(runs):
        async with gate:
            try:
                decision = await classify_intent(model, method, case["question"], [])
            except Exception as error:
                results.append({"error": type(error).__name__})
                continue
        plan = plan_route(decision, ResolvedQuestion(question=case["question"]))
        results.append({"intent": decision.intent, "confidence": decision.confidence, "task_kind": decision.task_kind,
                        "mismatches": mismatches(case, decision, plan)})
    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="first N cases only; prints per-case results (smoke run)")
    parser.add_argument("--runs", type=int, default=3, help="LLM classifier calls per case")
    parser.add_argument("--from-results", type=Path, help="re-summarize a saved run instead of calling the models")
    args = parser.parse_args()
    settings = get_settings()
    if args.from_results:
        saved = json.loads(args.from_results.read_text(encoding="utf-8"))
        cases, runs = saved["cases"], saved["summary"]["runs_per_case"]
    else:
        from typesafe_sdk import AsyncTypeSafeClient

        if not settings.typesafe_api_key:
            raise SystemExit("需要在 .env 设置 TYPESAFE_API_KEY")
        cases, runs = load_cases()[:args.limit], args.runs
        model, gate, jev_gate = create_chat_model(settings), asyncio.Semaphore(4), asyncio.Semaphore(8)
        async with AsyncTypeSafeClient(api_key=settings.typesafe_api_key.get_secret_value(),
                                       model=settings.jev_model, timeout=settings.jev_timeout_seconds) as client:
            async def one(case):
                async def jev():
                    async with jev_gate:
                        return await jev_choice(client, case["question"], model=settings.jev_model,
                                                timeout=settings.jev_timeout_seconds)
                case["jev"], case["llm"] = await asyncio.gather(
                    jev(), llm_runs(model, settings.structured_output_method, case, runs, gate))
            await asyncio.gather(*(one(case) for case in cases))
    if args.limit:
        for c in cases:
            print(c["id"], c["question"], c["intents"], "| jev:", {k: c["jev"].get(k) for k in ("top", "error", "latency_ms")},
                  "p=", {k: round(v, 3) for k, v in sorted(c["jev"].get("probabilities", {}).items(), key=lambda kv: -kv[1])[:3]},
                  "| llm:", [(r.get("intent"), r.get("confidence"), r.get("mismatches") or r.get("error")) for r in c["llm"]])
        return 0
    jev_models = {c["jev"]["model"] for c in cases if "model" in c["jev"]}
    if len(jev_models) != 1:
        raise SystemExit(f"Jev 返回的模型版本不唯一：{jev_models or '全部调用失败'}")
    summary = summarize(cases, model=settings.openai_model, jev_model=jev_models.pop(), runs_per_case=runs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"jev-intent-{date.today():%Y%m%d}"
    stem.with_suffix(".json").write_text(json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
