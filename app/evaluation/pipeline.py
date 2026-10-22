"""Bounded manual/opt-in evaluation runs and comparable history."""
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import and_, func, select
from langchain_core.callbacks import AsyncCallbackHandler
from app.database.models import Conversation, EvalRun, Message, RequestObservation
from app.evaluation.runtime import DEFAULT_EVAL_CASES_PATH, DEFAULT_GENERATION_STRATEGIES, run_retrieval_evaluation

logger = logging.getLogger(__name__)


class ModelCallBudgetExceeded(RuntimeError): pass


class CallBudget(AsyncCallbackHandler):
    raise_error = True
    def __init__(self, limit): self.limit, self.calls = limit, 0
    async def on_chat_model_start(self, serialized, messages, **kwargs):
        if self.calls >= self.limit: raise ModelCallBudgetExceeded("评估模型调用预算已耗尽")
        self.calls += 1


def metrics_from_report(report):
    run = report.get("run", {})
    strategies = report.get("strategies", {})
    selected = strategies.get("hybrid_rerank", {})
    rows = [r for r in report.get("cases", []) if r.get("strategy") == "hybrid_rerank"]
    answerable_rows = [r for r in rows if r.get("expected_answerable", True)]
    faith_rows = [r for r in rows if r.get("faithfulness_status") == "completed"]
    errors = sum(any(r.get(k) == "failed" for k in ("generation_status", "coverage_status", "faithfulness_status")) for r in rows)
    config = {k: run.get(k) for k in ("cases_sha256", "corpus_sha256", "embedding_model", "reranker_model", "judge_model", "generation_strategies", "normalize_queries", "retrieval_configuration", "evidence_gate")}
    config["case_ids"] = sorted(r["case_id"] for r in rows)
    comparison_key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return {"recall_at_k": selected.get("recall_at_k"), "mrr": selected.get("mrr"),
            "faithfulness": sum(r["faithfulness"] for r in faith_rows) / len(faith_rows) if faith_rows else None,
            "faithfulness_sample_count": len(faith_rows), "error_cases": errors,
            # Absent-knowledge cases only evaluate refusal, not answer coverage/faithfulness.
            "complete": errors == 0 and bool(rows) and all(r.get("generation_status") == "completed" for r in rows)
                        and all(r.get("coverage_status") == "completed" and r.get("faithfulness_status") == "completed" for r in answerable_rows),
            "comparison_key": comparison_key, "configuration": config, "strategies": strategies}


class EvaluationPipeline:
    def __init__(self, settings, sessions, root, lock):
        self.settings, self.sessions, self.root, self.lock = settings, sessions, Path(root), lock
        self.last_error = None

    async def record(self, report, triggered_by="手动", path=None):
        metrics = metrics_from_report(report)
        metrics["report_path"] = str(path) if path else None
        async with self.sessions() as session, session.begin():
            row = EvalRun(triggered_by=triggered_by, dataset_size=report.get("dataset", {}).get("total_cases", 0), metrics=metrics)
            session.add(row)
            await session.flush()
            return row.id

    async def run(self, triggered_by="手动"):
        if self.lock.locked(): raise ValueError("已有评估正在运行")
        async with self.lock:
            directory = self.root / "reports" / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
            budget = CallBudget(self.settings.evaluation_max_model_calls)
            try:
                report, paths = await run_retrieval_evaluation(self.settings, self.root / DEFAULT_EVAL_CASES_PATH, directory,
                    sample_per_bucket=self.settings.evaluation_sample_per_bucket, generation_strategies=DEFAULT_GENERATION_STRATEGIES,
                    model_callbacks=[budget])
                run_id = await self.record(report, triggered_by, paths[0])
                self.last_error = None
                return {"run_id": run_id, "metrics": metrics_from_report(report), "model_calls": budget.calls, "report_path": str(paths[0])}
            except Exception:
                self.last_error = "评估未完成，请检查日志和模型余额；没有写入虚假的成功轮次"
                logger.exception("evaluation pipeline failed")
                raise

    async def history(self):
        async with self.sessions() as session:
            rows = list(await session.scalars(select(EvalRun).order_by(EvalRun.id.desc()).limit(100)))
        return [{"id": r.id, "triggered_by": r.triggered_by, "dataset_size": r.dataset_size, "metrics": r.metrics,
                 "created_at": r.created_at.replace(tzinfo=timezone.utc).isoformat()} for r in reversed(rows)]


async def cost_by_intent(sessions, settings, days=7, *, now=None):
    if not 1 <= days <= 90:
        raise ValueError("成本统计窗口应为 1 到 90 天")
    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    start = end - timedelta(days=days)
    async with sessions() as session:
        window = (RequestObservation.created_at >= start.replace(tzinfo=None),
                  RequestObservation.created_at <= end.replace(tzinfo=None))
        has_conversation = select(Conversation.id).where(Conversation.id == RequestObservation.conversation_id).exists()
        has_message = select(Message.id).where(Message.id == RequestObservation.user_message_id,
            Message.conversation_id == RequestObservation.conversation_id).exists()
        json_length = func.json_array_length if session.get_bind().dialect.name == "sqlite" else func.json_length
        # Quarantine only empty, unlinked records. Keep recorded historical costs
        # even if their original conversation has since been removed.
        isolated = and_(func.lower(func.json_type(RequestObservation.model_usage)) == "array",
            func.coalesce(json_length(RequestObservation.model_usage), -1) == 0, ~has_conversation, ~has_message)
        excluded = await session.scalar(select(func.count()).select_from(RequestObservation).where(*window, isolated))
        rows = list(await session.scalars(select(RequestObservation)
            .where(*window, ~isolated)
            .order_by(RequestObservation.created_at.desc(), RequestObservation.id.desc()).limit(5001)))
    truncated = len(rows) > 5000
    rows = rows[:5000]
    totals = {}
    for row in rows:
        target = totals.setdefault(row.intent, {"intent": row.intent, "requests": 0, "model_calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0, "unknown_usage_calls": 0, "unobserved_requests": 0})
        target["requests"] += 1
        target["unobserved_requests"] += not bool(row.model_usage)
        for usage in row.model_usage or []:
            target["model_calls"] += 1
            target["failed_calls"] += usage["status"] == "failed"
            if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
                target["unknown_usage_calls"] += 1
            target["input_tokens"] += usage.get("input_tokens") or 0
            target["output_tokens"] += usage.get("output_tokens") or 0
    for item in totals.values():
        item["estimated_cost"] = None if settings.input_price_per_million is None or settings.output_price_per_million is None or item["unknown_usage_calls"] or item["unobserved_requests"] else (item["input_tokens"] * settings.input_price_per_million + item["output_tokens"] * settings.output_price_per_million) / 1e6
        item["total_tokens"] = item["input_tokens"] + item["output_tokens"]
        item["average_tokens"] = item["total_tokens"] / item["requests"]
        item["usage_complete"] = item["unknown_usage_calls"] == 0 and item["unobserved_requests"] == 0
    items = sorted(totals.values(), key=lambda r: (-r["total_tokens"], r["intent"]))
    total = sum(r["total_tokens"] for r in items)
    for item in items:
        item["token_share"] = item["total_tokens"] / total if total and (item["total_tokens"] or item["usage_complete"]) else None
    return {"items": items, "scope": f"近 {days} 天，最多 5000 次请求；来源为应用用量台账，不是 Langfuse API 或厂商账单。金额仅按配置单价估算。",
            "langfuse_enabled": settings.langfuse_enabled and getattr(settings, "observability_enabled", True), "days": days, "truncated": truncated,
            "window_start": start.isoformat(), "window_end": end.isoformat(), "generated_at": end.isoformat(),
            "summary": {"requests": len(rows), "total_tokens": total, "intent_count": len(items),
                "highest_intent": items[0]["intent"] if total else None,
                "unknown_usage_calls": sum(r["unknown_usage_calls"] for r in items),
                "unobserved_requests": sum(r["unobserved_requests"] for r in items),
                "excluded_unlinked_empty_requests": excluded,
                "failed_calls": sum(r["failed_calls"] for r in items),
                "model_calls": sum(r["model_calls"] for r in items)}}
