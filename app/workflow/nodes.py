"""Per-turn graph nodes: understanding, routing, order binding, retrieval and logging.

The tool-agent nodes live in ``tool_agent``; ``graph`` only wires these together.
"""

import json
import logging
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from app.context import ContextBudgetExceeded
from app.knowledge.search import RetrievalStrategy
from app.observability import record_intent
from app.repositories.orders import OrderNotFound
from app.services.tool_chat import PreparedTurn, _KNOWLEDGE_REFUSAL, _KNOWLEDGE_REFUSAL_ACTIONS
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.workflow.understanding import (
    IntentDecision, ResolvedQuestion, classify_intent, expand_queries, explicit_order_ambiguity, explicit_order_id,
    explicit_order_ids, order_like_numbers, plan_route, resolve_question, understanding_history,
)

logger = logging.getLogger(__name__)
_POLICY_INTENTS = {"退款退货", "售后"}
# Only these intents can require a fresh order + logistics lookup in one turn.
_LOOKUP_INTENTS = {"订单", "物流"}


class TurnNodes:
    """Graph nodes bound to one prepared turn and its service dependencies."""

    def __init__(self, service, turn: PreparedTurn) -> None:
        self.service, self.turn = service, turn
        self._registry = None
        self._memory = None

    @property
    def memory(self):
        """The tool-budgeted memory once loaded; the service default before that."""
        return self._memory or self.service._memory

    async def resources(self):
        if self._registry is None:
            registry = await self.service._catalog.snapshot(self.turn.tool_runtime)
            memory = self.service._memory.for_tools(registry.tools)
            if self.turn.layered_context and memory.budget.history <= 0:
                raise ContextBudgetExceeded()
            self._registry, self._memory = registry, memory
        return self._registry, self._memory

    async def resolve_reference(self, state) -> dict[str, object]:
        service, turn = self.service, self.turn
        logger.info("workflow node=resolve_reference conversation=%s", turn.conversation_id)
        _, memory = await self.resources()
        if turn.layered_context:
            view = await memory.prepare(turn.conversation_id, turn.user_id, state["messages"], state["user_message_id"])
            history = list(view.messages)
            summary = view.summary
        else:
            history = [m for m in state["messages"] if not isinstance(m, SystemMessage) and m.content != state["raw_question"]]
            summary = ""
        shared = understanding_history(history, state["raw_question"], summary=summary,
                                       collecting_ticket=bool(state.get("ticket_requested")))
        service._memory.log_context("history_ctx", turn.conversation_id, shared, summary, turn_id=state["user_message_id"])
        resolution = await self._resolve(state["raw_question"], shared)
        logger.info("workflow resolved question=%s reference_status=%s", resolution.question, resolution.reference_status)
        return {"question": resolution.question, "reference_status": resolution.reference_status,
                "clarification_question": resolution.clarification_question,
                "history_messages": history, "history_summary": summary, "understanding_messages": shared}

    async def _resolve(self, raw: str, shared) -> ResolvedQuestion:
        """Let the model rewrite the turn, then discard rewrites the customer did not say."""
        service = self.service
        try:
            resolution = await resolve_question(service._model, service._settings.structured_output_method, raw, shared)
            known_numbers = order_like_numbers("\n".join(str(m.content) for m in shared if isinstance(m, HumanMessage)))
            too_long = service._memory.count([HumanMessage(content=resolution.question)]) > service._settings.max_user_input_tokens
            if not order_like_numbers(resolution.question).issubset(known_numbers) or too_long:
                logger.warning("workflow resolver invented an order-like number; using raw question")
                resolution = ResolvedQuestion(question=raw)
            if resolution.reference_status == "ambiguous":
                resolution = resolution.model_copy(update={"question": raw})
        except Exception:
            logger.exception("workflow reference resolution failed conversation=%s", self.turn.conversation_id)
            resolution = ResolvedQuestion(question=raw)
        requested_ids = explicit_order_ids(raw)
        if len(requested_ids) == 1 and explicit_order_ids(resolution.question) != requested_ids:
            # A number known from history is not permission to replace the
            # customer's explicit current target. Drop the unsafe rewrite.
            resolution = resolution.model_copy(update={"question": raw})
            logger.info("workflow current_order_binding preserved order=%s", requested_ids[0])
        clarification = explicit_order_ambiguity(raw)
        if clarification:
            # Inspect the raw turn, not an LLM rewrite that may silently pick
            # one of the orders. Selection still uses the owned-order cards.
            resolution = ResolvedQuestion(question=raw, reference_status="ambiguous", clarification_question=clarification)
            logger.info("workflow ambiguity_guard conversation=%s reason=multiple_explicit_orders", self.turn.conversation_id)
        return resolution

    async def classify_intent(self, state) -> dict[str, object]:
        service = self.service
        logger.info("workflow node=classify_intent conversation=%s", self.turn.conversation_id)
        try:
            decision = await classify_intent(
                service._model, service._settings.structured_output_method, state["question"],
                state.get("understanding_messages", []),
            )
            record_intent(decision.intent)
            return {"intent": decision.intent, "intent_confidence": decision.confidence,
                    "task_kind": decision.task_kind,
                    "required_tools": list(decision.requested_lookups) if decision.intent in _LOOKUP_INTENTS else [],
                    "clarification_question": state.get("clarification_question") or decision.clarification_question}
        except Exception:
            logger.exception("workflow intent parsing failed conversation=%s", self.turn.conversation_id)
            return {"intent": "其他", "intent_confidence": 0.0}

    async def route(self, state) -> dict[str, object]:
        collecting_ticket = bool(state.get("ticket_requested")) and state["intent"] in {"人工", "售后", "投诉"}
        decision = IntentDecision(intent=state["intent"], confidence=state["intent_confidence"],
                                  task_kind=state.get("task_kind", "unspecified"),
                                  clarification_question=state.get("clarification_question", ""))
        resolution = ResolvedQuestion(question=state["question"], reference_status=state.get("reference_status", "clear"),
                                      clarification_question=state.get("clarification_question", ""))
        plan = plan_route(decision, resolution, collecting_ticket=collecting_ticket)
        conversation_id = self.turn.conversation_id
        logger.info("workflow node=route intent=%s destination=%s conversation=%s", state["intent"], plan.route, conversation_id)
        logger.info("routing_decision conversation=%s raw=%s resolved=%s intent=%s confidence=%s task_kind=%s decision=%s reason=%s needs_order=%s",
                    conversation_id, json.dumps(state["raw_question"], ensure_ascii=False),
                    json.dumps(state["question"], ensure_ascii=False), state["intent"], state["intent_confidence"],
                    decision.task_kind, plan.decision, plan.reason, plan.needs_order)
        return {"route": plan.route, "routing_decision": plan.decision, "routing_reason": plan.reason,
                "needs_order": plan.needs_order, "clarification_question": plan.clarification,
                "ticket_requested": collecting_ticket and plan.route == "business" and (
                    state["intent"] == "人工" or plan.reason == "ticket_description_continuation")}

    async def select_order(self, state) -> dict[str, object]:
        """Ask once, then resume this same graph after an explicit card click."""

        service, turn = self.service, self.turn
        logger.info("workflow node=select_order conversation=%s", turn.conversation_id)
        ambiguous_order = state.get("routing_reason") == "ambiguous_order"
        order_id = None if ambiguous_order else explicit_order_id(state["question"])
        choices = await service._orders.list_orders(turn.user_id)
        valid_ids = {str(order["order_id"]) for order in choices}
        if order_id not in valid_ids:
            selected = interrupt({
                "kind": "select_order",
                "message_id": state["user_message_id"],
                "message": "请选择要查询或处理的订单；标注为演示订单的记录是测试数据。",
                "orders": choices,
            })
            order_id = str(selected)
            if order_id not in valid_ids:
                raise ValueError("invalid selected order")
        call, content, ok, status = await self._prefetch_order(order_id)
        observation = json.loads(content)
        if observation.get("code") == "order_not_found":
            raise OrderNotFound("没有找到您的这笔订单")
        if ambiguous_order:
            question = f"针对用户本次明确选择的订单 {order_id}：{state['question']}"
        elif explicit_order_id(state["question"]) == order_id:
            question = state["question"]
        else:
            question = f"{state['question']}（订单 {order_id}）"
        turn_tag = {"turn_message_id": state["user_message_id"]}
        return {
            "question": question,
            "order_id": order_id,
            "order_observation": content,
            "refund_available": ok and observation.get("status") in {"paid", "processing", "shipped", "delivered"},
            "trace": [status],
            "tool_records": [(call, content)],
            "messages": [
                AIMessage(content="", tool_calls=[call], id=f"prefetch-ai-{call['id']}", additional_kwargs=turn_tag),
                ToolMessage(content=content, tool_call_id=call["id"], name="query_order", id=f"prefetch-result-{call['id']}", additional_kwargs=turn_tag),
            ],
        }

    async def _prefetch_order(self, order_id: str):
        """Run query_order through the normal executor and stream its status."""
        service, turn = self.service, self.turn
        tool = next(tool for tool in service._tool_builder(turn.tool_runtime) if tool.name == "query_order")
        executor = ToolExecutor(
            ToolRegistry([tool], runtime=turn.tool_runtime),
            timeout_seconds=service._settings.tool_timeout_seconds,
            max_attempts=service._settings.tool_max_attempts,
        )
        call = {"id": f"order-prefetch-{uuid4().hex}", "name": "query_order", "args": {"order_id": order_id}}
        writer = get_stream_writer()
        status = {"call_id": call["id"], "name": "query_order", "label": "查询订单"}
        writer({"name": "tool_status", "data": {**status, "status": "running"}})
        try:
            result = await executor.execute(call)
        except Exception:
            writer({"name": "tool_status", "data": {**status, "status": "failed"}})
            raise
        result_status = "succeeded" if result.ok else "failed"
        writer({"name": "tool_status", "data": {**status, "status": result_status}})
        return call, result.content, result.ok, {**status, "status": result_status}

    async def retrieve(self, state) -> dict[str, object]:
        service, conversation_id = self.service, self.turn.conversation_id
        logger.info("workflow node=retrieve intent=%s conversation=%s", state["intent"], conversation_id)
        kwargs: dict[str, object] = {"strategy": RetrievalStrategy.HYBRID_RERANK, "limit": service._settings.rerank_top_k}
        queries = [state["question"]]
        if state["intent"] in _POLICY_INTENTS:
            kwargs["content_types"] = ("policy",)
            queries = await self._policy_queries(state)
        retrieval = await service._evidence_pipeline.retrieve(state["question"], extra_queries=queries, **kwargs)
        matches = retrieval.result.matches
        evidence = service._evidence_pipeline.evidence(matches)
        sources = [{key: item[key] for key in ("number", "chunk_id", "category", "section_path")} for item in evidence]
        snapshot = [{**item, "answer": match.chunk.answer} for item, match in zip(evidence, matches)][:service._settings.observation_snapshot_limit]
        if not retrieval.confidence["calibrated"]:
            logger.warning("evidence gate uncalibrated conversation=%s reason=%s", conversation_id, retrieval.confidence["reason"])
        return {
            "queries": retrieval.queries, "evidence": evidence, "sources": sources,
            "retrieved_chunks": snapshot,
            "confident": retrieval.result.confident,
            "evidence_confidence": retrieval.confidence,
        }

    async def _policy_queries(self, state) -> list[str]:
        """Expand policy questions from several angles, anchored to the bound order."""
        query = state["question"]
        if state.get("order_id") and not explicit_order_id(query):
            query += f"（订单 {state['order_id']}）"
        try:
            return await expand_queries(self.service._model, self.service._settings.structured_output_method, query)
        except Exception:
            logger.exception("workflow query expansion failed conversation=%s", self.turn.conversation_id)
            return [query]

    async def confidence_gate(self, state) -> dict[str, object]:
        logger.info("workflow node=confidence_gate confident=%s conversation=%s", state["confident"], self.turn.conversation_id)
        if state["confident"]:
            return {}
        await self.service._record_low_confidence(
            self.turn, source="retrieval_low_conf", reason=json.dumps(state.get("evidence_confidence", {}), ensure_ascii=False),
            retrieved_chunks=state.get("retrieved_chunks"),
        )
        return {"answer": _KNOWLEDGE_REFUSAL, "sources": [], "actions": list(_KNOWLEDGE_REFUSAL_ACTIONS)}

    async def complaint(self, state) -> dict[str, object]:
        logger.info("workflow node=complaint conversation=%s", self.turn.conversation_id)
        return {
            "answer": "很抱歉给您带来不好的体验。您可以选择转人工客服，或提交工单，我们会认真处理。",
            "actions": ["转人工", "建工单"],
        }

    async def smalltalk(self, state) -> dict[str, object]:
        logger.info("workflow node=smalltalk conversation=%s", self.turn.conversation_id)
        return {"answer": "您好，我是客服小猫！请问有什么可以帮您的？"}

    async def other(self, state) -> dict[str, object]:
        return {"answer": state.get("clarification_question") or "请补充您要查询的订单、商品或具体问题，我再帮您处理。"}

    async def log_result(self, state) -> dict[str, object]:
        service, turn = self.service, self.turn
        logger.info("workflow node=log conversation=%s", turn.conversation_id)
        answer = state["answer"]
        actions = state.get("actions", [])
        order_id = state.get("order_id") if "申请退款" in actions else None
        message_id = await service._persist_workflow_answer(
            turn.conversation_id, answer, actions, state.get("sources", []),
            related_order_id=order_id, tool_trace=state.get("trace", []),
            reply_to_message_id=turn.user_message_id, retrieved_chunks=state.get("retrieved_chunks"),
        )
        writer = get_stream_writer()
        # Agent answers were already streamed; fixed replies and refusals were not.
        if state["route"] not in {"business", "knowledge", "order_knowledge"}:
            writer({"name": "token", "data": {"content": answer}})
        elif state["route"] in {"knowledge", "order_knowledge"} and not state.get("confident", False):
            writer({"name": "token", "data": {"content": answer}})
        done = service._done_event(turn, message_id, answer, state.get("trace", []), state.get("sources", []))
        writer({"name": "done", "data": {**done.data, "suggested_actions": actions, "related_order_id": order_id}})
        tags = {"db_message_id": message_id, "turn_message_id": state["user_message_id"]}
        return {"message_id": message_id, "messages": [AIMessage(content=answer, id=f"db-{message_id}", additional_kwargs=tags)]}


def after_route(state) -> str:
    if state["route"] == "business" and state.get("routing_decision") == "select_order":
        return "select_order"
    return state["route"]


def after_select_order(state) -> str:
    return "retrieve" if state["route"] == "order_knowledge" else "agent"


def after_confidence_gate(state) -> str:
    return "agent" if state["confident"] else "log"


def after_agent(state) -> str:
    return state["agent_next"]
