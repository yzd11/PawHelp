"""Deterministic conversation workflow around the tool-aware agent node."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
import json
import logging
from pathlib import Path
from typing import Annotated, TypedDict
from weakref import WeakValueDictionary

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from app.chat import ChatEvent
from app.context import ContextBudgetExceeded, PreparedContext
from app.knowledge.confidence import EvidenceGate
from app.knowledge.evidence import EvidenceSufficiencyChecker
from app.knowledge.pipeline import KnowledgeEvidencePipeline, QuestionSplitter
from app.observability import Observability, finish_trace
from app.prompts import CUSTOMER_SERVICE_SYSTEM_TEMPLATE
from app.schemas import ChatRequest
from app.services.tool_chat import PreparedTurn, ToolAwareChatService
from app.tools.business import ToolRuntime
from app.tools.catalog import ToolCatalog
from app.tools.registry import ToolRegistry
from app.workflow.memory import SessionMemory
from app.workflow.nodes import TurnNodes, after_agent, after_confidence_gate, after_route, after_select_order
from app.workflow.tool_agent import ToolAgentNodes, ticket_requested
from app.workflow.titles import ConversationTitles
from app.workflow.understanding import Intent, Route


logger = logging.getLogger(__name__)


class PendingWorkflowAction(ValueError):
    """An unfinished confirmation must not be overwritten by a new turn."""


_AGENT_RULES = """You are the shop's customer-service agent. Use only supplied evidence and trusted tool observations.
Ask one concise follow-up if an order number or product name needed for a tool is missing.
Only propose create_ticket if the customer explicitly requests creating a ticket. Ask for a concrete problem description if missing; never invent one. The engine requires frontend confirmation before writing. A handoff request alone never authorizes a ticket. Do not claim a handoff occurred; offer the frontend option instead. Tool use is bounded by the server.
If the customer explicitly asks to check both order status and logistics, call both tools fresh in this turn; prior-turn results are not new lookups.
For knowledge answers cite each factual policy/product claim with [n] from the supplied evidence.
Do not ask the customer for a refund reason in chat; the user selects it only when submitting the refund form.
When human handling is appropriate, append a final line [建议选项:转人工] or [建议选项:建工单] or [建议选项:转人工,建工单].
Do not append that line unless appropriate. Do not promise outcomes or arrival times unsupported by evidence.
For returns/after-sales, order status alone never proves return eligibility. State missing conditions clearly; do not submit a refund without the user's explicit form submission.
If retrieved evidence cannot answer the user's knowledge question, do not invent an answer. Emit a final line [证据不足:具体原因] and give a brief fallback. This marker is processed by the server and not shown to the user.
Order/logistics tools read persisted, user-owned data. For logistics first use query_order to obtain tracking_no, then call query_logistics with it. An unpaid/unshipped order has no tracking number; explain this without inventing a shipment.
Logistics and aftersales MCP tools are explicitly demonstration services, not real purchase or refund systems. An unavailable tool must not be simulated or described as a completed lookup.
The order's delivered_at is the signing date for policy calculations. Dates ending in Z are UTC; use the current date and the user's local timezone when explaining elapsed days. Times labelled （香港/北京时间） are already local: copy them verbatim and never convert or re-annotate them. Do not claim a demo record is a real purchase."""


class WorkflowState(TypedDict, total=False):
    raw_question: str
    user_message_id: int
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    context_version: int
    history_messages: list[BaseMessage]
    history_summary: str
    understanding_messages: list[BaseMessage]
    layered_context: bool
    intent: Intent
    intent_confidence: float
    task_kind: str
    required_tools: list[str]
    reference_status: str
    clarification_question: str
    routing_decision: str
    routing_reason: str
    needs_order: bool
    route: Route
    order_id: str
    order_observation: str
    refund_available: bool
    queries: list[str]
    evidence: list[dict[str, object]]
    retrieved_chunks: list[dict] | None
    evidence_confidence: dict
    confident: bool
    answer: str
    sources: list[dict[str, object]]
    trace: list[dict[str, object]]
    tool_records: list[tuple[dict[str, object], str]]
    actions: list[str]
    message_id: int
    agent_context: list[BaseMessage]
    pending_calls: list[dict]
    agent_steps: int
    agent_reminders: int
    agent_next: str
    ticket_requested: bool


class WorkflowChatService(ToolAwareChatService):
    """Keep the existing chat API; swap its internal coordinator for a graph."""

    def __init__(self, *args: object, checkpointer: BaseCheckpointSaver | None = None, catalog=None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._checkpointer = checkpointer or InMemorySaver()
        self._observability = Observability(self._settings, self._session_factory)
        self._evidence_gate = EvidenceGate(self._settings, Path(__file__).resolve().parents[2] / "knowledge")
        self._turn_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._catalog = catalog or ToolCatalog(self._settings, builder=self._tool_builder)
        runtime = ToolRuntime(0, self._session_factory, self._rng, self._knowledge_search, user_id="", order_data=self._orders)
        tools = [tool for tool in self._tool_builder(runtime) if tool.name not in {"query_logistics", "query_faq"}]
        self._startup_registry = ToolRegistry(tools)
        system = CUSTOMER_SERVICE_SYSTEM_TEMPLATE.format(shop_name=self._settings.shop_name) + "\n" + _AGENT_RULES
        self._titles = ConversationTitles(self._settings, self._session_factory, self._model)
        self._memory = SessionMemory(self._settings, self._session_factory, self._model, system,
                                     json.dumps([convert_to_openai_tool(tool) for tool in tools], ensure_ascii=False))
        self._evidence_pipeline = KnowledgeEvidencePipeline(
            self._knowledge_search, self._evidence_gate,
            EvidenceSufficiencyChecker(self._model, method=self._settings.structured_output_method),
            QuestionSplitter(self._model, method=self._settings.structured_output_method),
            top_k=self._settings.rerank_top_k,
            bound_text=lambda text: self._memory.bound_text(text, self._settings.evidence_chunk_max_tokens),
        )

    def _prepare_history(self, history: list[BaseMessage]) -> PreparedContext:
        return PreparedContext(tuple(history), self._memory.count(history), 0)

    def is_summarizing(self, conversation_id: int) -> bool:
        return self._memory.is_summarizing(conversation_id)

    async def prepare_turn(self, request: ChatRequest) -> PreparedTurn:
        if (not self._settings.model_context_window and not self._memory.window_known) or self._memory.count([HumanMessage(content=request.message)]) > self._settings.max_user_input_tokens or self._memory.budget.history <= 0:
            raise ContextBudgetExceeded()
        key = f"{request.user_id}:{request.conversation_id or 'new'}"
        lock = self._turn_locks.setdefault(key, asyncio.Lock())
        await lock.acquire()
        try:
            if request.conversation_id and await self.pending_ticket(str(request.user_id), request.conversation_id):
                raise PendingWorkflowAction("confirm or cancel the pending ticket first")
            turn = await super().prepare_turn(request)
            self._turn_locks[f"{turn.user_id}:{turn.conversation_id}"] = lock
            return replace(turn, layered_context=True, turn_lock=lock)
        except BaseException:
            lock.release()
            raise

    async def close(self) -> None:
        await self._memory.close()
        await self._titles.close()
        await self._observability.close()

    def set_checkpointer(self, checkpointer: BaseCheckpointSaver) -> None:
        self._checkpointer = checkpointer

    def _build_graph(self, turn: PreparedTurn):
        """Compile this turn's graph; nodes carry the turn, the graph only routes."""
        nodes = TurnNodes(self, turn)
        agent = ToolAgentNodes(self, turn, nodes.resources)
        builder = StateGraph(WorkflowState)
        for name, node in {
            "resolve_reference": nodes.resolve_reference, "classify_intent": nodes.classify_intent, "route": nodes.route,
            "select_order": nodes.select_order, "retrieve": nodes.retrieve, "confidence_gate": nodes.confidence_gate,
            "agent": agent.step, "execute_tools": agent.execute,
            "complaint": nodes.complaint, "smalltalk": nodes.smalltalk, "other": nodes.other, "log": nodes.log_result,
        }.items():
            builder.add_node(name, node)
        builder.add_edge(START, "resolve_reference")
        builder.add_edge("resolve_reference", "classify_intent")
        builder.add_edge("classify_intent", "route")
        builder.add_conditional_edges("route", after_route, {
            "knowledge": "retrieve", "order_knowledge": "select_order", "business": "agent", "select_order": "select_order",
            "complaint": "complaint", "smalltalk": "smalltalk", "other": "other",
        })
        builder.add_conditional_edges("select_order", after_select_order)
        builder.add_edge("retrieve", "confidence_gate")
        builder.add_conditional_edges("confidence_gate", after_confidence_gate, {"agent": "agent", "log": "log"})
        builder.add_conditional_edges("agent", after_agent, {"agent": "agent", "execute_tools": "execute_tools", "log": "log"})
        builder.add_conditional_edges("execute_tools", after_agent, {"agent": "agent", "log": "log"})
        for node in ("complaint", "smalltalk", "other"):
            builder.add_edge(node, "log")
        builder.add_edge("log", END)
        return builder.compile(checkpointer=self._checkpointer), nodes

    async def _persist_workflow_answer(
        self, conversation_id: int, answer: str, actions: list[str], sources: list[dict[str, object]],
        *, related_order_id: str | None = None,
        tool_trace: list[dict[str, object]] | None = None,
        reply_to_message_id: int | None = None, retrieved_chunks: list[dict] | None = None,
    ) -> int:
        from app.repositories.messages import MessageRepository

        async with self._session_factory() as session, session.begin():
            row = await MessageRepository(session).add_assistant(
                conversation_id, answer, suggested_actions=actions, source_refs=sources,
                related_order_id=related_order_id,
                tool_trace=tool_trace,
                reply_to_message_id=reply_to_message_id, retrieved_chunks=retrieved_chunks,
            )
        return row.id

    async def stream_turn(self, turn: PreparedTurn) -> AsyncIterator[ChatEvent]:
        try:
            async for event in self._stream_turn_locked(turn):
                yield event
        finally:
            if turn.turn_lock is not None and turn.turn_lock.locked():
                turn.turn_lock.release()

    async def _stream_turn_locked(self, turn: PreparedTurn) -> AsyncIterator[ChatEvent]:
        yield ChatEvent("conversation", {"conversation_id": turn.conversation_id})
        graph, nodes = self._build_graph(turn)
        config = {"configurable": {"thread_id": f"{turn.user_id}:{turn.conversation_id}"}}
        checkpoint = await self._checkpointer.aget_tuple(config)
        values = checkpoint.checkpoint.get("channel_values", {}) if checkpoint else {}
        incoming = [m for m in turn.prepared_context.messages if not isinstance(m, SystemMessage)]
        if values.get("context_version") != 7:
            # One-time bootstrap also migrates old overwrite-style checkpoints from canonical DB history.
            delta = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *incoming]
        else:
            ids = {m.id for m in values.get("messages", [])}
            delta = [m for m in incoming if m.id not in ids]
        initial: WorkflowState = {
            "raw_question": turn.raw_question,
            "user_message_id": turn.user_message_id,
            "question": turn.raw_question,
            "messages": delta, "context_version": 7, "layered_context": turn.layered_context,
            "evidence": [], "sources": [], "trace": [], "tool_records": [], "actions": [],
            "retrieved_chunks": None, "evidence_confidence": {},
            "answer": "", "confident": False,
            "intent": "其他", "intent_confidence": 0.0, "task_kind": "unspecified", "required_tools": [], "reference_status": "clear",
            "clarification_question": "", "routing_decision": "execute", "routing_reason": "", "needs_order": False,
            "order_id": "", "order_observation": "", "refund_available": False, "queries": [],
            "agent_context": [], "pending_calls": [], "agent_steps": 0, "agent_reminders": 0, "agent_next": "agent",
            "ticket_requested": ticket_requested(turn.raw_question, bool(values.get("ticket_requested", False))),
        }
        async for event in self._run_graph(graph, nodes, initial, turn):
            yield event

    async def _run_graph(self, graph, nodes: TurnNodes, graph_input: object, turn: PreparedTurn) -> AsyncIterator[ChatEvent]:
        with self._observability.request(turn) as record:
            try:
                async for event in self._run_observed_graph(graph, nodes, graph_input, turn, record):
                    if event.name == "error": record["status"] = "failed"
                    yield event
            except BaseException:
                record["status"] = "cancelled"
                raise
            finally:
                await self._observability.save(record, turn)

    async def _run_observed_graph(self, graph, nodes: TurnNodes, graph_input, turn, record):
        config = {"configurable": {"thread_id": f"{turn.user_id}:{turn.conversation_id}"}}
        try:
            async for item in graph.astream(  # type: ignore[attr-defined]
                graph_input, config={**config, "recursion_limit": 50, "callbacks": record["callbacks"], "metadata": {"conversation_id": turn.conversation_id, "user_message_id": turn.user_message_id}}, stream_mode="custom", durability="sync",
            ):
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    yield ChatEvent(item["name"], item.get("data", {}))
            snapshot = await graph.aget_state(config)  # type: ignore[attr-defined]
            finish_trace(record, snapshot.values)
            if snapshot.tasks: record["status"] = "interrupted"
            if turn.layered_context and not snapshot.tasks and snapshot.values.get("message_id"):
                memory = nodes.memory
                view = await memory.prepare(turn.conversation_id, turn.user_id, snapshot.values["messages"], snapshot.values["user_message_id"])
                memory.schedule(turn.conversation_id, turn.user_id, view)
            # Greetings make poor titles; wait for the first substantive question.
            if not snapshot.tasks and snapshot.values.get("message_id") and snapshot.values.get("intent") != "闲聊":
                self._titles.schedule(turn.conversation_id, turn.user_id, snapshot.values["raw_question"])
            for task in snapshot.tasks:
                for pending in task.interrupts:
                    if isinstance(pending.value, dict) and pending.value.get("kind") == "select_order":
                        yield ChatEvent("order_selection", pending.value)
                    elif isinstance(pending.value, dict) and pending.value.get("kind") == "confirm_ticket":
                        yield ChatEvent("ticket_confirmation", pending.value)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("workflow workflow failed conversation=%s", turn.conversation_id)
            yield self._error_event(turn, code="workflow_error", message="客服暂时无法处理，请稍后重试")

    async def _pending_order_context(self, user_id: str, conversation_id: int, kind="select_order"):
        config = {"configurable": {"thread_id": f"{user_id}:{conversation_id}"}}
        checkpoint = await self._checkpointer.aget_tuple(config)
        if checkpoint is None:
            return None
        values = checkpoint.checkpoint.get("channel_values", {})
        if not isinstance(values, dict) or not isinstance(values.get("raw_question"), str):
            return None
        messages = values.get("messages")
        if not isinstance(messages, list):
            return None
        raw_question = values["raw_question"]
        prepared = PreparedContext(
            tuple(messages), count_tokens_approximately(messages), 0,
        )
        turn = PreparedTurn(
            conversation_id=conversation_id, user_id=user_id, user_message_id=int(values.get("user_message_id", 0)),
            raw_question=raw_question, prepared_context=prepared,
            tool_runtime=ToolRuntime(conversation_id, self._session_factory, self._rng, self._knowledge_search, user_id=user_id, order_data=self._orders),
            layered_context=bool(values.get("layered_context", False)),
        )
        graph, nodes = self._build_graph(turn)
        snapshot = await graph.aget_state(config)
        for task in snapshot.tasks:
            for pending in task.interrupts:
                if isinstance(pending.value, dict) and pending.value.get("kind") == kind:
                    return turn, (graph, nodes), pending.value
        return None

    async def pending_ticket(self, user_id: str, conversation_id: int):
        context = await self._pending_order_context(user_id, conversation_id, "confirm_ticket")
        return context[2] if context else None

    async def resume_ticket(self, user_id: str, conversation_id: int, proposal_id: str, message_id: int, confirmed: bool):
        lock = self._turn_locks.setdefault(f"{user_id}:{conversation_id}", asyncio.Lock())
        async with lock:
            context = await self._pending_order_context(user_id, conversation_id, "confirm_ticket")
            if context is None or context[2].get("message_id") != message_id or context[2].get("proposal_id") != proposal_id:
                raise ValueError("no matching pending ticket")
            turn, (graph, nodes), _ = context
            yield ChatEvent("conversation", {"conversation_id": conversation_id})
            async for event in self._run_graph(graph, nodes, Command(resume={"proposal_id": proposal_id, "confirmed": confirmed}), turn):
                yield event

    async def pending_order(self, user_id: str, conversation_id: int) -> dict[str, object] | None:
        context = await self._pending_order_context(user_id, conversation_id)
        return None if context is None else context[2]

    async def resume_order(self, user_id: str, conversation_id: int, order_id: str, message_id: int) -> AsyncIterator[ChatEvent]:
        lock = self._turn_locks.setdefault(f"{user_id}:{conversation_id}", asyncio.Lock())
        async with lock:
            async for event in self._resume_order_locked(user_id, conversation_id, order_id, message_id):
                yield event

    async def _resume_order_locked(self, user_id: str, conversation_id: int, order_id: str, message_id: int) -> AsyncIterator[ChatEvent]:
        context = await self._pending_order_context(user_id, conversation_id)
        if context is None or context[2].get("message_id") != message_id or order_id not in {
            str(order["order_id"]) for order in context[2].get("orders", [])
        }:
            raise ValueError("no matching pending order")
        await self._orders.get_order(user_id, order_id)
        turn, (graph, nodes), _ = context
        yield ChatEvent("conversation", {"conversation_id": conversation_id})
        async for event in self._run_graph(graph, nodes, Command(resume=order_id), turn):
            yield event
