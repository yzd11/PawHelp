"""Bounded tool-aware orchestration over short persistence transactions."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
import json
from random import Random
import re
from typing import Protocol

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chat import ChatEvent, StreamingChatModel, text_from_chunk
from app.config import Settings
from app.context import PreparedContext, prepare_context
from app.knowledge.embedding import BgeM3Embedder
from app.knowledge.evidence import EvidenceSufficiencyChecker
from app.knowledge.reranking import BgeReranker
from app.knowledge.search import HybridKnowledgeSearch, KnowledgeSearch
from app.knowledge.vector_store import MilvusKnowledgeStore
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository
from app.repositories.quality import QualityRepository
from app.schemas import ChatRequest
from app.services.history import to_langchain_messages
from app.services.orders import OrderData
from app.tools.business import ToolRuntime, build_business_tools
from app.tools.executor import (
    InvalidToolArguments,
    ToolExecutionFailed,
    ToolExecutor,
    UnknownToolError,
)
from app.tools.registry import ToolRegistry


class ToolCallingChatModel(StreamingChatModel, Protocol):
    """A model that can create one tool-bound streaming selector."""

    def bind_tools(
        self, tools: Sequence[BaseTool], **kwargs: object
    ) -> StreamingChatModel: ...


ToolBuilder = Callable[[ToolRuntime], list[BaseTool]]
_CITATION_PATTERN = re.compile(r"\[(\d+)]")
_KNOWLEDGE_REFUSAL = "抱歉，当前知识库没有足够证据回答这个问题，我不会猜测。"
# A refusal should not be a dead end: offer the human handoff buttons with it.
_KNOWLEDGE_REFUSAL_ACTIONS = ("转人工", "建工单")


def _raw_tool_arguments_are_strict(aggregate: AIMessageChunk) -> bool:
    """Require each accumulated raw argument string to match its parsed call."""

    calls = aggregate.tool_calls
    raw_calls = list(aggregate.tool_call_chunks)
    if len(raw_calls) != len(calls):
        return False

    for call in calls:
        call_id = call.get("id")
        raw_index = next(
            (
                index
                for index, raw_call in enumerate(raw_calls)
                if raw_call.get("id") == call_id
            ),
            0 if len(raw_calls) == 1 else None,
        )
        if raw_index is None:
            return False
        raw_call = raw_calls.pop(raw_index)
        raw_args = raw_call.get("args")
        if not isinstance(raw_args, str):
            return False
        try:
            decoded = json.loads(raw_args)
        except (TypeError, ValueError):
            return False
        if not isinstance(decoded, dict) or decoded != call.get("args"):
            return False
    return not raw_calls


def _prefer_specific_order_lookup(
    calls: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Collapse one redundant same-order lookup into the logistics call."""

    if len(calls) != 2:
        return calls
    logistics = [call for call in calls if call.get("name") == "query_logistics"]
    orders = [call for call in calls if call.get("name") == "query_order"]
    if (
        len(logistics) == 1
        and len(orders) == 1
        and logistics[0].get("args") == orders[0].get("args")
    ):
        return logistics
    return calls


@dataclass(frozen=True)
class PreparedTurn:
    """Persisted and budgeted input with no live database session."""

    conversation_id: int
    user_id: str
    user_message_id: int
    raw_question: str
    prepared_context: PreparedContext
    tool_runtime: ToolRuntime
    layered_context: bool = False
    turn_lock: asyncio.Lock | None = None


class ToolAwareChatService:
    """Select zero or one tool, then converge through the unbound model."""

    def __init__(
        self,
        *,
        settings: Settings,
        model: ToolCallingChatModel,
        session_factory: async_sessionmaker[AsyncSession],
        rng: Random | None = None,
        tool_builder: ToolBuilder = build_business_tools,
        knowledge_search: KnowledgeSearch | None = None,
        evidence_checker: EvidenceSufficiencyChecker | None = None,
        order_data: OrderData | None = None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._session_factory = session_factory
        self._rng = rng or Random()
        self._tool_builder = tool_builder
        self._orders = order_data or OrderData(session_factory)
        self._evidence_checker = evidence_checker or EvidenceSufficiencyChecker(
            model,  # type: ignore[arg-type]
            method=settings.structured_output_method,
        )
        self._knowledge_search = knowledge_search or HybridKnowledgeSearch(
            session_factory,
            BgeM3Embedder(
                settings.embedding_model,
                batch_size=settings.embedding_batch_size,
                max_length=settings.embedding_max_length,
            ),
            MilvusKnowledgeStore(
                settings.milvus_uri, settings.milvus_hybrid_collection
            ),
            BgeReranker(
                settings.reranker_model,
                batch_size=settings.reranker_batch_size,
                query_max_length=settings.reranker_query_max_length,
                passage_max_length=settings.reranker_passage_max_length,
            ),
            confidence_threshold=settings.knowledge_rerank_threshold,
            dense_limit=settings.dense_recall_limit,
            bm25_limit=settings.bm25_recall_limit,
            rerank_limit=settings.rerank_limit,
        )

    async def prepare_turn(self, request: ChatRequest) -> PreparedTurn:
        """Persist one user turn and prepare its bounded history."""

        user_id = str(request.user_id)
        async with self._session_factory() as session, session.begin():
            conversations = ConversationRepository(session)
            conversation = (
                await conversations.create(user_id)
                if request.conversation_id is None
                else await conversations.get_owned(request.conversation_id, user_id)
            )
            conversation_id = conversation.id
            messages = MessageRepository(session)
            history = await messages.list_ordered(conversation_id)
            row = await messages.add_user(conversation_id, request.message)
            history.append(row)
            prepared = self._prepare_history(to_langchain_messages(history))

        runtime = ToolRuntime(
            conversation_id,
            self._session_factory,
            self._rng,
            self._knowledge_search,
            user_id=user_id,
            order_data=self._orders,
        )
        return PreparedTurn(
            conversation_id=conversation_id,
            user_id=user_id,
            user_message_id=row.id,
            raw_question=request.message,
            prepared_context=prepared,
            tool_runtime=runtime,
        )

    def _prepare_history(self, history: list[BaseMessage]) -> PreparedContext:
        """Legacy callers retain their original bounded-history contract."""
        return prepare_context(history, shop_name=self._settings.shop_name, token_budget=self._settings.context_token_budget)

    async def stream_turn(self, turn: PreparedTurn) -> AsyncIterator[ChatEvent]:
        """Emit one complete bounded turn after its pre-stream transaction."""

        yield ChatEvent("conversation", {"conversation_id": turn.conversation_id})
        try:
            async for event in self._stream_started_turn(turn):
                yield event
        except asyncio.CancelledError:
            raise
        except Exception:
            yield self._error_event(
                turn,
                code="internal_error",
                message="服务暂时不可用，请稍后重试",
            )

    async def _stream_started_turn(
        self, turn: PreparedTurn
    ) -> AsyncIterator[ChatEvent]:
        tools = self._tool_builder(turn.tool_runtime)
        registry = ToolRegistry(tools, runtime=turn.tool_runtime)
        executor = ToolExecutor(
            registry,
            timeout_seconds=self._settings.tool_timeout_seconds,
            max_attempts=self._settings.tool_max_attempts,
        )
        aggregate: AIMessageChunk | None = None
        streamed_text: list[str] = []
        try:
            selector = self._model.bind_tools(tools, parallel_tool_calls=False)
            async for chunk in selector.astream(list(turn.prepared_context.messages)):
                aggregate = chunk if aggregate is None else aggregate + chunk
                text = text_from_chunk(chunk)
                if text:
                    streamed_text.append(text)
                    yield ChatEvent("token", {"content": text})
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception:
            yield self._error_event(
                turn,
                code="upstream_error",
                message="模型服务暂时不可用，请稍后重试",
            )
            return

        calls = [] if aggregate is None else aggregate.tool_calls
        invalid_calls = [] if aggregate is None else aggregate.invalid_tool_calls
        if invalid_calls or (
            aggregate is not None
            and calls
            and not _raw_tool_arguments_are_strict(aggregate)
        ):
            yield self._error_event(
                turn,
                code="malformed_tool_call",
                message="模型返回了无效的工具调用，请重试",
            )
            return
        calls = _prefer_specific_order_lookup(calls)
        if len(calls) > 1:
            yield self._error_event(
                turn,
                code="multiple_tool_calls",
                message="模型返回了过多的工具调用，请重试",
            )
            return
        if not calls:
            answer = "".join(streamed_text)
            try:
                message_id = await self._persist_assistant(turn.conversation_id, answer)
            except asyncio.CancelledError:
                raise
            except Exception:
                yield self._error_event(
                    turn,
                    code="persistence_error",
                    message="消息暂时无法保存，请稍后重试",
                )
                return
            yield self._done_event(turn, message_id, answer, [])
            return

        call = calls[0]
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            yield self._error_event(
                turn,
                code="malformed_tool_call",
                message="模型返回了无效的工具调用，请重试",
            )
            return
        try:
            name = call["name"]
            if not isinstance(name, str):
                raise InvalidToolArguments()
            label = registry.safe_label(name)
        except (InvalidToolArguments, UnknownToolError) as error:
            yield self._error_event(
                turn, code=error.public_code, message=error.public_message
            )
            return

        safe_status = {
            "call_id": call_id,
            "name": name,
            "label": label,
        }
        try:
            await self._persist_assistant(
                turn.conversation_id, "", tool_calls=[call]
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            yield self._error_event(
                turn,
                code="persistence_error",
                message="消息暂时无法保存，请稍后重试",
            )
            return

        yield ChatEvent("tool_status", {**safe_status, "status": "running"})
        try:
            result = await executor.execute(call)
        except asyncio.CancelledError:
            raise
        except (InvalidToolArguments, ToolExecutionFailed, UnknownToolError) as error:
            yield ChatEvent("tool_status", {**safe_status, "status": "failed"})
            yield self._error_event(
                turn, code=error.public_code, message=error.public_message
            )
            return

        try:
            await self._persist_tool(
                turn.conversation_id,
                content=result.content,
                tool_call_id=result.tool_call_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            yield ChatEvent("tool_status", {**safe_status, "status": "failed"})
            yield self._error_event(
                turn,
                code="persistence_error",
                message="消息暂时无法保存，请稍后重试",
            )
            return

        result_status = "succeeded" if result.ok else "failed"
        yield ChatEvent("tool_status", {**safe_status, "status": result_status})
        trace = [{**safe_status, "status": result_status}]
        sources: list[dict[str, object]] = []
        faq_payload: dict[str, object] | None = None
        if name == "query_faq":
            try:
                decoded_payload = json.loads(result.content)
            except (TypeError, ValueError):
                decoded_payload = None
            faq_payload = decoded_payload if isinstance(decoded_payload, dict) else None
            matches = faq_payload.get("matches") if faq_payload is not None else None
            valid_matches = matches if isinstance(matches, list) else []
            sources = self._citation_sources(valid_matches)
            confident = faq_payload is not None and faq_payload.get("confident") is True
            if not valid_matches or not confident:
                async for event in self._reject_knowledge(
                    turn,
                    trace,
                    source="retrieval_low_conf",
                    reason="混合检索没有返回达到重排置信度阈值的证据",
                ):
                    yield event
                return

            decision = await self._evidence_checker.check(
                turn.raw_question,
                [item for item in valid_matches if isinstance(item, dict)],
            )
            if not decision.useful:
                async for event in self._reject_knowledge(
                    turn,
                    trace,
                    source="self_check",
                    reason=decision.reason,
                ):
                    yield event
                return

        final_messages: list[BaseMessage] = [
            *turn.prepared_context.messages,
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(content=result.content, tool_call_id=result.tool_call_id),
        ]
        answer_parts: list[str] = []
        buffer_answer = name == "query_faq"
        try:
            async for chunk in self._model.astream(final_messages):
                text = text_from_chunk(chunk)
                if text:
                    answer_parts.append(text)
                    if not buffer_answer:
                        yield ChatEvent("token", {"content": text})
                        await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception:
            yield self._error_event(
                turn,
                code="upstream_error",
                message="模型服务暂时不可用，请稍后重试",
            )
            return

        answer = "".join(answer_parts)
        if buffer_answer:
            if not self._citations_are_valid(answer, sources):
                try:
                    await self._record_low_confidence(
                        turn,
                        source="self_check",
                        reason="生成答案缺少合法引用或包含越界引用",
                    )
                except Exception:
                    yield self._error_event(
                        turn,
                        code="persistence_error",
                        message="消息暂时无法保存，请稍后重试",
                    )
                    return
                answer = _KNOWLEDGE_REFUSAL
                sources = []
            for token in answer_parts if sources else answer:
                yield ChatEvent("token", {"content": token})
                await asyncio.sleep(0)
        try:
            message_id = await self._persist_assistant(turn.conversation_id, answer)
        except asyncio.CancelledError:
            raise
        except Exception:
            yield self._error_event(
                turn,
                code="persistence_error",
                message="消息暂时无法保存，请稍后重试",
            )
            return
        yield self._done_event(turn, message_id, answer, trace, sources)

    async def _reject_knowledge(
        self,
        turn: PreparedTurn,
        trace: list[dict[str, object]],
        *,
        source: str,
        reason: str,
    ) -> AsyncIterator[ChatEvent]:
        try:
            await self._record_low_confidence(turn, source=source, reason=reason)
            message_id = await self._persist_assistant(
                turn.conversation_id, _KNOWLEDGE_REFUSAL
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            yield self._error_event(
                turn,
                code="persistence_error",
                message="消息暂时无法保存，请稍后重试",
            )
            return
        for token in _KNOWLEDGE_REFUSAL:
            yield ChatEvent("token", {"content": token})
            await asyncio.sleep(0)
        yield self._done_event(turn, message_id, _KNOWLEDGE_REFUSAL, trace, [])

    async def _record_low_confidence(
        self, turn: PreparedTurn, *, source: str, reason: str,
        retrieved_chunks: list[dict] | None = None,
    ) -> None:
        async with self._session_factory() as session, session.begin():
            await QualityRepository(session).add_low_confidence(
                conversation_id=turn.conversation_id,
                raw_question=turn.raw_question,
                source=source,
                reason=reason,
                retrieved_chunks=retrieved_chunks,
                event_key=f"{source}:{turn.user_message_id}",
            )

    @staticmethod
    def _citation_sources(matches: list[object]) -> list[dict[str, object]]:
        sources: list[dict[str, object]] = []
        for item in matches:
            if not isinstance(item, dict):
                continue
            number = item.get("citation")
            chunk_id = item.get("chunk_id")
            category = item.get("category")
            if not isinstance(number, int) or number < 1:
                continue
            if not isinstance(chunk_id, int) or chunk_id < 1:
                continue
            if not isinstance(category, str):
                continue
            section_path = item.get("section_path")
            sources.append(
                {
                    "number": number,
                    "chunk_id": chunk_id,
                    "category": category,
                    "section_path": section_path if isinstance(section_path, str) else None,
                }
            )
        return sorted(sources, key=lambda item: int(item["number"]))

    @staticmethod
    def _citations_are_valid(
        answer: str, sources: Sequence[dict[str, object]]
    ) -> bool:
        allowed = {source.get("number") for source in sources}
        cited = {int(value) for value in _CITATION_PATTERN.findall(answer)}
        return bool(cited) and cited.issubset(allowed)

    async def _persist_assistant(
        self,
        conversation_id: int,
        content: str,
        *,
        tool_calls: list[dict[str, object]] | None = None,
    ) -> int:
        async with self._session_factory() as session, session.begin():
            row = await MessageRepository(session).add_assistant(
                conversation_id,
                content,
                tool_calls=tool_calls,
            )
        return row.id

    async def _persist_tool(
        self, conversation_id: int, *, content: str, tool_call_id: str
    ) -> int:
        async with self._session_factory() as session, session.begin():
            row = await MessageRepository(session).add_tool(
                conversation_id, content, tool_call_id
            )
        return row.id

    def _error_event(
        self, turn: PreparedTurn, *, code: str, message: str
    ) -> ChatEvent:
        return ChatEvent("error", {"code": code, "message": message})

    @staticmethod
    def _done_event(
        turn: PreparedTurn,
        message_id: int,
        content: str,
        tool_trace: list[dict[str, object]],
        sources: list[dict[str, object]] | None = None,
    ) -> ChatEvent:
        return ChatEvent(
            "done",
            {
                "content": content,
                "conversation_id": turn.conversation_id,
                "message_id": message_id,
                "input_tokens_estimate": turn.prepared_context.input_tokens_estimate,
                "trimmed_messages": turn.prepared_context.trimmed_messages,
                "tool_trace": tool_trace,
                "sources": sources or [],
            },
        )


__all__ = [
    "PreparedTurn",
    "ToolAwareChatService",
]
