"""Core chat orchestration through its public service seam."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.database.models import Conversation, KnowledgeChunk, LowConfidenceQuestion, Message, Ticket
from app.knowledge.evidence import EvidenceDecision
from app.knowledge.query_understanding import QueryPlan
from app.knowledge.search import KnowledgeMatch, RetrievalResult, RetrievalStrategy
from app.schemas import ChatRequest
from app.services.tool_chat import ToolAwareChatService
from tests.fakes import FakeToolCallingChatModel


class FixedEvidenceChecker:
    def __init__(self, useful: bool = True) -> None:
        self.useful = useful

    async def check(self, question, evidence):
        return EvidenceDecision(useful=self.useful, reason="证据足够" if self.useful else "缺少答案")


class FixedKnowledgeSearch:
    def __init__(self, *, confident: bool) -> None:
        self.confident = confident

    async def retrieve(self, query, *, limit=10, **kwargs):
        row = KnowledgeChunk(
            id=7,
            questions="配送费用如何计算？",
            answer="配送费用会在结算页展示。",
            category="配送服务",
            section_path="配送服务 / 配送费用",
            content_type="faq",
            vectorize_status="done",
        )
        matches = (KnowledgeMatch(row, 0.93),) if self.confident else ()
        return RetrievalResult(
            matches,
            QueryPlan(normalized_query=query),
            RetrievalStrategy.HYBRID_RERANK,
            self.confident,
        )


def faq_selector(call_id: str = "faq-1") -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "query_faq",
                "args": '{"keyword":"邮费是多少"}',
                "id": call_id,
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )


def _settings() -> Settings:
    return Settings(
        chat_base_url="http://localhost:11434/v1",
        openai_model="qwen3",
        openai_api_key="ollama",
        database_url="mysql+asyncmy://unused",
        shop_name="测试店",
        tool_timeout_seconds=1,
        tool_max_attempts=1,
    )


@pytest.fixture
async def database(mysql_session):
    factory = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    conversation_ids: list[int] = []
    yield factory, conversation_ids
    if conversation_ids:
        async with factory() as session, session.begin():
            await session.execute(
                delete(Message).where(Message.conversation_id.in_(conversation_ids))
            )
            await session.execute(
                delete(Ticket).where(Ticket.conversation_id.in_(conversation_ids))
            )
            await session.execute(
                delete(LowConfidenceQuestion).where(
                    LowConfidenceQuestion.conversation_id.in_(conversation_ids)
                )
            )
            await session.execute(
                delete(Conversation).where(Conversation.id.in_(conversation_ids))
            )


async def _events(service: ToolAwareChatService, request: ChatRequest):
    turn = await service.prepare_turn(request)
    events = [event async for event in service.stream_turn(turn)]
    return turn, events


async def test_new_and_followup_turns_share_database_generated_context(database) -> None:
    factory, conversation_ids = database
    model = FakeToolCallingChatModel(selector_items=["第一轮回答"])
    service = ToolAwareChatService(
        settings=_settings(), model=model, session_factory=factory
    )
    user_id = uuid4()

    first, first_events = await _events(
        service, ChatRequest(user_id=user_id, message="第一轮问题")
    )
    conversation_ids.append(first.conversation_id)
    model.selector_items = ("第二轮回答",)
    second, second_events = await _events(
        service,
        ChatRequest(
            user_id=user_id,
            conversation_id=first.conversation_id,
            message="接着问",
        ),
    )

    assert isinstance(first.conversation_id, int)
    assert first_events[0].data == {"conversation_id": first.conversation_id}
    assert second.conversation_id == first.conversation_id
    seen_human = [
        message.content
        for message in model.selector_seen_messages
        if isinstance(message, HumanMessage)
    ]
    assert seen_human == ["第一轮问题", "接着问"]
    assert second_events[-1].name == "done"


async def test_one_tool_call_is_persisted_and_converges(database) -> None:
    factory, conversation_ids = database
    selector = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "query_logistics",
                "args": '{"order_id":"1001"}',
                "id": "call-1",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )
    model = FakeToolCallingChatModel(
        selector_items=[selector], final_items=["包裹正在运输中。"]
    )
    service = ToolAwareChatService(
        settings=_settings(), model=model, session_factory=factory
    )

    turn, events = await _events(
        service, ChatRequest(user_id=uuid4(), message="订单 1001 到哪了")
    )
    conversation_ids.append(turn.conversation_id)

    assert [event.name for event in events] == [
        "conversation",
        "tool_status",
        "tool_status",
        "token",
        "done",
    ]
    async with factory() as session:
        rows = list(
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == turn.conversation_id)
                .order_by(Message.id)
            )
        )
    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant"]


async def test_selector_preamble_with_tool_call_still_executes_and_converges(
    database,
) -> None:
    """Some OpenAI-compatible models emit brief text before a valid tool call."""

    factory, conversation_ids = database
    selector = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "query_logistics",
                "args": '{"order_id":"1001"}',
                "id": "call-mixed",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )
    model = FakeToolCallingChatModel(
        selector_items=["我来查询一下。", selector],
        final_items=["包裹正在运输中。"],
    )
    service = ToolAwareChatService(
        settings=_settings(), model=model, session_factory=factory
    )

    turn, events = await _events(
        service, ChatRequest(user_id=uuid4(), message="订单 1001 的物流到哪了")
    )
    conversation_ids.append(turn.conversation_id)

    assert all(event.data.get("code") != "mixed_tool_response" for event in events)
    assert [event.name for event in events] == [
        "conversation",
        "token",
        "tool_status",
        "tool_status",
        "token",
        "done",
    ]
    assert events[-1].data["content"] == "包裹正在运输中。"


async def test_model_failure_returns_a_safe_error_without_closing_conversation(
    database,
) -> None:
    factory, conversation_ids = database
    model = FakeToolCallingChatModel(selector_items=[RuntimeError("secret")])
    service = ToolAwareChatService(
        settings=_settings(), model=model, session_factory=factory
    )

    turn, events = await _events(
        service, ChatRequest(user_id=uuid4(), message="你好")
    )
    conversation_ids.append(turn.conversation_id)

    assert events[-1].name == "error"
    assert "secret" not in str(events[-1].data)
    async with factory() as session:
        conversation = await session.get(Conversation, turn.conversation_id)
    assert conversation is not None
    assert conversation.status == "进行中"


async def test_faq_answer_returns_validated_citation_sources(database) -> None:
    factory, conversation_ids = database
    model = FakeToolCallingChatModel(
        selector_items=[faq_selector()],
        final_items=["运费请以结算页为准 [1]。"],
    )
    service = ToolAwareChatService(
        settings=_settings(),
        model=model,
        session_factory=factory,
        knowledge_search=FixedKnowledgeSearch(confident=True),
        evidence_checker=FixedEvidenceChecker(),
    )

    turn, events = await _events(
        service, ChatRequest(user_id=uuid4(), message="邮费是多少")
    )
    conversation_ids.append(turn.conversation_id)

    done = events[-1]
    assert done.name == "done"
    assert done.data["content"] == "运费请以结算页为准 [1]。"
    assert done.data["sources"] == [
        {
            "number": 1,
            "chunk_id": 7,
            "category": "配送服务",
            "section_path": "配送服务 / 配送费用",
        }
    ]


async def test_low_confidence_faq_refuses_and_enters_question_pool(database) -> None:
    factory, conversation_ids = database
    model = FakeToolCallingChatModel(selector_items=[faq_selector()])
    service = ToolAwareChatService(
        settings=_settings(),
        model=model,
        session_factory=factory,
        knowledge_search=FixedKnowledgeSearch(confident=False),
        evidence_checker=FixedEvidenceChecker(),
    )

    turn, events = await _events(
        service, ChatRequest(user_id=uuid4(), message="会员积分什么时候到账")
    )
    conversation_ids.append(turn.conversation_id)

    assert events[-1].name == "done"
    assert "不会猜测" in events[-1].data["content"]
    assert model.final_calls == 0
    async with factory() as session:
        row = await session.scalar(
            select(LowConfidenceQuestion).where(
                LowConfidenceQuestion.conversation_id == turn.conversation_id
            )
        )
    assert row is not None
    assert row.source == "retrieval_low_conf"
    assert row.raw_question == "会员积分什么时候到账"
