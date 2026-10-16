"""Behavior contracts for the five model-visible customer-service tools."""

import json
from random import Random
from uuid import uuid4

import pytest
from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.database.models import Ticket
from app.database.models import KnowledgeChunk
from app.knowledge.query_understanding import QueryPlan
from app.knowledge.search import KnowledgeMatch, RetrievalResult, RetrievalStrategy
from app.repositories.conversations import ConversationRepository
from app.tools.business import ToolRuntime, build_business_tools
from app.tools.registry import ToolRegistry


class SemanticPostageSearch:
    async def retrieve(self, query: str, *, limit: int = 10, **kwargs):
        if query != "邮费是多少":
            return RetrievalResult((), QueryPlan(normalized_query=query), RetrievalStrategy.HYBRID_RERANK, False)
        row = KnowledgeChunk(
            id=7,
            questions="配送费用如何计算？",
            answer="配送费用会在结算页展示。",
            category="配送费用",
            section_path="配送服务 / 配送费用",
            content_type="faq",
            vectorize_status="done",
        )
        return RetrievalResult(
            (KnowledgeMatch(row, 0.93),),
            QueryPlan(normalized_query="配送费用如何计算"),
            RetrievalStrategy.HYBRID_RERANK,
            True,
        )


@pytest.fixture
async def tool_runtime(mysql_session) -> ToolRuntime:
    """Supply one trusted, persisted conversation to database-backed tools."""

    session_factory = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    async with session_factory() as session:
        conversation = await ConversationRepository(session).create(str(uuid4()))
        await session.commit()
    return ToolRuntime(
        conversation_id=conversation.id,
        session_factory=session_factory,
        rng=Random(7),
        knowledge_search=SemanticPostageSearch(),
    )


@pytest.fixture
async def registry(tool_runtime: ToolRuntime) -> ToolRegistry:
    return ToolRegistry(build_business_tools(tool_runtime))


def tool_by_name(tools: list[BaseTool], name: str) -> BaseTool:
    return next(tool for tool in tools if tool.name == name)


async def test_business_tools_expose_only_customer_model_arguments(
    tool_runtime: ToolRuntime,
) -> None:
    """A runtime identity field accidentally exposed to the model must fail this test."""

    tools = build_business_tools(tool_runtime)

    assert all(isinstance(tool, BaseTool) for tool in tools)
    assert {tool.name: set(tool.args_schema.model_fields) for tool in tools} == {
        "query_order": {"order_id"},
        "query_product": {"product"},
        "query_logistics": {"order_id", "tracking_no"},
        "query_faq": {"keyword"},
        "create_ticket": {"description", "ticket_type"},
    }


async def test_query_faq_keeps_contract_while_postage_paraphrase_is_recalled(
    tool_runtime: ToolRuntime,
) -> None:
    """Changing lexical FAQ lookup or seed wording must change these observable results."""

    faq_tool = tool_by_name(build_business_tools(tool_runtime), "query_faq")

    postage = json.loads(await faq_tool.ainvoke({"keyword": "邮费是多少"}))

    assert postage == {
        "matches": [
            {
                "citation": 1,
                "chunk_id": 7,
                "question": "配送费用如何计算？",
                "answer": "配送费用会在结算页展示。",
                "category": "配送费用",
                "section_path": "配送服务 / 配送费用",
                "score": 0.93,
            }
        ],
        "normalized_query": "配送费用如何计算",
        "confident": True,
    }


async def test_demo_product_is_deterministic_when_its_rng_is_injected(mysql_session) -> None:
    """Replacing injected randomness with global state must break repeatable demo replies."""

    session_factory = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    first = ToolRuntime(1, session_factory, Random(11))
    second = ToolRuntime(2, session_factory, Random(11))

    first_tools = {tool.name: tool for tool in build_business_tools(first)}
    second_tools = {tool.name: tool for tool in build_business_tools(second)}

    assert await first_tools["query_product"].ainvoke({"product": "保温杯"}) == await second_tools[
        "query_product"
    ].ainvoke({"product": "保温杯"})


async def test_ticket_uses_trusted_conversation_and_stable_retry_identity(
    tool_runtime: ToolRuntime,
) -> None:
    """Using a model argument or unstable ticket key must fail this trusted-context contract."""

    ticket_tool = tool_by_name(build_business_tools(tool_runtime), "create_ticket")

    first = json.loads(
        await ticket_tool.ainvoke({"description": " 包裹破损 ", "ticket_type": "投诉"})
    )
    second = json.loads(
        await ticket_tool.ainvoke({"description": "包裹破损", "ticket_type": "投诉"})
    )

    assert set(ticket_tool.args_schema.model_fields) == {"description", "ticket_type"}
    assert first == second
    assert first["status"] == "待处理"
    assert first["ticket_no"].startswith("TICKET-")
    async with tool_runtime.session_factory() as session:
        ticket = await session.get(Ticket, first["ticket_no"])
    assert ticket is not None
    assert ticket.conversation_id == tool_runtime.conversation_id


def test_registry_contains_exact_business_tools(registry: ToolRegistry) -> None:
    """Registering an unreviewed tool or omitting a required one must fail the allowlist."""

    assert registry.names == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    }
