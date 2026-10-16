"""The five bounded business tools available to the model."""

from dataclasses import dataclass
from datetime import timezone
import hashlib
import json
from random import Random

from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.knowledge.search import KnowledgeSearch
from app.database.models import Conversation
from app.repositories.orders import OrderNotFound
from app.repositories.tickets import TicketRepository
from app.services.orders import OrderData
from app.tools.schemas import FAQArgs, LogisticsArgs, OrderArgs, ProductArgs, TicketArgs


@dataclass(frozen=True)
class ToolRuntime:
    """Trusted context captured outside model-controlled tool arguments."""

    conversation_id: int
    session_factory: async_sessionmaker[AsyncSession]
    rng: Random
    knowledge_search: KnowledgeSearch | None = None
    user_id: str | None = None
    order_data: OrderData | None = None
    ticket_key: str | None = None


def _encode(payload: dict[str, object]) -> str:
    """Return compact UTF-8-safe JSON suitable for a LangChain ToolMessage."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _normalized(value: str) -> str:
    """Collapse incidental whitespace before deriving an idempotency identity."""

    return " ".join(value.split()).casefold()


def _ticket_no(conversation_id: int, description: str, ticket_type: str) -> str:
    """Derive a stable, opaque primary key for one normalized escalation."""

    identity = "\x1f".join(
        (_normalized(str(conversation_id)), _normalized(description), _normalized(ticket_type))
    )
    return f"TICKET-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def _head_tail_evidence(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Place rank 1/2 at the context edges, then alternate inward."""

    left = rows[::2]
    right = list(reversed(rows[1::2]))
    return [*left, *right]


def build_business_tools(runtime: ToolRuntime) -> list[BaseTool]:
    """Build the exact tool set, closing over trusted runtime-only dependencies."""

    orders = runtime.order_data or OrderData(runtime.session_factory)

    async def trusted_user_id() -> str:
        if runtime.user_id:
            return runtime.user_id
        async with runtime.session_factory() as session:
            conversation = await session.get(Conversation, runtime.conversation_id)
            if conversation is None:
                raise OrderNotFound("没有找到您的这笔订单")
            return conversation.user_id

    @tool(args_schema=OrderArgs)
    async def query_order(order_id: str) -> str:
        """查询当前用户的订单、商品、金额、签收时间和运单号。查物流时先取 tracking_no，再传给 query_logistics。无运单表示尚未发货。"""

        try:
            return _encode(await orders.get_order(await trusted_user_id(), order_id))
        except OrderNotFound:
            return _encode({"code": "order_not_found", "error": "没有找到您的这笔订单"})

    @tool(args_schema=ProductArgs)
    async def query_product(product: str) -> str:
        """Look up demonstration availability using a product name."""

        return _encode(
            {
                "product": product,
                "availability": runtime.rng.choice(("in_stock", "limited", "preorder")),
                "stock": runtime.rng.randint(1, 99),
            }
        )

    @tool(args_schema=LogisticsArgs)
    async def query_logistics(tracking_no: str | None = None, order_id: str | None = None) -> str:
        """用 query_order 返回的 tracking_no 查询当前用户的物流状态及轨迹。兼容旧 order_id 调用。无物流时不会生成运单。"""

        try:
            return _encode(await orders.get_logistics(
                await trusted_user_id(), order_id=order_id, tracking_no=tracking_no,
            ))
        except OrderNotFound:
            return _encode({"code": "order_not_found", "error": "没有找到您的这笔订单"})

    @tool(args_schema=FAQArgs)
    async def query_faq(keyword: str) -> str:
        """Search customer-service knowledge by the meaning of the question."""

        if runtime.knowledge_search is None:
            raise RuntimeError("knowledge search is not configured")
        result = await runtime.knowledge_search.retrieve(keyword, limit=10)
        ranked = [
            {
                "citation": rank,
                "chunk_id": match.chunk.id,
                "question": match.chunk.questions,
                "answer": match.chunk.answer,
                "category": match.chunk.category,
                "section_path": match.chunk.section_path,
                "score": match.score,
            }
            for rank, match in enumerate(result.matches, start=1)
        ]
        return _encode(
            {
                "matches": _head_tail_evidence(ranked),
                "normalized_query": result.query_plan.normalized_query,
                "confident": result.confident,
            }
        )

    @tool(args_schema=TicketArgs)
    async def create_ticket(description: str, ticket_type: str) -> str:
        """Create or retrieve a human-support ticket for the trusted conversation."""

        normalized_description = description.strip()
        ticket_no = (_ticket_no(runtime.conversation_id, runtime.ticket_key, "proposal") if runtime.ticket_key
                     else _ticket_no(runtime.conversation_id, description, ticket_type))
        async with runtime.session_factory() as session:
            async with session.begin():
                ticket, _ = await TicketRepository(session).create_or_get(
                    ticket_no=ticket_no,
                    conversation_id=runtime.conversation_id,
                    description=normalized_description,
                    ticket_type=ticket_type,
                    status="待处理",
                )
        # tickets.created_at is a UTC DATETIME; make its timezone explicit at the API boundary.
        created_at_utc = ticket.created_at.replace(tzinfo=timezone.utc)
        return _encode({
            "ticket_no": ticket.ticket_no,
            "status": ticket.status,
            "created_at": created_at_utc.isoformat().replace("+00:00", "Z"),
        })

    return [query_order, query_product, query_logistics, query_faq, create_ticket]
