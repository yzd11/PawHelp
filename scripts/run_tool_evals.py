"""Run the labelled one-tool acceptance set against the configured live service."""

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
import re
from uuid import uuid4

from app.config import Settings
from app.database.models import Conversation, Message, Ticket, Order, Shipment, ShipmentEvent
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.llm import create_chat_model
from app.repositories.messages import MessageRepository
from app.schemas import ChatRequest
from app.services.tool_chat import ToolAwareChatService
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "tool_call_cases.jsonl"
_STATUS_PARAPHRASES = {
    "accepted": ("accepted", "已揽收", "已接单", "已受理"),
    "in_transit": ("in_transit", "运输中", "正在运输", "运输途中"),
    "out_for_delivery": ("out_for_delivery", "派送中", "正在派送"),
    "delivered": ("delivered", "已签收", "已送达"),
}
_LOCATION_PARAPHRASES = {
    "配送中心": ("配送中心",),
    "运输途中": ("运输途中", "还在路上", "在路上"),
    "派送站": ("派送站", "派送站点"),
}
_UNSUPPORTED_LOGISTICS_CLAIMS = ("退款", "退货", "取消订单", "签收", "丢失", "赔偿")
_UNSUPPORTED_POLICY_CLAIMS = (
    "退款",
    "全额",
    "赔偿",
    "邮费",
    "配送费",
    "免费",
    "七天无理由",
    "七天",
    "七日",
    "无理由",
    "换货",
    "不符合条件",
    "无条件",
    "都能",
    "任意商品",
    "包邮",
    "收费",
    "费用",
    "金额",
    "时效",
)
_SHIPPING_FEE_TERMS = ("邮费", "运费", "配送费", "配送费用")
_SHIPPING_BASIS_TERMS = (
    "订单金额",
    "收货地区",
    "地区",
    "活动规则",
    "结算页",
    "提交订单",
    "实际运费",
    "计算",
)
_UNSUPPORTED_SHIPPING_CLAIMS = ("免费", "包邮", "固定收费", "统一收费")
_EXACT_FEE_CLAIM = re.compile(r"\d+(?:\.\d+)?(?:元|块|rmb)", re.IGNORECASE)
_ETA_UNKNOWN = re.compile(
    r"(?:无法|不能|暂无|未能|未提供|不知道|不确定).{0,6}(?:预计|预估|确定|提供)?.{0,6}(?:送达|到达|送到)(?:时间)?"
)
_ETA_INVENTION = re.compile(
    r"(?:(?:预计|预估|大概|约|可能).{0,12}(?:送达|到达|能送到|送到)|"
    r"(?:\d+|一|两|三|四|五|六|七|八|九|十)(?:个)?(?:小时|分钟|天|日).{0,8}(?:送达|到达|能送到|送到)|"
    r"(?:今天|明天|后天).{0,8}(?:送达|到达|能送到|送到))"
)
_STATUS_NEGATION_PREFIX = re.compile(r"(?:尚未|还没|并未|未|没有|没|不在|并非|不是)")


@dataclass(frozen=True)
class ScoreResult:
    """A deterministic outcome for one labelled acceptance observation."""

    passed: bool
    classification: str


@dataclass(frozen=True)
class EvalCase:
    """One approved phrase and its expected bounded workflow outcome."""

    case_id: str
    message: str
    expected_tool: str
    expected_outcome: str


@dataclass(frozen=True)
class EvalObservation:
    """Private execution facts used to score, but never printed verbatim."""

    conversation_id: int
    observed_tool: str | None
    tool_succeeded: bool
    answer: str
    tool_result: str | None
    terminal_event: str


def _normalise(value: str) -> str:
    return "".join(value.casefold().split())


def _contains_any(answer: str, phrases: tuple[str, ...]) -> bool:
    return any(_normalise(phrase) in answer for phrase in phrases)


def _logistics_answer_is_grounded(answer: str, payload: object) -> bool:
    """Validate the fixed logistics facts and known contradictory claim classes."""

    if not isinstance(payload, dict):
        return False
    order_id = payload.get("order_id")
    status = payload.get("status")
    location = payload.get("location")
    if not all(isinstance(item, str) and item for item in (order_id, status, location)):
        return False
    unsupported = tuple(term for term in _UNSUPPORTED_LOGISTICS_CLAIMS if term != "签收" or status != "delivered")
    if _contains_any(answer, unsupported):
        return False
    status_terms = _STATUS_PARAPHRASES.get(status, (status,))
    location_terms = _LOCATION_PARAPHRASES.get(location, (location,))
    if _status_is_negated(answer, status_terms):
        return False
    if _normalise(order_id) not in answer or not _contains_any(answer, location_terms):
        return False
    if not _contains_any(answer, status_terms):
        return False
    eta = payload.get("eta")
    if isinstance(eta, str):
        return _normalise(eta) in answer
    if _ETA_UNKNOWN.search(answer) is not None:
        return True
    return _ETA_INVENTION.search(answer) is None


def _status_is_negated(answer: str, status_terms: tuple[str, ...]) -> bool:
    """Reject a bounded negation immediately before the selected fixed status."""

    return any(
        re.search(_STATUS_NEGATION_PREFIX.pattern + r".{0,3}" + re.escape(term), answer)
        is not None
        for term in status_terms
    )


def _return_policy_answer_is_grounded(answer: str, payload: object) -> bool:
    """Accept faithful return-policy paraphrases while rejecting added benefits."""

    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        return False
    has_return_policy = any(
        isinstance(item, dict)
        and any(
            "退货" in str(item.get(field, ""))
            for field in ("question", "answer", "category")
        )
        for item in payload["matches"]
    )
    required_groups = (("符合条件", "条件"), ("退货", "申请退货"), ("售后页面", "页面说明", "具体规则"))
    return (
        has_return_policy
        and all(_contains_any(answer, group) for group in required_groups)
        and not _contains_any(answer, _UNSUPPORTED_POLICY_CLAIMS)
    )


def _postage_answer_is_grounded(answer: str, payload: object) -> bool:
    """Require a non-empty shipping-fee retrieval and a faithful answer."""

    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        return False
    matches = payload["matches"]
    has_shipping_knowledge = any(
        isinstance(item, dict)
        and any(
            _contains_any(_normalise(str(item.get(field, ""))), _SHIPPING_FEE_TERMS)
            for field in ("question", "answer", "category")
        )
        for item in matches
    )
    return (
        has_shipping_knowledge
        and _contains_any(answer, _SHIPPING_FEE_TERMS)
        and _contains_any(answer, _SHIPPING_BASIS_TERMS)
        and not _contains_any(answer, _UNSUPPORTED_SHIPPING_CLAIMS)
        and _EXACT_FEE_CLAIM.search(answer) is None
    )


def score_case(
    *,
    expected_tool: str,
    expected_outcome: str,
    observed_tool: str | None,
    answer: str,
    tool_result: str | None,
    terminal_event: str,
    tool_succeeded: bool,
) -> ScoreResult:
    """Score a tool trace without network access or model-dependent heuristics."""

    if terminal_event != "done":
        return ScoreResult(False, f"terminal_{terminal_event}")
    if not tool_succeeded:
        return ScoreResult(False, "tool_not_succeeded")
    if observed_tool != expected_tool:
        return ScoreResult(False, "wrong_tool")
    if tool_result is None:
        return ScoreResult(False, "missing_tool_result")
    try:
        payload = json.loads(tool_result)
    except (TypeError, ValueError):
        return ScoreResult(False, "invalid_tool_result")

    if expected_outcome == "postage_answer":
        if _postage_answer_is_grounded(_normalise(answer), payload):
            return ScoreResult(True, "postage_answer")
        return ScoreResult(False, "unexpected_answer")
    if expected_outcome != "answer":
        return ScoreResult(False, "unknown_expected_outcome")
    normalised_answer = _normalise(answer)
    if expected_tool == "query_logistics":
        grounded = _logistics_answer_is_grounded(normalised_answer, payload)
    elif expected_tool == "query_faq":
        grounded = _return_policy_answer_is_grounded(normalised_answer, payload)
    else:
        grounded = False
    if not grounded:
        return ScoreResult(False, "ungrounded_answer")
    return ScoreResult(True, "answer")


def has_usable_upstream(settings: Settings) -> bool:
    """Reject absent and known placeholder credentials before any network call."""

    key = settings.openai_api_key.get_secret_value().strip()
    return bool(key) and not key.casefold().startswith(("replace-with", "placeholder"))


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    """Load only the checked-in, small labelled acceptance dataset."""

    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            EvalCase(
                case_id=str(raw["id"]),
                message=str(raw["message"]),
                expected_tool=str(raw["expected_tool"]),
                expected_outcome=str(raw["expected_outcome"]),
            )
        )
    return cases


async def _observe_case(
    service: ToolAwareChatService,
    session_factory: async_sessionmaker[AsyncSession],
    case: EvalCase,
    request: ChatRequest,
) -> EvalObservation:
    turn = await service.prepare_turn(request)
    observed_tool: str | None = None
    tool_succeeded = False
    answer_parts: list[str] = []
    terminal_event = "incomplete"
    async for event in service.stream_turn(turn):
        if event.name == "tool_status":
            name = event.data.get("name")
            if observed_tool is None and isinstance(name, str):
                observed_tool = name
            tool_succeeded = tool_succeeded or event.data.get("status") == "succeeded"
        elif event.name == "token":
            content = event.data.get("content")
            if isinstance(content, str):
                answer_parts.append(content)
        elif event.name in {"done", "error"}:
            terminal_event = event.name

    tool_result: str | None = None
    async with session_factory() as session:
        rows = await MessageRepository(session).list_ordered(turn.conversation_id)
    for row in reversed(rows):
        if row.role == "tool":
            tool_result = row.content
            break
    return EvalObservation(
        turn.conversation_id,
        observed_tool,
        tool_succeeded,
        "".join(answer_parts),
        tool_result,
        terminal_event,
    )


async def _cleanup_case(
    session_factory: async_sessionmaker[AsyncSession], conversation_id: int
) -> None:
    """Remove only rows correlated to one random evaluation conversation."""

    async with session_factory() as session, session.begin():
        owner = await session.scalar(select(Conversation.user_id).where(Conversation.id == conversation_id))
        if owner:
            # This runner creates a fresh UUID per case. Never clean arbitrary customer data.
            tracking = select(Shipment.tracking_no).where(Shipment.user_id == owner)
            await session.execute(delete(ShipmentEvent).where(ShipmentEvent.tracking_no.in_(tracking)))
            await session.execute(delete(Shipment).where(Shipment.user_id == owner))
            await session.execute(delete(Order).where(Order.user_id == owner, Order.is_demo == 1))
        await session.execute(delete(Message).where(Message.conversation_id == conversation_id))
        await session.execute(delete(Ticket).where(Ticket.conversation_id == conversation_id))
        await session.execute(delete(Conversation).where(Conversation.id == conversation_id))


def _safe_summary(case: EvalCase, observation: EvalObservation, result: ScoreResult) -> str:
    observed = observation.observed_tool or "none"
    return (
        f"{case.case_id}: expected_tool={case.expected_tool} observed_tool={observed} "
        f"outcome={result.classification} terminal={observation.terminal_event} "
        f"answer_chars={len(observation.answer)} pass={result.passed}"
    )


async def run_cases(
    cases: Iterable[EvalCase],
    settings: Settings,
    *,
    service: ToolAwareChatService | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> int:
    """Execute each labelled phrase through the real service and print safe summaries."""

    owned_engine = None
    try:
        if service is None:
            owned_engine = create_async_engine_from_settings(settings)
            session_factory = create_session_factory(owned_engine)
            service = ToolAwareChatService(
                settings=settings,
                model=create_chat_model(settings),
                session_factory=session_factory,
                rng=Random(7),
            )
        if session_factory is None:
            raise ValueError("injected evaluation service requires its session factory")
        results: list[ScoreResult] = []
        for case in cases:
            request = ChatRequest(
                user_id=uuid4(),
                message=case.message,
            )
            observation = EvalObservation(
                0, None, False, "", None, "unavailable"
            )
            result = ScoreResult(False, "unavailable")
            try:
                observation = await _observe_case(service, session_factory, case, request)
                result = score_case(
                    expected_tool=case.expected_tool,
                    expected_outcome=case.expected_outcome,
                    observed_tool=observation.observed_tool,
                    answer=observation.answer,
                    tool_result=observation.tool_result,
                    terminal_event=observation.terminal_event,
                    tool_succeeded=observation.tool_succeeded,
                )
            except Exception:
                pass
            finally:
                if observation.conversation_id:
                    try:
                        await _cleanup_case(session_factory, observation.conversation_id)
                    except Exception:
                        result = ScoreResult(False, "cleanup_failed")
            results.append(result)
            print(_safe_summary(case, observation, result))
    finally:
        if owned_engine is not None:
            await owned_engine.dispose()

    passed = sum(result.passed for result in results)
    print(f"summary: passed={passed}/{len(results)}")
    return 0 if passed == len(results) else 1


def main() -> int:
    """Run the acceptance set, explicitly skipping only missing configuration."""

    try:
        settings = Settings()
    except Exception:
        print("SKIPPED: configure a usable OpenAI-compatible upstream in .env")
        return 2
    if not has_usable_upstream(settings):
        print("SKIPPED: configure a usable OpenAI-compatible upstream in .env")
        return 2
    try:
        return asyncio.run(run_cases(load_cases(), settings))
    except Exception:
        print("UNAVAILABLE: evaluation runner could not start")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
