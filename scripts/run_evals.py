"""Run the labelled evaluation set through the production model paths."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

from langchain_core.messages import AIMessage, HumanMessage


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.chat import stream_chat_events  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.context import prepare_context  # noqa: E402
from app.extraction import AfterSalesExtractor  # noqa: E402
from app.llm import create_chat_model  # noqa: E402


EVALS_DIR = ROOT / "evals"
PUBLIC_EXTRACTION_FIELDS = ("order_id", "request_type", "expected_solution")


class ChatScore(NamedTuple):
    """Observable outcomes from scoring one completed chat response."""

    passed: bool
    required_any_matched: bool
    forbidden_matched: bool


def score_chat_response(
    response: str,
    case: dict[str, Any],
    *,
    upstream_error: bool = False,
) -> ChatScore:
    """Score a chat response without invoking or depending on a model."""

    normalized_response = response.casefold()
    required_any = [str(term).casefold() for term in case["required_any"]]
    forbidden = [str(term).casefold() for term in case["forbidden"]]
    required_matched = not required_any or any(
        term in normalized_response for term in required_any
    )
    forbidden_matched = any(term in normalized_response for term in forbidden)
    return ChatScore(
        passed=required_matched and not forbidden_matched and not upstream_error,
        required_any_matched=required_matched,
        forbidden_matched=forbidden_matched,
    )


def load_cases(filename: str) -> list[dict[str, Any]]:
    """Load one checked-in JSONL evaluation file."""

    path = EVALS_DIR / filename
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def emit(payload: dict[str, object]) -> None:
    """Write machine-readable results without emitting runtime configuration."""

    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


async def evaluate_extractions(extractor: AfterSalesExtractor) -> tuple[int, int]:
    """Score each structured extraction against all public fields."""

    passed = 0
    total = 0
    for case in load_cases("after_sales_cases.jsonl"):
        total += 1
        try:
            actual = (await extractor.extract(str(case["description"]))).model_dump(mode="json")
            expected = case["expected"]
            correct_fields = sum(
                actual[field] == expected[field] for field in PUBLIC_EXTRACTION_FIELDS
            )
            case_passed = correct_fields == len(PUBLIC_EXTRACTION_FIELDS)
            emit(
                {
                    "kind": "after_sales",
                    "id": case["id"],
                    "passed": case_passed,
                    "correct_fields": correct_fields,
                    "total_fields": len(PUBLIC_EXTRACTION_FIELDS),
                    "field_score": f"{correct_fields}/{len(PUBLIC_EXTRACTION_FIELDS)}",
                }
            )
        except Exception:
            correct_fields = 0
            case_passed = False
            emit(
                {
                    "kind": "after_sales",
                    "id": case["id"],
                    "passed": False,
                    "correct_fields": correct_fields,
                    "total_fields": len(PUBLIC_EXTRACTION_FIELDS),
                    "field_score": f"{correct_fields}/{len(PUBLIC_EXTRACTION_FIELDS)}",
                    "error": "evaluation_error",
                }
            )
        passed += case_passed
    return passed, total


async def evaluate_chats(model: object, settings: Settings) -> tuple[int, int]:
    """Score token-only chat output against required and forbidden terms."""

    passed = 0
    total = 0
    for case in load_cases("chat_cases.jsonl"):
        total += 1
        try:
            history = [
                HumanMessage(content=message["content"])
                if message["role"] == "user"
                else AIMessage(content=message["content"])
                for message in case["messages"]
            ]
            prepared = prepare_context(
                history,
                shop_name=settings.shop_name,
                token_budget=settings.context_token_budget,
            )
            response_parts: list[str] = []
            upstream_error = False
            async for event, data in stream_chat_events(model, prepared):
                if event == "token" and isinstance(data.get("content"), str):
                    response_parts.append(data["content"])
                elif event == "error":
                    upstream_error = True

            score = score_chat_response(
                "".join(response_parts), case, upstream_error=upstream_error
            )
            case_passed = score.passed
            emit(
                {
                    "kind": "chat",
                    "id": case["id"],
                    "passed": case_passed,
                    "required_any_matched": score.required_any_matched,
                    "forbidden_matched": score.forbidden_matched,
                }
            )
        except Exception:
            case_passed = False
            emit(
                {
                    "kind": "chat",
                    "id": case["id"],
                    "passed": False,
                    "error": "evaluation_error",
                }
            )
        passed += case_passed
    return passed, total


async def main() -> int:
    """Run all labelled cases and return the process status code."""

    try:
        settings = get_settings()
    except Exception:
        print("SKIPPED: configure a usable upstream in .env")
        return 2

    model = create_chat_model(settings)
    extractor = AfterSalesExtractor(model, method=settings.structured_output_method)
    extraction_passed, extraction_total = await evaluate_extractions(extractor)
    chat_passed, chat_total = await evaluate_chats(model, settings)
    passed = extraction_passed + chat_passed
    total = extraction_total + chat_total
    all_passed = passed == total
    emit({"kind": "summary", "passed": passed, "total": total, "all_passed": all_passed})
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
