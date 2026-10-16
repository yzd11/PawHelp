"""Integrity checks for the hand-labelled real-model evaluation data."""

import json
from pathlib import Path

from app.schemas import AfterSalesExtraction, ExtractionRequest


ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = ROOT / "evals"
EXTRACTION_CASES = EVALS_DIR / "after_sales_cases.jsonl"
CHAT_CASES = EVALS_DIR / "chat_cases.jsonl"
EXPECTED_EXTRACTION_FIELDS = {"order_id", "request_type", "expected_solution"}
REQUIRED_EXTRACTION_IDS = {
    "complete_refund",
    "missing_order_id",
    "exchange_not_return",
    "logistics_complaint",
    "injection_in_complaint",
}
REQUIRED_CHAT_IDS = {
    "ask_for_missing_order_id",
    "does_not_claim_completed_action",
    "resists_prompt_injection",
    "replies_in_chinese",
}


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a labelled JSONL fixture without silently ignoring blank rows."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_extraction_cases_are_valid_and_cover_required_scenarios() -> None:
    """Catch missing or malformed labels before a live model run is trusted."""

    cases = load_jsonl(EXTRACTION_CASES)
    ids = [case["id"] for case in cases]

    assert len(ids) == len(set(ids))
    assert REQUIRED_EXTRACTION_IDS <= set(ids)
    for case in cases:
        assert set(case) == {"id", "description", "expected"}
        ExtractionRequest.model_validate({"description": case["description"]})
        expected = case["expected"]
        assert isinstance(expected, dict)
        assert set(expected) == EXPECTED_EXTRACTION_FIELDS
        AfterSalesExtraction.model_validate(expected)


def test_chat_cases_are_valid_and_cover_required_scenarios() -> None:
    """Catch non-runnable chat labels or assertions that cannot evaluate behavior."""

    cases = load_jsonl(CHAT_CASES)
    ids = [case["id"] for case in cases]

    assert len(ids) == len(set(ids))
    assert REQUIRED_CHAT_IDS <= set(ids)
    for case in cases:
        assert set(case) == {"id", "messages", "required_any", "forbidden"}
        messages = case["messages"]
        assert isinstance(messages, list) and messages
        for index, message in enumerate(messages):
            assert isinstance(message, dict)
            assert set(message) == {"role", "content"}
            assert message["role"] == ("user" if index % 2 == 0 else "assistant")
            assert isinstance(message["content"], str) and message["content"].strip()
        assert messages[-1]["role"] == "user"
        required_any = case["required_any"]
        forbidden = case["forbidden"]
        assert isinstance(required_any, list)
        assert isinstance(forbidden, list)
        assert required_any or forbidden


def test_intent_cases_are_valid_and_their_route_labels_agree_with_the_router() -> None:
    """A label the router cannot reach would make a live run fail for the wrong reason."""

    from typing import get_args

    from app.workflow.understanding import IntentDecision, ResolvedQuestion, plan_route
    from scripts.run_intent_evals import load_cases, mismatches

    cases = load_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    assert {case["group"] for case in cases} == {"fresh_lookup", "fresh_lookup_negative", "task_kind"}
    all_kinds = get_args(IntentDecision.model_fields["task_kind"].annotation)
    for case in cases:
        assert set(case) == {"id", "group", "question", "intents", "task_kind", "requested_lookups", "route", "select_order"}
        # An unlabelled task_kind means the route must hold whichever kind the model picks.
        for intent in case["intents"]:
            for kind in [case["task_kind"]] if case["task_kind"] else all_kinds:
                decision = IntentDecision(intent=intent, confidence=0.95, task_kind=kind,
                                          requested_lookups=case["requested_lookups"])
                plan = plan_route(decision, ResolvedQuestion(question=case["question"]))
                assert mismatches(case, decision, plan) == [], (case["id"], intent, kind)
