"""Run the labelled intent set through the live classifier and deterministic router.

Each case checks the structured decision (intent, task_kind, requested_lookups)
and where the customer ends up (route, whether an order card is shown). Every
case runs several times because a single pass cannot show instability.

    uv run python -m scripts.run_intent_evals [--runs 3]
"""

import argparse
import asyncio
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm import create_chat_model  # noqa: E402
from app.workflow.understanding import IntentDecision, ResolvedQuestion, RoutePlan, classify_intent, plan_route  # noqa: E402

CASES_PATH = ROOT / "evals" / "intent_cases.jsonl"


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def mismatches(case: dict[str, Any], decision: IntentDecision, plan: RoutePlan) -> list[str]:
    """Labelled fields the decision or route got wrong; empty means the case passed."""
    wrong = []
    if decision.intent not in case["intents"]:
        wrong.append(f"intent={decision.intent}")
    if case["task_kind"] is not None and decision.task_kind != case["task_kind"]:
        wrong.append(f"task_kind={decision.task_kind}")
    if decision.requested_lookups != case["requested_lookups"]:
        wrong.append(f"requested_lookups={decision.requested_lookups}")
    if plan.route != case["route"]:
        wrong.append(f"route={plan.route}")
    if (plan.decision == "select_order") != case["select_order"]:
        wrong.append(f"select_order={plan.decision == 'select_order'}")
    return wrong


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3, help="classifier calls per case")
    args = parser.parse_args()
    try:
        settings = get_settings()
    except Exception:
        print("SKIPPED: configure a usable upstream in .env")
        return 2
    model = create_chat_model(settings)
    method = settings.structured_output_method
    cases = load_cases()
    gate = asyncio.Semaphore(4)

    async def run(case: dict[str, Any]) -> list[str]:
        async with gate:
            try:
                decision = await classify_intent(model, method, case["question"], [])
            except Exception as error:  # one bad call is a failed run, not a crashed report
                return [f"error={type(error).__name__}"]
        return mismatches(case, decision, plan_route(decision, ResolvedQuestion(question=case["question"])))

    results = await asyncio.gather(*(run(case) for case in cases for _ in range(args.runs)))
    print(f"model={settings.openai_model} method={method} runs={args.runs}")
    passed, total = Counter(), Counter()
    for index, case in enumerate(cases):
        runs = results[index * args.runs:(index + 1) * args.runs]
        passed[case["group"]] += sum(not wrong for wrong in runs)
        total[case["group"]] += len(runs)
        marks = "".join("✓" if not wrong else "✗" for wrong in runs)
        problems = sorted({item for wrong in runs for item in wrong})
        print(f"{marks} {case['id']}: {case['question']}" + (f"  -> {', '.join(problems)}" if problems else ""))
    for group in total:
        print(f"{group}: {passed[group]}/{total[group]}")
    all_passed = sum(passed.values()) == sum(total.values())
    print(f"total: {sum(passed.values())}/{sum(total.values())}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
