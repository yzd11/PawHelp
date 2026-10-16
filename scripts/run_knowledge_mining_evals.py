"""Evaluate the conversation-mining prompt with one labelled model call."""

import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.knowledge.mining import ConversationKnowledgeExtractor
from app.llm import create_chat_model
from app.repositories.knowledge_mining import Transcript


CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "conversation_knowledge_cases.jsonl"


async def run() -> bool:
    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    transcripts = [
        Transcript(
            conversation_id=case["conversation_id"],
            messages=tuple(case["messages"]),
        )
        for case in cases
    ]
    settings = get_settings()
    candidates = await ConversationKnowledgeExtractor(
        create_chat_model(settings),
        method=settings.structured_output_method,
    ).extract(transcripts)
    all_passed = True
    for case in cases:
        actual = [
            candidate
            for candidate in candidates
            if candidate.source_conversation_id == case["conversation_id"]
        ]
        minimum_ok = len(actual) >= case.get("expected_min", 0)
        maximum_ok = len(actual) <= case.get("expected_max", len(actual))
        combined_questions = "\n".join(item.question for item in actual)
        combined_answers = "\n".join(item.answer for item in actual)
        question_terms = case.get("required_any", [])
        answer_terms = case.get("answer_required_any", [])
        question_ok = not question_terms or any(
            term in combined_questions for term in question_terms
        )
        answer_ok = not answer_terms or any(term in combined_answers for term in answer_terms)
        passed = minimum_ok and maximum_ok and question_ok and answer_ok
        all_passed = all_passed and passed
        print(
            json.dumps(
                {
                    "id": case["id"],
                    "passed": passed,
                    "extracted": len(actual),
                },
                ensure_ascii=False,
            )
        )
    print(json.dumps({"all_passed": all_passed}, ensure_ascii=False))
    return all_passed


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(run()) else 1)
