"""Structured generation-time evidence sufficiency check."""

import json
from collections.abc import Sequence
from typing import Literal, Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.prompts import render_evidence_sufficiency_messages


class EvidenceDecision(BaseModel):
    useful: bool
    reason: str = Field(min_length=1, max_length=500)
    check_failed: bool = False


class JudgedDecision(EvidenceDecision):
    """A decision plus who made it; kept out of the LLM's output schema."""

    judge: Literal["llm", "jev"] = "llm"
    jev: dict | None = None


def evidence_payload(question: str, evidence: Sequence[dict[str, object]]) -> str:
    """The exact payload every sufficiency judge reads."""
    return json.dumps(
        {"question": question, "evidence": list(evidence)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


class StructuredRunnable(Protocol):
    async def ainvoke(self, messages: Sequence[BaseMessage]) -> object: ...


class StructuredOutputModel(Protocol):
    def with_structured_output(
        self, schema: type[EvidenceDecision], *, method: str
    ) -> StructuredRunnable: ...


class EvidenceSufficiencyChecker:
    """Ask the model once whether retrieved evidence can safely answer."""

    def __init__(self, model: StructuredOutputModel, *, method: str) -> None:
        self._model = model
        self._method = method

    async def check(
        self, question: str, evidence: Sequence[dict[str, object]]
    ) -> EvidenceDecision:
        payload = evidence_payload(question, evidence)
        try:
            runnable = self._model.with_structured_output(
                EvidenceDecision, method=self._method
            )
            result = await runnable.ainvoke(
                render_evidence_sufficiency_messages(payload)
            )
            return EvidenceDecision.model_validate(result)
        except Exception:
            return EvidenceDecision(
                useful=False,
                reason="证据充分性检查不可用，按安全策略拒答",
                check_failed=True,
            )
