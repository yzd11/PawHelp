"""One-shot query normalization and retrieval-only synonym expansion."""

from collections.abc import Sequence
from typing import Annotated, Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, StringConstraints, field_validator

from app.prompts import render_query_understanding_messages


NormalizedText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QueryPlan(BaseModel):
    """A bounded standalone retrieval plan produced from one raw question."""

    normalized_query: NormalizedText
    synonyms: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("synonyms")
    @classmethod
    def normalize_list(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            normalized = " ".join(value.split())
            if normalized and normalized not in result:
                result.append(normalized)
        return result


class StructuredRunnable(Protocol):
    async def ainvoke(self, messages: Sequence[BaseMessage]) -> object: ...


class StructuredOutputModel(Protocol):
    def with_structured_output(
        self, schema: type[QueryPlan], *, method: str
    ) -> StructuredRunnable: ...


class QueryUnderstandingService:
    """Normalize a query once, falling back to the raw question on provider failure."""

    def __init__(self, model: StructuredOutputModel, *, method: str) -> None:
        self._model = model
        self._method = method

    async def understand(self, query: str) -> QueryPlan:
        try:
            runnable = self._model.with_structured_output(QueryPlan, method=self._method)
            result = await runnable.ainvoke(render_query_understanding_messages(query))
            return QueryPlan.model_validate(result)
        except Exception:
            return QueryPlan(normalized_query=query)


class IdentityQueryUnderstanding:
    """Network-free fallback used by scripts and focused tests."""

    async def understand(self, query: str) -> QueryPlan:
        return QueryPlan(normalized_query=query)
