"""One-call structured extraction from a bounded conversation batch."""

import json
from collections.abc import Sequence
from typing import Annotated, Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, StringConstraints

from app.prompts import render_conversation_knowledge_messages
from app.repositories.knowledge_mining import KnowledgeCandidate, Transcript


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ExtractedQA(BaseModel):
    source_conversation_id: int = Field(gt=0)
    question: NonEmptyText
    answer: NonEmptyText


class ExtractedQABatch(BaseModel):
    items: list[ExtractedQA] = Field(default_factory=list)


class StructuredRunnable(Protocol):
    async def ainvoke(self, messages: Sequence[BaseMessage]) -> object: ...


class StructuredOutputModel(Protocol):
    def with_structured_output(
        self, schema: type[ExtractedQABatch], *, method: str
    ) -> StructuredRunnable: ...


class ConversationKnowledgeExtractor:
    """Extract only source-bound QAs from one supplied conversation batch."""

    def __init__(self, model: StructuredOutputModel, *, method: str) -> None:
        self._structured_model = model.with_structured_output(
            ExtractedQABatch,
            method=method,
        )

    async def extract(
        self, transcripts: Sequence[Transcript]
    ) -> list[KnowledgeCandidate]:
        allowed_ids = {item.conversation_id for item in transcripts}
        payload = [
            {
                "conversation_id": item.conversation_id,
                "messages": list(item.messages),
            }
            for item in transcripts
        ]
        result = await self._structured_model.ainvoke(
            render_conversation_knowledge_messages(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
        )
        batch = ExtractedQABatch.model_validate(result)
        return [
            KnowledgeCandidate(item.source_conversation_id, item.question, item.answer)
            for item in batch.items
            if item.source_conversation_id in allowed_ids
        ]
