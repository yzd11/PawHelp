"""Write-time check of a new knowledge entry against the existing knowledge."""

import asyncio
from dataclasses import dataclass
import json
import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.knowledge.search import RetrievalStrategy
from app.prompts import render_knowledge_consistency_messages

logger = logging.getLogger(__name__)


class ConsistencyDecision(BaseModel):
    verdict: Literal["duplicate", "conflict", "new"]
    existing_id: int | None = None
    reason: str = Field(min_length=1, max_length=500)


@dataclass(frozen=True)
class ConsistencyResult:
    verdict: Literal["duplicate", "conflict", "new", "unavailable"]
    reason: str
    chunk_id: int | None = None
    label: str | None = None

    @property
    def message(self) -> str:
        target = f"已有知识 #{self.chunk_id}（{self.label}）" if self.chunk_id else "已有知识"
        label = {"duplicate": "与" + target + "重复", "conflict": "与" + target + "冲突",
                 "unavailable": "冲突检查不可用，未写入知识库", "new": "可以写入"}[self.verdict]
        return label if self.verdict == "unavailable" else f"{label}：{self.reason}"


class KnowledgeConsistencyChecker:
    """Retrieve the closest existing entries locally, then ask the model once."""

    def __init__(self, search, model, *, method: str, limit: int = 5) -> None:
        self._search, self._model, self._method, self._limit = search, model, method, limit

    async def check(self, question: str, answer: str, *, exclude_id: int | None = None) -> ConsistencyResult:
        try:
            results = await asyncio.gather(*(
                self._search.retrieve(query, strategy=RetrievalStrategy.HYBRID_RERANK, limit=self._limit + 1)
                for query in (question, answer)))
            by_id = {}
            for result in results:
                for match in result.matches:
                    if match.chunk.id != exclude_id and match.score > by_id.get(match.chunk.id, (None, -1.0))[1]:
                        by_id[match.chunk.id] = (match.chunk, match.score)
            closest = [chunk for chunk, _ in sorted(by_id.values(), key=lambda item: item[1], reverse=True)][:self._limit]
            if not closest:
                return ConsistencyResult("new", "知识库中没有相近知识")
            existing = [{"id": chunk.id, "source": "document" if chunk.source_kind == "markdown" else "added",
                         "section_path": chunk.section_path, "question": chunk.questions, "answer": chunk.answer}
                        for chunk in closest]
            payload = json.dumps({"candidate": {"question": question, "answer": answer}, "existing": existing},
                                 ensure_ascii=False, separators=(",", ":"))
            runnable = self._model.with_structured_output(ConsistencyDecision, method=self._method)
            decision = ConsistencyDecision.model_validate(
                await runnable.ainvoke(render_knowledge_consistency_messages(payload)))
        except Exception:
            logger.exception("knowledge consistency check failed")
            return ConsistencyResult("unavailable", "冲突检查不可用")
        if decision.verdict == "new":
            return ConsistencyResult("new", decision.reason)
        cited = next((chunk for chunk in closest if chunk.id == decision.existing_id), None)
        return ConsistencyResult(decision.verdict, decision.reason,
                                 cited.id if cited else None, (cited.section_path or cited.questions) if cited else None)
