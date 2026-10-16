"""Four-strategy retrieval and grounded-answer evaluation."""

from __future__ import annotations

import asyncio
import csv
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Protocol, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chat import text_from_chunk
from app.database.models import KnowledgeChunk
from app.knowledge.query_understanding import QueryPlan
from app.knowledge.evidence import EvidenceSufficiencyChecker
from app.knowledge.pipeline import KnowledgeEvidencePipeline
from app.knowledge.search import HybridKnowledgeSearch, RetrievalResult, RetrievalStrategy
from app.repositories.quality import QualityRepository


REFUSAL = "抱歉，当前知识库没有足够证据回答这个问题，我不会猜测。"
_CallResult = TypeVar("_CallResult")


class EvalCase(BaseModel):
    case_id: str
    query: str
    query_type: str
    difficulty: str
    relevant_chunk_ids: list[int] = Field(default_factory=list)
    relevant_questions: list[str] = Field(default_factory=list)
    relevant_sections: list[str] = Field(default_factory=list)
    relevant_section_groups: list[list[str]] = Field(default_factory=list)
    expected_points: list[str] = Field(default_factory=list)
    reference_answer: str
    expected_answerable: bool
    categories: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)


class FaithfulnessDecision(BaseModel):
    faithful: bool
    score: float = Field(ge=0, le=1)
    reason: str


class CoverageDecision(BaseModel):
    covered_count: int = Field(ge=0)


class StreamingModel(Protocol):
    def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[object]: ...

    def with_structured_output(self, schema: type[BaseModel], **kwargs: object): ...


class LabeledKnowledge(Protocol):
    questions: str
    answer: str


_EvidenceRow = TypeVar("_EvidenceRow", bound=LabeledKnowledge)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    query: str
    query_type: str
    difficulty: str
    strategy: str
    relevant_chunk_ids: list[int]
    retrieved_chunk_ids: list[int]
    recall_at_k: float
    reciprocal_rank: float
    answer: str
    refused: bool | None
    faithfulness: float | None
    faithful: bool | None
    faithfulness_reason: str
    citations: list[dict[str, object]]
    expected_answerable: bool = True
    expected_points: tuple[str, ...] = ()
    evidence_coverage: float = 0.0
    answer_completeness: float | None = None
    relevant_chunk_groups: tuple[tuple[int, ...], ...] = ()
    answer_coverage: float | None = None
    generation_status: str = "completed"
    generation_error: str = ""
    coverage_status: str = "not_run"
    faithfulness_status: str = "not_run"
    ledger_error: str = ""
    evidence_confidence: dict | None = None


@dataclass(frozen=True)
class ResolvedGroundTruth:
    chunk_ids: list[int]
    groups: tuple[tuple[int, ...], ...] = ()


def recall_at_k(
    retrieved_ids: Sequence[int], relevant_ids: Sequence[int], *, k: int
) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0 if not retrieved_ids else 0.0
    return len(relevant.intersection(retrieved_ids[:k])) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[int], relevant_ids: Sequence[int]
) -> float:
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def group_retrieval_scores(
    retrieved_ids: Sequence[int],
    groups: Sequence[Sequence[int]],
    *,
    k: int,
) -> tuple[float, float]:
    """Score each required evidence group once; IDs within a group are alternatives."""

    if not groups:
        return 0.0, 0.0
    ranks = [
        next(
            (rank for rank, chunk_id in enumerate(retrieved_ids, 1) if chunk_id in group),
            0,
        )
        for group in groups
    ]
    return (
        sum(0 < rank <= k for rank in ranks) / len(ranks),
        sum(1.0 / rank for rank in ranks if rank) / len(ranks),
    )


_CSV_FIELDS = {
    "id": "id",
    "bucket": "桶(bucket)",
    "query": "问题(query)",
    "section": "期望章节(expect_section)",
    "points": "标准要点(expect_points)",
    "refuse": "应拒答(should_refuse)",
}


def _split_labels(value: str | None) -> list[str]:
    return [part.strip() for part in re.split(r"\s*[|+]\s*", value or "") if part.strip()]


def _split_section_groups(value: str | None) -> list[list[str]]:
    """In the CSV, '+' joins required sections; '|' and '/' name alternatives."""

    groups: list[list[str]] = []
    for required in re.split(r"\s*\+\s*", value or ""):
        aliases = [
            part.strip()
            for part in re.split(r"\s*[|/]\s*", required)
            if part.strip()
        ]
        if aliases:
            groups.append(aliases)
    return groups


def _csv_difficulty(bucket: str) -> str:
    return {
        "A_policy": "easy",
        "B_model": "medium",
        "C_colloquial": "medium",
        "D_absent": "hard",
        "E_multi": "hard",
        "F_identifier": "medium",
    }.get(bucket, "medium")


def _load_csv_cases(path: Path) -> list[EvalCase]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(_CSV_FIELDS.values()).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"evaluation CSV is missing columns: {sorted(missing)}")
        cases: list[EvalCase] = []
        for line_number, row in enumerate(reader, 2):
            try:
                bucket = (row[_CSV_FIELDS["bucket"]] or "").strip()
                should_refuse = (row[_CSV_FIELDS["refuse"]] or "").strip() == "是"
                points = _split_labels(row[_CSV_FIELDS["points"]])
                section_groups = _split_section_groups(row[_CSV_FIELDS["section"]])
                cases.append(
                    EvalCase(
                        case_id=(row[_CSV_FIELDS["id"]] or "").strip(),
                        query=(row[_CSV_FIELDS["query"]] or "").strip(),
                        query_type=bucket,
                        difficulty=_csv_difficulty(bucket),
                        relevant_sections=[
                            section for group in section_groups for section in group
                        ],
                        relevant_section_groups=section_groups,
                        expected_points=points,
                        reference_answer=(
                            REFUSAL if should_refuse else "；".join(points)
                        ),
                        expected_answerable=not should_refuse,
                    )
                )
            except Exception as error:
                raise ValueError(
                    f"invalid evaluation CSV row at line {line_number}"
                ) from error
    return cases


def load_cases(path: Path) -> list[EvalCase]:
    if path.suffix.lower() == ".csv":
        return _load_csv_cases(path)
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except Exception as error:
            raise ValueError(f"invalid evaluation case at line {line_number}") from error
    return cases


class CachedQueryPlanner:
    """Reuse one query-understanding call across the four retrieval strategies."""

    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self._cache: dict[str, QueryPlan] = {}

    async def understand(self, query: str) -> QueryPlan:
        if query not in self._cache:
            self._cache[query] = await self._delegate.understand(query)
        return self._cache[query]


class RetrievalQualityEvaluator:
    def __init__(
        self,
        *,
        search: HybridKnowledgeSearch,
        model: StreamingModel,
        session_factory: async_sessionmaker[AsyncSession],
        structured_output_method: str,
        judge_model: str,
        top_k: int = 10,
        call_timeout_seconds: float = 45.0,
        retry_delay_seconds: float = 2.0,
        evidence_checker: EvidenceSufficiencyChecker | None = None,
        evaluation_source_paths: Sequence[str] | None = None,
        evidence_pipeline: KnowledgeEvidencePipeline | None = None,
    ) -> None:
        self._search = search
        self._model = model
        self._session_factory = session_factory
        self._structured_output_method = structured_output_method
        self._judge_model = judge_model
        self._evidence_pipeline = evidence_pipeline
        self._top_k = top_k
        self._call_timeout_seconds = call_timeout_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._evaluation_source_paths = tuple(evaluation_source_paths) if evaluation_source_paths is not None else None
        self._evidence_checker = evidence_checker or EvidenceSufficiencyChecker(
            model, method=structured_output_method  # type: ignore[arg-type]
        )

    async def run(
        self,
        cases: Sequence[EvalCase],
        *,
        strategies: Sequence[RetrievalStrategy] = tuple(RetrievalStrategy),
        generate_answers: bool = True,
        generation_strategies: Sequence[RetrievalStrategy] | None = None,
    ) -> dict[str, object]:
        resolved, corpus_chunk_ids = await self._resolve_ground_truth(cases)
        jobs = []
        for case in cases:
            ground_truth = resolved[case.case_id]
            for strategy in strategies:
                search_options = {"allowed_chunk_ids": corpus_chunk_ids} if corpus_chunk_ids is not None else {}
                should_generate = generate_answers and (
                    case.expected_answerable or strategy is RetrievalStrategy.HYBRID_RERANK
                ) and (generation_strategies is None or strategy in generation_strategies)
                confidence = None
                if self._evidence_pipeline is not None and strategy is RetrievalStrategy.HYBRID_RERANK and should_generate:
                    retrieval = await self._evidence_pipeline.retrieve(
                        case.query, categories=case.categories, content_types=case.content_types, **search_options
                    )
                    result, confidence = retrieval.result, retrieval.confidence
                else:
                    result = await self._search.retrieve(
                        case.query, strategy=strategy, limit=self._top_k,
                        categories=case.categories, content_types=case.content_types, **search_options,
                    )
                jobs.append(
                    (
                        case,
                        ground_truth,
                        result,
                        should_generate,
                        confidence,
                    )
                )
        semaphore = asyncio.Semaphore(5)

        async def score_job(
            job
        ) -> CaseResult:
            async with semaphore:
                return await self._score_case(
                    job[0], job[1], job[2], generate_answer=job[3], evidence_confidence=job[4]
                )

        results = list(await asyncio.gather(*(score_job(job) for job in jobs)))
        answerable_attempted = [
            row for row in results
            if row.expected_answerable and row.generation_status != "not_run"
        ]
        if generate_answers and answerable_attempted and all(
            row.generation_status == "failed" for row in answerable_attempted
        ):
            raise RuntimeError("全部答案生成调用失败；保留上一份评估报告")
        return build_report(results)

    async def _resolve_ground_truth(
        self, cases: Sequence[EvalCase]
    ) -> tuple[dict[str, ResolvedGroundTruth], list[int] | None]:
        resolved: dict[str, ResolvedGroundTruth] = {}
        async with self._session_factory() as session:
            query = select(KnowledgeChunk).where(KnowledgeChunk.vectorize_status == "done")
            if self._evaluation_source_paths is not None:
                query = query.where(
                    KnowledgeChunk.source_kind == "markdown",
                    KnowledgeChunk.source_path.in_(self._evaluation_source_paths),
                )
            chunks = list(await session.scalars(query))
            corpus_chunk_ids = (
                sorted(row.id for row in chunks)
                if self._evaluation_source_paths is not None else None
            )
            if corpus_chunk_ids == []:
                raise ValueError("评测知识库没有已向量化的指定 Markdown 知识块；请先重建并向量化知识库")
            for case in cases:
                ids = set(case.relevant_chunk_ids)
                if case.relevant_questions:
                    ids.update(
                        row.id
                        for row in chunks
                        if row.questions in case.relevant_questions
                    )
                groups: list[tuple[int, ...]] = []
                if case.relevant_section_groups:
                    if self._evaluation_source_paths is not None:
                        matched_groups = matching_evidence_groups(case, chunks)
                    else:
                        matched_groups = [
                            [
                                row for row in chunks
                                if any(_section_matches(alias, row) for alias in aliases)
                            ]
                            for aliases in case.relevant_section_groups
                        ]
                    for matched in matched_groups:
                        group_ids = tuple(
                            sorted({row.id for row in matched})
                        )
                        groups.append(group_ids)
                        ids.update(group_ids)
                else:
                    for section in case.relevant_sections:
                        ids.update(
                            row.id for row in chunks if _section_matches(section, row)
                        )
                if self._evaluation_source_paths is not None and case.expected_answerable and any(not group for group in groups):
                    raise ValueError(f"评测题 {case.case_id} 的期望章节在本次评测知识库中没有对应知识块")
                resolved[case.case_id] = ResolvedGroundTruth(sorted(ids), tuple(groups))
        return resolved, corpus_chunk_ids

    async def _score_case(
        self,
        case: EvalCase,
        ground_truth: ResolvedGroundTruth,
        result: RetrievalResult,
        *,
        generate_answer: bool,
        evidence_confidence: dict | None = None,
    ) -> CaseResult:
        retrieved_ids = [match.chunk.id for match in result.matches]
        citations = [
            {
                "n": index,
                "chunk_id": match.chunk.id,
                "section_path": match.chunk.section_path,
                "question": match.chunk.questions,
                "answer": match.chunk.answer,
                "score": match.score,
            }
            for index, match in enumerate(result.matches, start=1)
        ]
        if evidence_confidence is not None:
            citations = self._evidence_pipeline.evidence(result.matches)
        evidence_text = "\n".join(str(item["answer"]) for item in citations)
        evidence_coverage = point_coverage(evidence_text, case.expected_points)
        answer = ""
        generation_status = "not_run"
        generation_error = ""
        if generate_answer:
            if evidence_confidence is not None and evidence_confidence.get("check_failed"):
                generation_status = "failed"
                generation_error = "EvidenceCheckUnavailable"
            elif not result.matches or (
                result.strategy is RetrievalStrategy.HYBRID_RERANK and not result.confident
            ):
                answer = REFUSAL
                generation_status = "completed"
            elif evidence_confidence is None and not case.expected_answerable and result.strategy is RetrievalStrategy.HYBRID_RERANK:
                sufficient = await self._evidence_checker.check(case.query, citations)
                if sufficient.check_failed:
                    generation_status = "failed"
                    generation_error = "EvidenceCheckUnavailable"
                elif not sufficient.useful:
                    answer = REFUSAL
                    generation_status = "completed"
            if generation_status == "not_run":
                generated, generation_error = await self._try_model_call(
                    lambda: self._generate(case.query, citations)
                )
                if generated is not None:
                    answer = generated
                    generation_status = "completed"
                else:
                    generation_status = "failed"
        refused = answer.strip() == REFUSAL if generation_status == "completed" else None
        answer_coverage: float | None = None
        coverage_status = "not_run"
        if generation_status == "completed" and case.expected_answerable:
            if refused or not case.expected_points:
                answer_coverage = 0.0 if refused else 1.0
                coverage_status = "completed"
            else:
                covered, error = await self._try_model_call(
                    lambda: self._judge_coverage(case.query, answer, case.expected_points)
                )
                if covered is not None:
                    answer_coverage = min(covered.covered_count, len(case.expected_points)) / len(case.expected_points)
                    coverage_status = "completed"
                else:
                    coverage_status = "failed"
                    generation_error = "; ".join(filter(None, (generation_error, f"coverage: {error}")))
        decision: FaithfulnessDecision | None = None
        faithfulness_status = "not_run"
        if (
            generation_status == "completed"
            and case.expected_answerable
            and result.strategy is RetrievalStrategy.HYBRID_RERANK
        ):
            if refused:
                decision = FaithfulnessDecision(
                    faithful=True,
                    score=1.0,
                    reason="答案为明确拒答，没有添加证据外事实",
                )
            else:
                decision, error = await self._try_model_call(
                    lambda: self._judge(case.query, answer, citations)
                )
                if decision is None:
                    generation_error = "; ".join(filter(None, (generation_error, f"faithfulness: {error}")))
            faithfulness_status = "completed" if decision is not None else "failed"
        ledger_error = ""
        if (
            decision is not None
            and not decision.faithful
            and result.strategy is RetrievalStrategy.HYBRID_RERANK
        ):
            try:
                async with self._session_factory() as session, session.begin():
                    await QualityRepository(session).upsert_faith_case(
                        eval_id=case.case_id,
                        bucket=case.query_type,
                        query=case.query,
                        strategy=result.strategy.value,
                        answer=answer,
                        reason=decision.reason,
                        citations=citations,
                        judge_model=self._judge_model,
                    )
            except Exception as error:
                ledger_error = type(error).__name__
        relevant_ids = ground_truth.chunk_ids
        recall, rank_score = (
            group_retrieval_scores(retrieved_ids, ground_truth.groups, k=5)
            if ground_truth.groups
            else (
                recall_at_k(retrieved_ids, relevant_ids, k=5),
                reciprocal_rank(retrieved_ids, relevant_ids),
            )
        )
        return CaseResult(
            case_id=case.case_id,
            query=case.query,
            query_type=case.query_type,
            difficulty=case.difficulty,
            strategy=result.strategy.value,
            relevant_chunk_ids=relevant_ids,
            retrieved_chunk_ids=retrieved_ids,
            recall_at_k=recall,
            reciprocal_rank=rank_score,
            answer=answer,
            refused=refused,
            faithfulness=decision.score if decision is not None else None,
            faithful=decision.faithful if decision is not None else None,
            faithfulness_reason=decision.reason if decision is not None else "",
            citations=citations,
            expected_answerable=case.expected_answerable,
            expected_points=tuple(case.expected_points),
            evidence_coverage=evidence_coverage,
            answer_completeness=(
                point_coverage(answer, case.expected_points)
                if case.expected_answerable
                else float(refused)
            ) if generation_status == "completed" else None,
            relevant_chunk_groups=ground_truth.groups,
            answer_coverage=answer_coverage,
            generation_status=generation_status,
            generation_error=generation_error,
            coverage_status=coverage_status,
            faithfulness_status=faithfulness_status,
            ledger_error=ledger_error,
            evidence_confidence=evidence_confidence,
        )

    async def _try_model_call(
        self, factory: Callable[[], Awaitable[_CallResult]]
    ) -> tuple[_CallResult | None, str]:
        """Retry one bounded model call without turning an outage into a zero score."""

        for attempt in range(2):
            try:
                return await asyncio.wait_for(factory(), timeout=self._call_timeout_seconds), ""
            except Exception as error:
                if attempt == 0:
                    await asyncio.sleep(self._retry_delay_seconds)
                else:
                    return None, type(error).__name__
        raise AssertionError("unreachable")

    async def _judge_coverage(
        self, query: str, answer: str, expected_points: Sequence[str]
    ) -> CoverageDecision:
        runnable = self._model.with_structured_output(
            CoverageDecision, method=self._structured_output_method
        )
        result = await runnable.ainvoke(coverage_messages(query, answer, expected_points))
        return CoverageDecision.model_validate(result)

    async def _generate(
        self, query: str, citations: Sequence[dict[str, object]]
    ) -> str:
        messages = [
            SystemMessage(
                content=(
                    "仅根据证据回答客服问题。每个事实都必须带对应 [n] 引用；"
                    "证据不足时只输出固定拒答。禁止承诺到账、退款、赔付或送达时间。"
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {"query": query, "evidence": list(citations), "refusal": REFUSAL},
                    ensure_ascii=False,
                )
            ),
        ]
        parts: list[str] = []
        async for chunk in self._model.astream(messages):
            text = text_from_chunk(chunk)  # type: ignore[arg-type]
            if text:
                parts.append(text)
        return "".join(parts).strip() or REFUSAL

    async def _judge(
        self,
        query: str,
        answer: str,
        citations: Sequence[dict[str, object]],
    ) -> FaithfulnessDecision:
        runnable = self._model.with_structured_output(
            FaithfulnessDecision,
            method=self._structured_output_method,
        )
        result = await runnable.ainvoke(
            faithfulness_messages(query, answer, citations)
        )
        return FaithfulnessDecision.model_validate(result)


def _normalize_match_text(value: str) -> str:
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE).casefold()


def _section_matches(section: str, row: KnowledgeChunk) -> bool:
    expected = _normalize_match_text(section)
    if not expected:
        return False
    candidates = (row.questions, row.section_path or "")
    return any(
        expected in _normalize_match_text(candidate)
        for candidate in candidates
        if candidate
    )


def matching_section_rows(alias: str, rows: Sequence[_EvidenceRow]) -> list[_EvidenceRow]:
    """Prefer a whole leaf heading; use a unique shorthand only when necessary."""

    label = _normalize_match_text(alias)
    exact = [row for row in rows if _normalize_match_text(row.questions) == label]
    return exact or [
        row for row in rows if label in _normalize_match_text(row.questions)
    ]


def matching_evidence_groups(case: EvalCase, rows: Sequence[_EvidenceRow]) -> list[list[_EvidenceRow]]:
    """Resolve benchmark labels to passages that actually contain a labeled fact.

    A parent heading alone is insufficient: the matched passage must have the
    named leaf heading and support at least one expected point. This remains a
    document-derived GT proxy, not a human passage-level relevance judgment.
    """

    point_labels = [_normalize_match_text(point) for point in case.expected_points]
    groups: list[list[_EvidenceRow]] = []
    for aliases in case.relevant_section_groups:
        candidate_objects = {
            id(row)
            for alias in aliases
            for row in matching_section_rows(alias, rows)
        }
        groups.append([
            row for row in rows
            if id(row) in candidate_objects
            and any(point in _normalize_match_text(row.answer) for point in point_labels)
        ])
    return groups


def point_coverage(text: str, points: Sequence[str]) -> float:
    if not points:
        return 1.0
    normalized = _normalize_match_text(text)
    matched = sum(_normalize_match_text(point) in normalized for point in points)
    return matched / len(points)


def faithfulness_messages(
    query: str,
    answer: str,
    citations: Sequence[dict[str, object]],
) -> list[BaseMessage]:
    """Spell out JSON-mode fields because compatible providers do not infer schemas."""

    return [
        SystemMessage(
            content=(
                "只返回一个 JSON 对象，字段必须且只能是："
                'faithful（布尔值）、score（0 到 1 的数字）、reason（字符串）。'
                "faithful 仅在回答中的全部事实都被给定证据支持时为 true。"
                "遗漏不是编造；新增政策、数值、时间、状态或承诺是编造。"
            )
        ),
        HumanMessage(
            content=json.dumps(
                {"query": query, "answer": answer, "evidence": list(citations)},
                ensure_ascii=False,
            )
        ),
    ]


def coverage_messages(
    query: str, answer: str, expected_points: Sequence[str]
) -> list[BaseMessage]:
    return [
        SystemMessage(
            content=(
                "只返回 JSON 对象，字段 covered_count 为非负整数。"
                "数出客服答案正确覆盖了多少项标准要点；遗漏、答错或编造的要点不计。"
                "不得因为关键词相似就把事实不符的要点计入。"
            )
        ),
        HumanMessage(
            content=json.dumps(
                {"query": query, "answer": answer, "expected_points": list(expected_points)},
                ensure_ascii=False,
            )
        ),
    ]


def _mean_available(values: Sequence[float | None]) -> float | None:
    scored = [value for value in values if value is not None]
    return sum(scored) / len(scored) if scored else None


def _summary(rows: Sequence[CaseResult]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "cases": 0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "evidence_coverage": 0.0,
            "answer_completeness": None,
            "answer_coverage": None,
            "faithfulness": None,
            "refusal_accuracy": None,
            "false_refusal_rate": None,
            "generation_completed": 0,
            "generation_failed": 0,
            "coverage_scored": 0,
            "coverage_failed": 0,
            "faithfulness_scored": 0,
            "faithfulness_failed": 0,
            "refusal_scored": 0,
            "ledger_failed": 0,
        }
    answerable = [row for row in rows if row.expected_answerable]
    refusal = [row for row in rows if not row.expected_answerable]
    generated = [row for row in rows if row.generation_status == "completed"]
    judged = [row for row in rows if row.faithfulness_status == "completed"]
    refusal_scored = [row for row in refusal if row.refused is not None]
    answerable_scored = [row for row in answerable if row.refused is not None]
    return {
        "cases": len(rows),
        "recall_at_k": (
            sum(row.recall_at_k for row in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        "mrr": (
            sum(row.reciprocal_rank for row in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        "evidence_coverage": (
            sum(row.evidence_coverage for row in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        "answer_completeness": _mean_available(
            [row.answer_completeness for row in answerable]
        ),
        "answer_coverage": _mean_available(
            [row.answer_coverage for row in answerable]
        ),
        "faithfulness": _mean_available([row.faithfulness for row in judged]),
        "refusal_accuracy": (
            sum(bool(row.refused) for row in refusal_scored) / len(refusal_scored)
            if refusal_scored else None
        ),
        "false_refusal_rate": (
            sum(bool(row.refused) for row in answerable_scored) / len(answerable_scored)
            if answerable_scored else None
        ),
        "generation_completed": len(generated),
        "generation_failed": sum(row.generation_status == "failed" for row in rows),
        "coverage_scored": sum(row.answer_coverage is not None for row in rows),
        "coverage_failed": sum(row.coverage_status == "failed" for row in rows),
        "faithfulness_scored": len(judged),
        "faithfulness_failed": sum(row.faithfulness_status == "failed" for row in rows),
        "refusal_scored": len(refusal_scored),
        "ledger_failed": sum(bool(row.ledger_error) for row in rows),
    }


def build_report(results: Sequence[CaseResult]) -> dict[str, object]:
    by_strategy: dict[str, list[CaseResult]] = defaultdict(list)
    by_query_type: dict[str, list[CaseResult]] = defaultdict(list)
    by_difficulty: dict[str, list[CaseResult]] = defaultdict(list)
    for row in results:
        by_strategy[row.strategy].append(row)
        by_query_type[f"{row.strategy}:{row.query_type}"].append(row)
        by_difficulty[f"{row.strategy}:{row.difficulty}"].append(row)
    return {
        "dataset": {
            "total_cases": len({row.case_id for row in results}),
            "answerable_cases": len(
                {row.case_id for row in results if row.expected_answerable}
            ),
            "refusal_cases": len(
                {row.case_id for row in results if not row.expected_answerable}
            ),
        },
        "strategies": {key: _summary(value) for key, value in by_strategy.items()},
        "by_query_type": {key: _summary(value) for key, value in by_query_type.items()},
        "by_difficulty": {key: _summary(value) for key, value in by_difficulty.items()},
        "cases": [asdict(row) for row in results],
    }


def render_markdown(report: dict[str, object]) -> str:
    def score(value: object) -> str:
        return "—" if value is None else f"{float(value):.3f}"

    lines = ["# retrieval 四策略检索与生成质量报告", "", "## 总体", ""]
    lines.append("| 策略 | Recall@5 | MRR | 证据覆盖 | 答案语义覆盖 | Faithfulness | 拒答正确率 | 未评上 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    strategies = report.get("strategies", {})
    if isinstance(strategies, dict):
        for name, raw in strategies.items():
            values = raw if isinstance(raw, dict) else {}
            lines.append(
                f"| {name} | {score(values.get('recall_at_k'))} | "
                f"{score(values.get('mrr'))} | "
                f"{score(values.get('evidence_coverage'))} | "
                f"{score(values.get('answer_coverage'))} | "
                f"{score(values.get('faithfulness'))} | "
                f"{score(values.get('refusal_accuracy'))} | "
                f"{values.get('generation_failed', 0)} |"
            )
    for heading, key in [("按 Query 类型", "by_query_type"), ("按难度", "by_difficulty")]:
        lines.extend(["", f"## {heading}", "", "| 分桶 | Recall@K | MRR | Faithfulness |", "|---|---:|---:|---:|"])
        groups = report.get(key, {})
        if isinstance(groups, dict):
            for name, raw in groups.items():
                values = raw if isinstance(raw, dict) else {}
                lines.append(
                    f"| {name} | {score(values.get('recall_at_k'))} | "
                    f"{score(values.get('mrr'))} | {score(values.get('faithfulness'))} |"
                )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval-evaluation.json"
    markdown_path = output_dir / "retrieval-evaluation.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
