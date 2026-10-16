from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evaluation.retrieval_quality import (
    CaseResult,
    CoverageDecision,
    EvalCase,
    FaithfulnessDecision,
    ResolvedGroundTruth,
    RetrievalQualityEvaluator,
    build_report,
    coverage_messages,
    faithfulness_messages,
    load_cases,
    point_coverage,
    recall_at_k,
    reciprocal_rank,
    render_markdown,
)
from app.knowledge.query_understanding import QueryPlan
from app.knowledge.evidence import EvidenceDecision
from app.knowledge.search import KnowledgeMatch, RetrievalResult, RetrievalStrategy


async def test_cross_document_case_scores_required_evidence_groups() -> None:
    """One alternative section must not inflate the denominator or mask missing groups."""

    case = next(
        case
        for case in load_cases(
            Path(__file__).parents[1] / "evals" / "customer_service_500.csv"
        )
        if case.case_id == "E2"
    )
    chunks = [
        SimpleNamespace(id=1, questions="退货政策", section_path="退货政策", category="退货"),
        SimpleNamespace(id=2, questions="适用范围", section_path="适用范围", category="退货"),
        SimpleNamespace(id=3, questions="退换货运费承担", section_path="退换货运费承担", category="退货"),
        SimpleNamespace(id=4, questions="退款时效", section_path="退款时效", category="退货"),
    ]
    for row in chunks:
        row.answer = row.questions

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalars(self, _query):
            return chunks

    class Search:
        async def retrieve(self, _query, *, strategy, **_kwargs):
            return RetrievalResult(
                matches=tuple(KnowledgeMatch(row, 1.0) for row in (chunks[0], chunks[2], chunks[3])),
                query_plan=QueryPlan(normalized_query=case.query),
                strategy=strategy,
                confident=True,
            )

    evaluator = RetrievalQualityEvaluator(
        search=Search(),
        model=object(),
        session_factory=Session,
        structured_output_method="json_mode",
        judge_model="unused",
    )
    report = await evaluator.run(
        [case],
        strategies=[RetrievalStrategy.HYBRID],
        generate_answers=False,
    )

    result = report["cases"][0]
    assert result["recall_at_k"] == 1.0
    assert result["reciprocal_rank"] == (1.0 + 0.5 + 1.0 / 3.0) / 3.0
    assert result["relevant_chunk_groups"] == ((1, 2), (3,), (4,))
    assert case.relevant_section_groups == [
        ["退货政策", "适用范围"],
        ["退换货运费承担"],
        ["退款时效"],
    ]


def test_evaluator_exposes_generation_and_judge_steps() -> None:
    assert callable(RetrievalQualityEvaluator._generate)
    assert callable(RetrievalQualityEvaluator._judge)


def test_retrieval_metrics_use_labeled_relevant_chunks() -> None:
    assert recall_at_k([9, 2, 7], [2, 7], k=2) == 0.5
    assert reciprocal_rank([9, 2, 7], [2, 7]) == 0.5


def test_committed_eval_set_covers_required_query_types_and_difficulties() -> None:
    path = Path(__file__).parents[1] / "evals" / "customer_service_500.csv"
    cases = load_cases(path)

    assert len(cases) == 500
    assert {case.query_type for case in cases} == {
        "A_policy",
        "B_model",
        "C_colloquial",
        "D_absent",
        "E_multi",
        "F_identifier",
    }
    assert {case.difficulty for case in cases} == {"easy", "medium", "hard"}
    assert sum(case.expected_answerable for case in cases) == 410
    assert cases[0].relevant_sections == ["退货政策是什么"]
    assert cases[0].expected_points == [
        "7 天无理由退货", "商品需保持完好、不影响二次销售"
    ]
    e4 = next(case for case in cases if case.case_id == "E4")
    assert e4.relevant_section_groups == [
        ["换货流程怎么走"],
        ["换货条件"],
        ["换货运费"],
    ]


def test_report_contains_four_strategy_comparison_and_buckets() -> None:
    rows = [
        CaseResult(
            case_id=f"{strategy}-1",
            query="问题",
            query_type="policy",
            difficulty="easy",
            strategy=strategy,
            relevant_chunk_ids=[1],
            retrieved_chunk_ids=[1],
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            answer="答案 [1]",
            refused=False,
            faithfulness=1.0,
            faithful=True,
            faithfulness_reason="supported",
            citations=[],
        )
        for strategy in ["dense", "bm25", "hybrid", "hybrid_rerank"]
    ]

    report = build_report(rows)
    markdown = render_markdown(report)

    assert set(report["strategies"]) == {"dense", "bm25", "hybrid", "hybrid_rerank"}
    assert "dense:policy" in report["by_query_type"]
    assert "hybrid_rerank:easy" in report["by_difficulty"]
    assert "| hybrid_rerank | 1.000 | 1.000 |" in markdown


def test_refusal_accuracy_uses_dataset_label_instead_of_bucket_name() -> None:
    row = CaseResult(
        case_id="D1",
        query="库外问题",
        query_type="D_absent",
        difficulty="hard",
        strategy="hybrid_rerank",
        relevant_chunk_ids=[],
        retrieved_chunk_ids=[],
        recall_at_k=0.0,
        reciprocal_rank=0.0,
        answer="明确拒答",
        refused=True,
        faithfulness=1.0,
        faithful=True,
        faithfulness_reason="没有编造",
        citations=[],
        expected_answerable=False,
        answer_completeness=1.0,
    )

    report = build_report([row])

    assert report["strategies"]["hybrid_rerank"]["refusal_accuracy"] == 1.0


def test_expected_point_coverage_ignores_spacing_and_punctuation() -> None:
    assert point_coverage("满99元包邮，不满收取10元运费。", ["满 99 元包邮", "10 元运费"]) == 1.0


def test_faithfulness_json_mode_prompt_names_the_exact_schema_fields() -> None:
    text = " ".join(
        str(message.content)
        for message in faithfulness_messages("问题", "答案 [1]", [])
    )

    assert "JSON" in text
    assert "faithful" in text
    assert "score" in text
    assert "reason" in text
    assert "supported" not in text


def test_coverage_prompt_requires_correct_semantic_points() -> None:
    prompt = " ".join(str(message.content) for message in coverage_messages(
        "多久退款", "三个工作日", ["退款到原支付渠道"]
    ))

    assert "covered_count" in prompt
    assert "正确覆盖" in prompt
    assert "退款到原支付渠道" in prompt


@pytest.mark.asyncio
async def test_model_failure_keeps_retrieval_result_without_faking_zero_score() -> None:
    row = SimpleNamespace(id=1, questions="退款时效", answer="三天", section_path="退款时效")

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalars(self, _query):
            return [row]

    class Search:
        async def retrieve(self, query, *, strategy, **_kwargs):
            return RetrievalResult(
                matches=(KnowledgeMatch(row, 1.0),),
                query_plan=QueryPlan(normalized_query=query),
                strategy=strategy,
                confident=True,
            )

    evaluator = RetrievalQualityEvaluator(
        search=Search(), model=object(), session_factory=Session,
        structured_output_method="json_mode", judge_model="unused",
        retry_delay_seconds=0,
    )
    attempts: dict[str, int] = {}

    async def generate(query, _citations):
        attempts[query] = attempts.get(query, 0) + 1
        if query == "失败题":
            raise TimeoutError("upstream unavailable")
        return "三天 [1]"

    async def coverage(_query, _answer, _points):
        return CoverageDecision(covered_count=1)

    async def judge(_query, _answer, _citations):
        return FaithfulnessDecision(faithful=True, score=1.0, reason="有证据")

    evaluator._generate = generate
    evaluator._judge_coverage = coverage
    evaluator._judge = judge
    cases = [
        EvalCase(
            case_id=case_id, query=query, query_type="A_policy", difficulty="easy",
            relevant_chunk_ids=[1], expected_points=["三天"],
            reference_answer="三天", expected_answerable=True,
        )
        for case_id, query in [("A1", "失败题"), ("A2", "成功题")]
    ]

    report = await evaluator.run(cases, strategies=[RetrievalStrategy.HYBRID_RERANK])
    failed, passed = report["cases"]

    assert attempts == {"失败题": 2, "成功题": 1}
    assert failed["reciprocal_rank"] == 1.0
    assert failed["generation_status"] == "failed"
    assert failed["answer_coverage"] is None
    assert failed["refused"] is None
    assert passed["answer_coverage"] == 1.0
    assert report["strategies"]["hybrid_rerank"]["answer_coverage"] == 1.0
    assert report["strategies"]["hybrid_rerank"]["generation_failed"] == 1


@pytest.mark.asyncio
async def test_all_generation_failures_do_not_publish_a_false_zero_report() -> None:
    row = SimpleNamespace(id=1, questions="退款时效", answer="三天", section_path="退款时效")

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalars(self, _query):
            return [row]

    class Search:
        async def retrieve(self, query, *, strategy, **_kwargs):
            return RetrievalResult(
                matches=(KnowledgeMatch(row, 1.0),),
                query_plan=QueryPlan(normalized_query=query),
                strategy=strategy,
                confident=True,
            )

    evaluator = RetrievalQualityEvaluator(
        search=Search(), model=object(), session_factory=Session,
        structured_output_method="json_mode", judge_model="unused",
        retry_delay_seconds=0,
    )

    async def fail(_query, _citations):
        raise TimeoutError("upstream unavailable")

    evaluator._generate = fail
    case = EvalCase(
        case_id="A1", query="退款时效", query_type="A_policy", difficulty="easy",
        relevant_chunk_ids=[1], expected_points=["三天"],
        reference_answer="三天", expected_answerable=True,
    )
    absent = EvalCase(
        case_id="D1", query="库外问题", query_type="D_absent", difficulty="hard",
        expected_points=[], reference_answer="拒答", expected_answerable=False,
    )

    with pytest.raises(RuntimeError, match="全部答案生成调用失败"):
        await evaluator.run([case, absent], strategies=[RetrievalStrategy.HYBRID_RERANK])


def test_evidence_checker_failure_is_distinct_from_genuine_insufficient_evidence() -> None:
    assert EvidenceDecision(useful=False, reason="证据不足").check_failed is False
    assert EvidenceDecision(
        useful=False, reason="模型不可用", check_failed=True
    ).check_failed is True


@pytest.mark.asyncio
async def test_ledger_write_failure_does_not_discard_scored_case() -> None:
    row = SimpleNamespace(id=1, questions="退款时效", answer="三天", section_path="退款时效")

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def begin(self):
            raise RuntimeError("ledger unavailable")

    evaluator = RetrievalQualityEvaluator(
        search=object(), model=object(), session_factory=Session,
        structured_output_method="json_mode", judge_model="unused",
        retry_delay_seconds=0,
    )

    async def generate(_query, _citations):
        return "没有证据的承诺 [1]"

    async def coverage(_query, _answer, _points):
        return CoverageDecision(covered_count=0)

    async def judge(_query, _answer, _citations):
        return FaithfulnessDecision(faithful=False, score=0.0, reason="无证据")

    evaluator._generate = generate
    evaluator._judge_coverage = coverage
    evaluator._judge = judge
    case = EvalCase(
        case_id="A1", query="退款时效", query_type="A_policy", difficulty="easy",
        relevant_chunk_ids=[1], expected_points=["三天"],
        reference_answer="三天", expected_answerable=True,
    )
    result = RetrievalResult(
        matches=(KnowledgeMatch(row, 1.0),),
        query_plan=QueryPlan(normalized_query=case.query),
        strategy=RetrievalStrategy.HYBRID_RERANK,
        confident=True,
    )

    scored = await evaluator._score_case(
        case, ResolvedGroundTruth([1]), result, generate_answer=True
    )

    assert scored.reciprocal_rank == 1.0
    assert scored.faithful is False
    assert scored.ledger_error == "RuntimeError"
