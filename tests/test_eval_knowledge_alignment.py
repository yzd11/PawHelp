"""Keep answerable evaluation labels grounded in the active knowledge documents."""

from pathlib import Path
from types import SimpleNamespace

from app.evaluation.corpus import validate_evaluation_cases
from app.evaluation.retrieval_quality import (
    EvalCase,
    load_cases,
    matching_evidence_groups,
)


ROOT = Path(__file__).resolve().parents[1]


def test_answerable_eval_points_exist_in_their_labeled_knowledge_sections() -> None:
    cases = load_cases(ROOT / "evals" / "customer_service_500.csv")
    validate_evaluation_cases(cases)


def test_refund_arrival_labels_do_not_treat_handling_time_as_arrival_time() -> None:
    cases = load_cases(ROOT / "evals" / "customer_service_500.csv")
    confused = [
        case.case_id
        for case in cases
        if "到账" in case.query
        and any("3 个工作日" in point for point in case.expected_points)
    ]

    assert not confused, f"退款到账问题误用了处理时限: {confused}"


def test_ground_truth_requires_leaf_heading_and_labeled_fact() -> None:
    case = EvalCase(
        case_id="example", query="不足包邮门槛的运费", query_type="A_policy",
        difficulty="easy", relevant_section_groups=[["运费怎么算"]],
        expected_points=["未满收取 10 元运费"], reference_answer="未满收取 10 元运费",
        expected_answerable=True,
    )
    rows = [
        SimpleNamespace(id=1, questions="运费怎么算", section_path="商品与购物 FAQ / 运费怎么算", answer="未满收取 10 元运费"),
        SimpleNamespace(id=2, questions="配送范围与偏远地区", section_path="商品与购物 FAQ / 运费怎么算 / 配送范围与偏远地区", answer="未满收取 10 元运费"),
        SimpleNamespace(id=3, questions="运费怎么算", section_path="商品与购物 FAQ / 运费怎么算", answer="满 99 元包邮"),
    ]

    assert [[row.id for row in group] for group in matching_evidence_groups(case, rows)] == [[1]]


def test_curated_labels_keep_comparisons_and_warranty_evidence_separate() -> None:
    cases = {case.case_id: case for case in load_cases(ROOT / "evals" / "customer_service_500.csv")}

    assert len(cases["B55"].relevant_section_groups) == 2
    assert "差多少" not in cases["B55"].query
    assert cases["E5"].relevant_section_groups == [["保修说明"], ["维修寄修流程"]]


def test_exact_exchange_freight_heading_wins_over_parent_substring() -> None:
    case = EvalCase(
        case_id="exchange", query="换货运费谁出", query_type="A_policy",
        difficulty="easy", relevant_section_groups=[["换货运费"]],
        expected_points=["往返运费由平台承担"],
        reference_answer="往返运费由平台承担", expected_answerable=True,
    )
    rows = [
        SimpleNamespace(id=1, questions="退换货运费承担", answer="往返运费由平台承担"),
        SimpleNamespace(id=2, questions="换货运费", answer="往返运费由平台承担"),
    ]

    assert [[row.id for row in group] for group in matching_evidence_groups(case, rows)] == [[2]]
