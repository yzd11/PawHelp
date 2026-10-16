"""Report metadata must identify every route whose answer was generated."""

from pathlib import Path

from app.evaluation.retrieval_quality import EvalCase
from app.evaluation.runtime import (
    generation_strategy_names,
    report_dataset_alignment,
    report_dataset_mismatch_count,
)
from app.knowledge.search import RetrievalStrategy


def test_all_route_generation_records_explicit_strategy_names() -> None:
    assert generation_strategy_names(
        True,
        None,
        (RetrievalStrategy.DENSE, RetrievalStrategy.BM25),
    ) == ["dense", "bm25"]


def test_saved_report_detects_changed_question_labels_without_rewriting_it() -> None:
    case = EvalCase(
        case_id="A21",
        query="当前问题",
        query_type="A_policy",
        difficulty="easy",
        relevant_sections=["会员等级与积分规则"],
        expected_points=["当前要点"],
        reference_answer="当前要点",
        expected_answerable=True,
    )
    old_report = {
        "cases": [
            {
                "case_id": "A21",
                "query": "旧问题",
                "expected_points": ["旧要点"],
                "expected_answerable": True,
            }
        ]
    }

    assert report_dataset_mismatch_count(old_report, [case]) == 1
    assert old_report["cases"][0]["query"] == "旧问题"


def test_existing_report_with_old_labels_is_marked_stale() -> None:
    cases_path = Path(__file__).parents[1] / "evals" / "customer_service_500.csv"
    old_report = {
        "cases": [
            {
                "case_id": "A21",
                "query": "喵金等级要年消费多少钱,有什么生日礼",
                "expected_points": ["5000 元起", "猫罐头礼盒"],
                "expected_answerable": True,
            }
        ]
    }

    assert report_dataset_alignment(old_report, cases_path) == {
        "status": "stale",
        "changed_cases": 1,
    }
