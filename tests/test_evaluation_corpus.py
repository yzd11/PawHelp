from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evaluation.corpus import EVALUATION_SOURCES, validate_evaluation_corpus
from app.knowledge.chunking import chunk_markdown
from app.knowledge.search import build_filter_expression


ROOT = Path(__file__).resolve().parents[1]


def _indexed_rows():
    rows = []
    for relative_path, content_type in EVALUATION_SOURCES.items():
        text = (ROOT / "knowledge" / relative_path).read_text(encoding="utf-8")
        for draft in chunk_markdown(text, content_type=content_type):
            rows.append(SimpleNamespace(
                id=len(rows) + 1,
                questions=draft.questions,
                answer=draft.answer,
                source_kind="markdown",
                source_path=relative_path,
                vectorize_status="done",
            ))
    return rows


def test_benchmark_scope_excludes_retained_manual_history() -> None:
    rows = _indexed_rows()
    rows.append(SimpleNamespace(
        id=1000, questions="发票", answer="旧版发票规则",
        source_kind="manual", source_path=None, vectorize_status="done",
    ))

    allowed = validate_evaluation_corpus(rows)

    assert len(allowed) == 97
    assert 1000 not in allowed
    assert build_filter_expression([], [], allowed).startswith('id in ["1","2"')


def test_stale_or_pending_benchmark_corpus_fails_before_model_calls() -> None:
    rows = _indexed_rows()
    rows[0].answer = "过期政策"
    with pytest.raises(ValueError, match="缺失或未更新"):
        validate_evaluation_corpus(rows)

    rows = _indexed_rows()
    rows[0].vectorize_status = "pending"
    with pytest.raises(ValueError, match="待向量化 1 块"):
        validate_evaluation_corpus(rows)


def test_empty_allowed_corpus_fails_closed() -> None:
    with pytest.raises(ValueError, match="no vectorized knowledge chunks"):
        build_filter_expression([], [], [])
