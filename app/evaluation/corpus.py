"""The fixed, document-backed corpus for the retrieval benchmark."""

from collections import Counter
from collections.abc import Sequence
import hashlib
from pathlib import Path

from app.database.models import KnowledgeChunk
from app.evaluation.retrieval_quality import (
    EvalCase,
    _normalize_match_text,
    matching_evidence_groups,
    matching_section_rows,
)
from app.knowledge.chunking import chunk_markdown
from app.repositories.knowledge import knowledge_identity


EVALUATION_SOURCES = {
    "manuals/after-sales.md": "manual",
    "manuals/app-devices.md": "manual",
    "manuals/camera-guide.md": "manual",
    "manuals/error-codes.md": "manual",
    "manuals/feeder-guide.md": "manual",
    "manuals/fountain-care.md": "manual",
    "manuals/heating-and-cat-tree.md": "manual",
    "manuals/litter-box-care.md": "manual",
    "policies/account-security.md": "policy",
    "policies/invoice-rules.md": "policy",
    "policies/logistics-exceptions.md": "policy",
    "policies/points-coupons.md": "policy",
    "policies/returns.md": "policy",
    "product-faq/accessories.md": "faq",
    "product-faq/models.md": "faq",
    "product-faq/shipping.md": "faq",
}
KNOWLEDGE_ROOT = Path(__file__).resolve().parents[2] / "knowledge"


def evaluation_corpus_sha256(knowledge_root: Path = KNOWLEDGE_ROOT) -> str:
    digest = hashlib.sha256()
    for relative_path in EVALUATION_SOURCES:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((knowledge_root / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_evaluation_corpus(
    rows: Sequence[KnowledgeChunk], *, knowledge_root: Path = KNOWLEDGE_ROOT
) -> list[int]:
    """Fail before model calls when the indexed benchmark text differs from disk.

    Manual and conversation knowledge remain in the product index, but are not
    eligible to supply benchmark ground truth or retrieval results.
    """

    expected = Counter()
    for relative_path, content_type in EVALUATION_SOURCES.items():
        path = knowledge_root / relative_path
        drafts = chunk_markdown(path.read_text(encoding="utf-8"), content_type=content_type)
        expected.update(
            (relative_path, knowledge_identity(draft.questions, draft.answer))
            for draft in drafts
        )

    scoped = [
        row for row in rows
        if row.source_kind == "markdown" and row.source_path in EVALUATION_SOURCES
    ]
    actual = Counter(
        (row.source_path, knowledge_identity(row.questions, row.answer))
        for row in scoped if row.vectorize_status == "done"
    )
    if actual != expected or len(scoped) != sum(expected.values()):
        missing = sum((expected - actual).values())
        extra = sum((actual - expected).values())
        pending = sum(row.vectorize_status != "done" for row in scoped)
        raise ValueError(
            "评测知识库与当前评测 Markdown 不一致："
            f"缺失或未更新 {missing} 块，多余或旧版 {extra} 块，待向量化 {pending} 块。"
            "请先在知识库工作台同步材料并补齐向量，再运行评测。"
        )
    return sorted(row.id for row in scoped)


def validate_evaluation_cases(
    cases: Sequence[EvalCase], *, knowledge_root: Path = KNOWLEDGE_ROOT
) -> None:
    """Check every answer label against its own named source sections."""

    sourced_drafts = [
        (relative_path, draft)
        for relative_path, content_type in EVALUATION_SOURCES.items()
        for draft in chunk_markdown(
            (knowledge_root / relative_path).read_text(encoding="utf-8"),
            content_type=content_type,
        )
    ]
    drafts = [draft for _, draft in sourced_drafts]
    problems = []
    for case in cases:
        if not case.expected_answerable:
            if case.expected_points or case.relevant_section_groups:
                problems.append(f"{case.case_id}: 拒答题不应有标准答案或章节")
            continue
        matched_groups = matching_evidence_groups(case, drafts)
        if not case.expected_points or not matched_groups or any(not group for group in matched_groups):
            problems.append(f"{case.case_id}: 缺少标准要点或找不到期望章节")
            continue
        for aliases in case.relevant_section_groups:
            for alias in aliases:
                matching_objects = {id(row) for row in matching_section_rows(alias, drafts)}
                sections = {
                    (path, draft.questions)
                    for path, draft in sourced_drafts
                    if id(draft) in matching_objects
                }
                if len(sections) != 1:
                    problems.append(
                        f"{case.case_id}: 章节别名 {alias} 对应 {len(sections)} 个末级章节"
                    )
        evidence = _normalize_match_text(
            "\n".join(draft.answer for group in matched_groups for draft in group)
        )
        for point in case.expected_points:
            if _normalize_match_text(point) not in evidence:
                problems.append(f"{case.case_id}: 要点无精确章节证据：{point}")
    if problems:
        raise ValueError("评测集与现行知识文档未对齐：" + "；".join(problems[:8]))
