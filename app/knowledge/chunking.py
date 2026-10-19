"""Structure-aware Markdown conversion into retrievable knowledge drafts."""

from dataclasses import dataclass
import re


_HEADERS = [("#" * level, f"h{level}") for level in range(1, 7)]
_SENTENCE_ENDS = "。！？；.!?;"
_TABLE_DIVIDER = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")


@dataclass(frozen=True)
class KnowledgeDraft:
    """One not-yet-persisted knowledge chunk."""

    category: str
    questions: str
    answer: str
    section_path: str | None
    content_type: str
    is_key_clause: bool = False


def _section_identity(metadata: dict[str, object]) -> tuple[str, str, str]:
    titles = [
        str(metadata[key]).strip()
        for _, key in _HEADERS
        if metadata.get(key) and str(metadata[key]).strip()
    ]
    if not titles:
        return "未分类", "未命名章节", "未命名章节"
    question = titles[-1]
    category = " / ".join(titles[:-1]) or titles[0]
    return category, question, " / ".join(titles)


def _is_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and bool(_TABLE_DIVIDER.match(lines[index + 1]))
    )


def _segments(text: str) -> list[tuple[str, str, bool]]:
    """Separate prose, important callouts, and Markdown tables."""

    lines = text.splitlines()
    result: list[tuple[str, str, bool]] = []
    prose: list[str] = []

    def flush_prose() -> None:
        value = "\n".join(prose).strip()
        if value:
            result.append(("prose", value, False))
        prose.clear()

    index = 0
    while index < len(lines):
        if _is_table_start(lines, index):
            flush_prose()
            table = [lines[index].strip(), lines[index + 1].strip()]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table.append(lines[index].strip())
                index += 1
            result.append(("table", "\n".join(table), False))
            continue

        if lines[index].strip().upper() == "> [!IMPORTANT]":
            flush_prose()
            index += 1
            important: list[str] = []
            while index < len(lines) and lines[index].lstrip().startswith(">"):
                important.append(lines[index].lstrip()[1:].lstrip())
                index += 1
            value = "\n".join(important).strip()
            if value:
                result.append(("prose", value, True))
            continue

        prose.append(lines[index])
        index += 1

    flush_prose()
    return result


def _complete_sentence_overlap(previous: str, budget: int) -> str:
    if budget <= 0 or not previous:
        return ""
    tail = previous[-budget:]
    boundary = next((index for index, char in enumerate(tail) if char in _SENTENCE_ENDS), None)
    if boundary is None or boundary == len(tail) - 1:
        return ""
    return tail[boundary + 1 :].lstrip()


def _split_prose(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text.strip()]
    body_size = max(1, chunk_size - chunk_overlap)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=body_size,
        chunk_overlap=0,
        keep_separator="end",
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
    )
    bodies = [part.strip() for part in splitter.split_text(text) if part.strip()]
    result: list[str] = []
    for body in bodies:
        overlap = _complete_sentence_overlap(result[-1], chunk_overlap) if result else ""
        result.append(f"{overlap}{body}" if overlap else body)
    return result


def _split_table(text: str, chunk_size: int) -> list[str]:
    lines = text.splitlines()
    header = lines[:2]
    rows = lines[2:]
    if not rows:
        return [text]
    chunks: list[str] = []
    current = list(header)
    for row in rows:
        candidate = "\n".join([*current, row])
        if len(current) > 2 and len(candidate) > chunk_size:
            chunks.append("\n".join(current))
            current = [*header, row]
        else:
            current.append(row)
    chunks.append("\n".join(current))
    return chunks


def chunk_markdown(
    markdown: str,
    *,
    content_type: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[KnowledgeDraft]:
    """Convert one Markdown document into self-contained knowledge chunks."""

    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and exceed chunk_overlap")
    # The package __init__ imports sentence-transformers (~16s); keep it off app start-up.
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS,
        strip_headers=True,
    )
    drafts: list[KnowledgeDraft] = []
    for document in splitter.split_text(markdown):
        category, question, section_path = _section_identity(document.metadata)
        for kind, content, is_key_clause in _segments(document.page_content):
            answers = (
                _split_table(content, chunk_size)
                if kind == "table"
                else _split_prose(content, chunk_size, chunk_overlap)
            )
            drafts.extend(
                KnowledgeDraft(
                    category=category,
                    questions=question,
                    answer=answer,
                    section_path=section_path,
                    content_type=content_type,
                    is_key_clause=is_key_clause,
                )
                for answer in answers
                if answer
            )
    return drafts
