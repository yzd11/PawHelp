"""One-shot curated knowledge ingestion from Markdown files."""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge.chunking import chunk_markdown
from app.repositories.knowledge import KnowledgeAlreadyIngested, KnowledgeRepository


_DIRECTORY_TYPES = {
    "policies": "policy",
    "product-faq": "faq",
    "manuals": "manual",
}


async def ingest_curated_knowledge(
    session: AsyncSession, documents_root: Path
) -> int:
    """Persist one curated corpus, refusing an ambiguous repeated import."""

    repository = KnowledgeRepository(session)
    if await repository.has_curated_knowledge():
        raise KnowledgeAlreadyIngested(
            "curated knowledge already exists; clear it explicitly before re-importing"
        )

    inserted = 0
    for path in sorted(documents_root.rglob("*.md")):
        try:
            relative = path.relative_to(documents_root)
        except ValueError:
            continue
        content_type = _DIRECTORY_TYPES.get(relative.parts[0]) if relative.parts else None
        if content_type is None:
            continue
        drafts = chunk_markdown(path.read_text(encoding="utf-8"), content_type=content_type)
        inserted += len(
            await repository.add_drafts(
                drafts,
                link_neighbors=True,
                source_kind="markdown",
                source_path=relative.as_posix(),
            )
        )
    return inserted
