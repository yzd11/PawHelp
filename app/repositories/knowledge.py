"""MySQL authority operations for knowledge chunks."""

from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import KnowledgeChunk
from app.knowledge.chunking import KnowledgeDraft


CURATED_CONTENT_TYPES = ("faq", "policy", "manual")


def knowledge_identity(question: str, answer: str) -> tuple[str, str]:
    """Build the same whitespace-insensitive identity for every ingestion path."""

    return (
        " ".join(question.split()).casefold(),
        " ".join(answer.split()).casefold(),
    )


class KnowledgeAlreadyIngested(RuntimeError):
    """Raised when a source-less schema cannot safely identify a repeated import."""


class KnowledgeRepository:
    """Persist knowledge drafts and maintain within-document neighbor pointers."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def has_curated_knowledge(self) -> bool:
        count = await self._session.scalar(
            select(func.count(KnowledgeChunk.id)).where(
                KnowledgeChunk.content_type.in_(CURATED_CONTENT_TYPES)
            )
        )
        return bool(count)

    async def list_pending(self, *, limit: int) -> list[KnowledgeChunk]:
        """Return one stable batch that still needs a Milvus vector."""

        rows = await self._session.scalars(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
            .limit(limit)
        )
        return list(rows)

    async def list_after_id(
        self, *, after_id: int, limit: int
    ) -> list[KnowledgeChunk]:
        """Page through every authoritative chunk for a collection rebuild."""

        rows = await self._session.scalars(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.id > after_id)
            .order_by(KnowledgeChunk.id)
            .limit(limit)
        )
        return list(rows)

    async def list_vectorized_after_id(
        self, *, after_id: int, limit: int
    ) -> list[KnowledgeChunk]:
        """Page through rows that claim to have a Milvus counterpart."""

        rows = await self._session.scalars(
            select(KnowledgeChunk)
            .where(
                KnowledgeChunk.id > after_id,
                KnowledgeChunk.vectorize_status == "done",
            )
            .order_by(KnowledgeChunk.id)
            .limit(limit)
        )
        return list(rows)

    async def list_chunks(
        self, *, status: str | None = None, offset: int = 0, limit: int = 20
    ) -> list[KnowledgeChunk]:
        """List one deterministic page for the local knowledge console."""

        statement = select(KnowledgeChunk)
        if status is not None:
            statement = statement.where(KnowledgeChunk.vectorize_status == status)
        rows = await self._session.scalars(
            statement
            .order_by(KnowledgeChunk.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(rows)

    async def count_chunks_by_status(self) -> dict[str, int]:
        """Return whole-library totals without loading every chunk."""

        result = await self._session.execute(
            select(
                KnowledgeChunk.vectorize_status,
                func.count(KnowledgeChunk.id),
            ).group_by(KnowledgeChunk.vectorize_status)
        )
        return {str(status): int(count) for status, count in result.all()}

    async def list_identities(
        self, *, exclude_id: int | None = None
    ) -> set[tuple[str, str]]:
        """Return lightweight identities used by dry-run and idempotent ingestion."""

        statement = select(KnowledgeChunk.questions, KnowledgeChunk.answer)
        if exclude_id is not None:
            statement = statement.where(KnowledgeChunk.id != exclude_id)
        result = await self._session.execute(statement)
        return {
            knowledge_identity(question, answer)
            for question, answer in result.all()
        }

    async def mark_vectorized(self, rows: Sequence[KnowledgeChunk]) -> None:
        """Record the stable Milvus primary key after a successful upsert."""

        for row in rows:
            row.vector_id = str(row.id)
            row.vectorize_status = "done"
        await self._session.flush()

    async def get_vectorized_by_ids(
        self, vector_ids: Sequence[str]
    ) -> list[KnowledgeChunk]:
        """Return authoritative rows in Milvus ranking order."""

        if not vector_ids:
            return []
        rows = await self._session.scalars(
            select(KnowledgeChunk).where(
                KnowledgeChunk.vector_id.in_(vector_ids),
                KnowledgeChunk.vectorize_status == "done",
            )
        )
        by_vector_id = {row.vector_id: row for row in rows}
        return [by_vector_id[value] for value in vector_ids if value in by_vector_id]

    async def get_vectorized(self, chunk_id: int) -> KnowledgeChunk | None:
        return await self._session.scalar(
            select(KnowledgeChunk).where(
                KnowledgeChunk.id == chunk_id,
                KnowledgeChunk.vectorize_status == "done",
            )
        )

    async def get_by_id(self, chunk_id: int) -> KnowledgeChunk | None:
        """Return authoritative state regardless of vectorization status."""

        return await self._session.get(KnowledgeChunk, chunk_id)

    async def list_replaceable_imports(self) -> list[KnowledgeChunk]:
        """List only imports whose provenance is known to be replaceable."""

        rows = await self._session.scalars(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.source_kind.in_(("markdown", "faq_table")))
            .order_by(KnowledgeChunk.id)
        )
        return list(rows)

    async def mark_pending(self, rows: Sequence[KnowledgeChunk]) -> None:
        """Hide stale vectors from retrieval before replacing or deleting them."""

        for row in rows:
            row.vector_id = None
            row.vectorize_status = "pending"
        await self._session.flush()

    async def delete_ids(self, chunk_ids: Sequence[int]) -> None:
        if not chunk_ids:
            return
        await self._session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.id.in_(chunk_ids))
        )
        await self._session.flush()

    async def add_drafts(
        self,
        drafts: Sequence[KnowledgeDraft],
        *,
        link_neighbors: bool,
        source_kind: str = "manual",
        source_path: str | None = None,
    ) -> list[KnowledgeChunk]:
        rows = [
            KnowledgeChunk(
                category=draft.category,
                questions=draft.questions,
                answer=draft.answer,
                section_path=draft.section_path,
                content_type=draft.content_type,
                source_kind=source_kind,
                source_path=source_path,
                is_key_clause=draft.is_key_clause,
                vectorize_status="pending",
            )
            for draft in drafts
        ]
        self._session.add_all(rows)
        await self._session.flush()
        if link_neighbors:
            for index, row in enumerate(rows):
                row.prev_chunk_id = rows[index - 1].id if index else None
                row.next_chunk_id = rows[index + 1].id if index + 1 < len(rows) else None
            await self._session.flush()
        return rows
