"""Small local knowledge-console service over the existing RAG components."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
import re
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.models import KnowledgeChunk
from app.knowledge.chunking import KnowledgeDraft, chunk_markdown
from app.knowledge.embedding import BgeM3Embedder
from app.knowledge.mining import ConversationKnowledgeExtractor
from app.knowledge.search import (
    RetrievalResult,
    RetrievalStrategy,
    SemanticKnowledgeSearch,
)
from app.knowledge.vector_store import MilvusKnowledgeStore, milvus_call
from app.knowledge.vectorization import reconcile_missing_vectors, vectorize_pending
from app.repositories.knowledge import KnowledgeRepository
from app.repositories.knowledge import knowledge_identity
from app.repositories.knowledge_mining import KnowledgeMiningRepository
from app.schemas import (
    KnowledgeMaterial,
    KnowledgeMiningResult,
    KnowledgeMiningStats,
    KnowledgeOverview,
    KnowledgeSyncResult,
    MarkdownKnowledgeIngestResult,
    MarkdownKnowledgePreviewItem,
    MarkdownKnowledgePreviewResult,
    MarkdownKnowledgeRequest,
    ManualKnowledgeRequest,
)


DIRECTORY_CONTENT_TYPES = {
    "policies": "policy",
    "product-faq": "faq",
    "manuals": "manual",
}
_TABLE_DIVIDER = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")


def _is_table_block(answer: str) -> bool:
    lines = answer.splitlines()
    return len(lines) > 1 and bool(_TABLE_DIVIDER.match(lines[1]))


def _duplicate_flags(
    drafts: list[KnowledgeDraft], existing: set[tuple[str, str]]
) -> list[bool]:
    seen = set(existing)
    flags: list[bool] = []
    for draft in drafts:
        identity = knowledge_identity(draft.questions, draft.answer)
        duplicate = identity in seen
        flags.append(duplicate)
        seen.add(identity)
    return flags


def dual_write_is_consistent(
    counts: dict[str, int], milvus_total: int | None
) -> bool | None:
    """Require every authoritative row to have exactly one Milvus counterpart."""

    if milvus_total is None:
        return None
    mysql_total = sum(counts.values())
    return (
        counts.get("pending", 0) == 0
        and counts.get("done", 0) == mysql_total
        and milvus_total == mysql_total
    )


class KnowledgeMaterialNotFound(ValueError):
    """Raised when a preview path is not an allowed local Markdown source."""


class KnowledgeChunkNotFound(ValueError):
    """Raised when a cited authoritative chunk is missing or not vectorized."""


class KnowledgeDuplicateError(ValueError):
    """Raised when a write repeats an existing question-and-answer identity."""


class KnowledgeAdminService:
    """Expose curated sources, pending writes, vectorization, and search."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: BgeM3Embedder,
        vector_store: MilvusKnowledgeStore,
        search: SemanticKnowledgeSearch,
        *,
        documents_root: Path,
        vectorize_batch_size: int,
        conversation_extractor: ConversationKnowledgeExtractor | None = None,
        conversation_batch_size: int = 20,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._vector_store = vector_store
        self._search = search
        self._documents_root = documents_root.resolve()
        self._vectorize_batch_size = vectorize_batch_size
        self._conversation_extractor = conversation_extractor
        self._conversation_batch_size = conversation_batch_size
        self._mutation_lock = asyncio.Lock()

    def list_materials(self) -> list[KnowledgeMaterial]:
        """List only supported Markdown files below the configured root."""

        materials: list[KnowledgeMaterial] = []
        if not self._documents_root.exists():
            return materials
        for path in sorted(self._documents_root.rglob("*.md")):
            relative = path.relative_to(self._documents_root)
            content_type = DIRECTORY_CONTENT_TYPES.get(relative.parts[0])
            if content_type is None:
                continue
            text = path.read_text(encoding="utf-8")
            drafts = chunk_markdown(text, content_type=content_type)
            materials.append(
                KnowledgeMaterial(
                    path=relative.as_posix(),
                    name=path.name,
                    content_type=content_type,
                    size=path.stat().st_size,
                    character_count=len(text),
                    line_count=len(text.splitlines()),
                    chunk_count=len(drafts),
                    key_clause_count=sum(row.is_key_clause for row in drafts),
                    table_chunk_count=sum(
                        _is_table_block(row.answer) for row in drafts
                    ),
                )
            )
        return materials

    def preview(self, relative_path: str) -> list[KnowledgeDraft]:
        """Chunk one allowed material without writing to MySQL."""

        relative = Path(relative_path)
        if relative.is_absolute():
            raise KnowledgeMaterialNotFound(relative_path)
        candidate = (self._documents_root / relative).resolve()
        try:
            normalized = candidate.relative_to(self._documents_root)
        except ValueError as error:
            raise KnowledgeMaterialNotFound(relative_path) from error
        content_type = (
            DIRECTORY_CONTENT_TYPES.get(normalized.parts[0])
            if normalized.parts
            else None
        )
        if content_type is None or candidate.suffix.lower() != ".md" or not candidate.is_file():
            raise KnowledgeMaterialNotFound(relative_path)
        return chunk_markdown(
            candidate.read_text(encoding="utf-8"),
            content_type=content_type,
        )

    async def preview_markdown(
        self, request: MarkdownKnowledgeRequest
    ) -> MarkdownKnowledgePreviewResult:
        drafts = chunk_markdown(
            request.markdown,
            content_type=request.content_type,
        )
        async with self._session_factory() as session:
            existing = await KnowledgeRepository(session).list_identities()
        flags = _duplicate_flags(drafts, existing)
        items = [
            MarkdownKnowledgePreviewItem(
                category=draft.category,
                questions=draft.questions,
                answer=draft.answer,
                section_path=draft.section_path,
                content_type=draft.content_type,  # type: ignore[arg-type]
                is_key_clause=draft.is_key_clause,
                is_table_block=_is_table_block(draft.answer),
                duplicate=duplicate,
            )
            for draft, duplicate in zip(drafts, flags, strict=True)
        ]
        duplicate_count = sum(flags)
        return MarkdownKnowledgePreviewResult(
            items=items,
            total=len(items),
            key_clause_count=sum(item.is_key_clause for item in items),
            table_block_count=sum(item.is_table_block for item in items),
            duplicate_count=duplicate_count,
            ingestible_count=len(items) - duplicate_count,
        )

    async def ingest_markdown(
        self, request: MarkdownKnowledgeRequest
    ) -> MarkdownKnowledgeIngestResult:
        drafts = chunk_markdown(
            request.markdown,
            content_type=request.content_type,
        )
        async with self._mutation_lock:
            async with self._session_factory() as session, session.begin():
                repository = KnowledgeRepository(session)
                flags = _duplicate_flags(drafts, await repository.list_identities())
                novel = [
                    draft
                    for draft, duplicate in zip(drafts, flags, strict=True)
                    if not duplicate
                ]
                if novel:
                    await repository.add_drafts(
                        novel,
                        link_neighbors=True,
                        source_kind="manual_markdown",
                    )
            vectorized = (
                await self._vectorize_pending_unlocked()
                if request.vectorize_after and novel
                else 0
            )
        return MarkdownKnowledgeIngestResult(
            inserted=len(novel),
            skipped_duplicates=sum(flags),
            vectorized=vectorized,
        )

    async def list_chunks(
        self, *, status: str | None, page: int, page_size: int
    ) -> tuple[list[KnowledgeChunk], dict[str, int]]:
        async with self._session_factory() as session:
            repository = KnowledgeRepository(session)
            rows = await repository.list_chunks(
                status=status,
                offset=(page - 1) * page_size,
                limit=page_size,
            )
            counts = await repository.count_chunks_by_status()
            return rows, counts

    async def overview(self) -> KnowledgeOverview:
        async with self._session_factory() as session:
            counts = await KnowledgeRepository(session).count_chunks_by_status()
        done = counts.get("done", 0)
        try:
            milvus_total: int | None = await milvus_call(
                self._vector_store.count_entities
            )
        except Exception:
            milvus_total = None
        return KnowledgeOverview(
            materials=len(self.list_materials()),
            mysql_total=sum(counts.values()),
            pending_total=counts.get("pending", 0),
            done_total=done,
            milvus_total=milvus_total,
            milvus_collection=self._vector_store.collection_name,
            dual_write_consistent=dual_write_is_consistent(counts, milvus_total),
        )

    async def mining_stats(self) -> KnowledgeMiningStats:
        async with self._session_factory() as session:
            values = await KnowledgeMiningRepository(session).stats()
        return KnowledgeMiningStats(**values)

    async def mine_conversations(self, *, vectorize_after: bool) -> KnowledgeMiningResult:
        if self._conversation_extractor is None:
            raise RuntimeError("conversation knowledge extractor is not configured")
        async with self._mutation_lock:
            async with self._session_factory() as session:
                transcripts = await KnowledgeMiningRepository(
                    session
                ).load_unstaged_transcripts(limit=self._conversation_batch_size)
            if not transcripts:
                return KnowledgeMiningResult(
                    conversations=0,
                    extracted=0,
                    kept=0,
                    discarded=0,
                    vectorized=0,
                )
            candidates = await self._conversation_extractor.extract(transcripts)
            batch_no = f"{datetime.now(UTC):%Y%m%d%H%M%S}-{uuid4().hex[:8]}"
            async with self._session_factory() as session, session.begin():
                repository = KnowledgeMiningRepository(session)
                await repository.add_extracted(batch_no, candidates)
                represented = {
                    candidate.source_conversation_id for candidate in candidates
                }
                await repository.add_empty_sources(
                    batch_no,
                    [
                        transcript.conversation_id
                        for transcript in transcripts
                        if transcript.conversation_id not in represented
                    ],
                )
                kept, discarded = await repository.deduplicate_and_promote()
            vectorized = (
                await self._vectorize_pending_unlocked()
                if vectorize_after and kept
                else 0
            )
        return KnowledgeMiningResult(
            conversations=len(transcripts),
            extracted=len(candidates),
            kept=kept,
            discarded=discarded,
            vectorized=vectorized,
        )

    async def create_manual(self, request: ManualKnowledgeRequest) -> KnowledgeChunk:
        draft = KnowledgeDraft(
            category=request.category,
            questions=request.questions,
            answer=request.answer,
            section_path=request.section_path,
            content_type=request.content_type,
            is_key_clause=request.is_key_clause,
        )
        async with self._mutation_lock:
            async with self._session_factory() as session, session.begin():
                repository = KnowledgeRepository(session)
                if knowledge_identity(draft.questions, draft.answer) in (
                    await repository.list_identities()
                ):
                    raise KnowledgeDuplicateError("相同问法和正文的知识已经存在")
                rows = await repository.add_drafts(
                    [draft],
                    link_neighbors=False,
                    source_kind="manual",
                )
        return rows[0]

    async def update_chunk(
        self, chunk_id: int, request: ManualKnowledgeRequest
    ) -> KnowledgeChunk:
        """Replace one chunk, then upsert its stable Milvus primary key."""

        async with self._mutation_lock:
            async with self._session_factory() as session, session.begin():
                repository = KnowledgeRepository(session)
                row = await repository.get_by_id(chunk_id)
                if row is None:
                    raise KnowledgeChunkNotFound(chunk_id)
                identity = knowledge_identity(request.questions, request.answer)
                if identity in await repository.list_identities(exclude_id=chunk_id):
                    raise KnowledgeDuplicateError("相同问法和正文的知识已经存在")
                row.category = request.category
                row.questions = request.questions
                row.answer = request.answer
                row.section_path = request.section_path
                row.content_type = request.content_type
                row.is_key_clause = request.is_key_clause
                if row.source_kind == "legacy":
                    row.source_kind = "manual"
                await repository.mark_pending([row])
            await self._vectorize_pending_unlocked()
            return await self._get_any_chunk(chunk_id)

    async def delete_chunk(self, chunk_id: int) -> None:
        """Hide, delete from Milvus, then remove the MySQL authority row."""

        async with self._mutation_lock:
            async with self._session_factory() as session, session.begin():
                repository = KnowledgeRepository(session)
                row = await repository.get_by_id(chunk_id)
                if row is not None:
                    await repository.mark_pending([row])
            await milvus_call(self._vector_store.delete, [str(chunk_id)])
            async with self._session_factory() as session, session.begin():
                await KnowledgeRepository(session).delete_ids([chunk_id])

    async def sync_materials(self) -> KnowledgeSyncResult:
        """Replace Markdown imports and remove legacy FAQ-table vectors."""

        markdown_sources = [
            (material.path, self.preview(material.path))
            for material in self.list_materials()
        ]
        async with self._mutation_lock:
            async with self._session_factory() as session, session.begin():
                repository = KnowledgeRepository(session)
                stale_rows = await repository.list_replaceable_imports()
                stale_ids = [row.id for row in stale_rows]
                await repository.mark_pending(stale_rows)

            await milvus_call(
                self._vector_store.delete,
                [str(chunk_id) for chunk_id in stale_ids],
            )

            async with self._session_factory() as session, session.begin():
                repository = KnowledgeRepository(session)
                await repository.delete_ids(stale_ids)
                existing = await repository.list_identities()
                inserted = 0
                for source_path, drafts in markdown_sources:
                    flags = _duplicate_flags(drafts, existing)
                    novel = [
                        draft
                        for draft, duplicate in zip(drafts, flags, strict=True)
                        if not duplicate
                    ]
                    inserted += len(
                        await repository.add_drafts(
                            novel,
                            link_neighbors=True,
                            source_kind="markdown",
                            source_path=source_path,
                        )
                    )
                    existing.update(
                        knowledge_identity(draft.questions, draft.answer)
                        for draft in novel
                    )

            vectorized = await self._vectorize_pending_unlocked()
            return KnowledgeSyncResult(
                replaced=len(stale_ids),
                ingested=inserted,
                vectorized=vectorized,
            )

    async def get_chunk(self, chunk_id: int) -> KnowledgeChunk:
        async with self._session_factory() as session:
            row = await KnowledgeRepository(session).get_vectorized(chunk_id)
        if row is None:
            raise KnowledgeChunkNotFound(chunk_id)
        return row

    async def _get_any_chunk(self, chunk_id: int) -> KnowledgeChunk:
        async with self._session_factory() as session:
            row = await KnowledgeRepository(session).get_by_id(chunk_id)
        if row is None:
            raise KnowledgeChunkNotFound(chunk_id)
        return row

    async def _vectorize_pending_unlocked(self, *, reconcile: bool = False) -> int:
        async with self._session_factory() as session:
            repository = KnowledgeRepository(session)
            if reconcile:
                await reconcile_missing_vectors(
                    repository,
                    self._vector_store,
                    session.commit,
                    batch_size=self._vectorize_batch_size,
                )
            return await vectorize_pending(
                repository,
                self._embedder,
                self._vector_store,
                session.commit,
                batch_size=self._vectorize_batch_size,
            )

    async def vectorize_pending(self) -> int:
        """Drain pending rows once; concurrent button presses share one run."""

        async with self._mutation_lock:
            return await self._vectorize_pending_unlocked(reconcile=True)

    async def search(
        self,
        query: str,
        *,
        strategy: str,
        limit: int,
        categories: list[str],
        content_types: list[str],
    ) -> RetrievalResult:
        return await self._search.retrieve(
            query,
            strategy=RetrievalStrategy(strategy),
            limit=limit,
            categories=categories,
            content_types=content_types,
        )
