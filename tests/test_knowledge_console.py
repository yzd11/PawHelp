"""Public knowledge-console page and API workflow."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.database.models import KnowledgeChunk
from app.knowledge.chunking import KnowledgeDraft
from app.main import KNOWLEDGE_PAGE_PATH, create_app
from app.repositories.knowledge import KnowledgeRepository
from app.schemas import (
    KnowledgeMaterial,
    KnowledgeMiningResult,
    KnowledgeMiningStats,
    KnowledgeOverview,
    KnowledgeSyncResult,
    MarkdownKnowledgeIngestResult,
    MarkdownKnowledgePreviewItem,
    MarkdownKnowledgePreviewResult,
    ManualKnowledgeRequest,
)
import app.services.knowledge_admin as knowledge_admin_module
from app.services.knowledge_admin import (
    KnowledgeAdminService,
    KnowledgeMaterialNotFound,
    dual_write_is_consistent,
)
from tests.fakes import FakeCombinedModel


class UnusedSessionFactory:
    def __call__(self) -> object:
        raise AssertionError("the injected knowledge service owns this test boundary")


def knowledge_row(*, status: str = "pending") -> SimpleNamespace:
    return SimpleNamespace(
        id=9,
        category="配送服务",
        questions="邮费在哪里看？",
        answer="提交订单前可在结算页查看实际运费。",
        section_path="配送服务 / 配送费用",
        content_type="manual",
        is_key_clause=False,
        vector_id="9" if status == "done" else None,
        vectorize_status=status,
        source_kind="manual",
        source_path=None,
    )


class FakeKnowledgeAdmin:
    def __init__(self) -> None:
        self.created = None
        self.list_status = None
        self.updated = None
        self.deleted = None

    def list_materials(self):
        return [
            KnowledgeMaterial(
                path="policies/refund.md",
                name="refund.md",
                content_type="policy",
                size=128,
            )
        ]

    def preview(self, path: str):
        assert path == "policies/refund.md"
        return [
            KnowledgeDraft(
                category="售后政策",
                questions="退货条件",
                answer="符合条件的商品可以申请退货。",
                section_path="售后政策 / 退货条件",
                content_type="policy",
            )
        ]

    async def overview(self):
        return KnowledgeOverview(
            materials=1,
            mysql_total=3,
            pending_total=1,
            done_total=2,
            milvus_total=2,
            milvus_collection="knowledge_hybrid",
            dual_write_consistent=True,
        )

    async def preview_markdown(self, request):
        return MarkdownKnowledgePreviewResult(
            items=[
                MarkdownKnowledgePreviewItem(
                    category="发票",
                    questions="开票时效",
                    answer="三个工作日内开具。",
                    section_path="发票 / 开票时效",
                    content_type=request.content_type,
                    is_key_clause=True,
                    is_table_block=False,
                    duplicate=False,
                )
            ],
            total=1,
            key_clause_count=1,
            table_block_count=0,
            duplicate_count=0,
            ingestible_count=1,
        )

    async def ingest_markdown(self, request):
        return MarkdownKnowledgeIngestResult(
            inserted=1,
            skipped_duplicates=0,
            vectorized=1 if request.vectorize_after else 0,
        )

    async def mining_stats(self):
        return KnowledgeMiningStats(
            unprocessed_conversations=2,
            extracted=3,
            kept=2,
            discarded=1,
            batches=1,
        )

    async def mine_conversations(self, *, vectorize_after: bool):
        return KnowledgeMiningResult(
            conversations=2,
            extracted=3,
            kept=2,
            discarded=1,
            vectorized=2 if vectorize_after else 0,
        )

    async def list_chunks(self, *, status: str | None, page: int, page_size: int):
        self.list_status = (status, page, page_size)
        return [knowledge_row()], {"pending": 1, "done": 2}

    async def create_manual(self, request):
        self.created = request
        return knowledge_row()

    async def update_chunk(self, chunk_id, request):
        self.updated = (chunk_id, request)
        return knowledge_row(status="done")

    async def delete_chunk(self, chunk_id):
        self.deleted = chunk_id

    async def sync_materials(self):
        return KnowledgeSyncResult(replaced=8, ingested=10, vectorized=10)

    async def vectorize_pending(self):
        return 2

    async def search(self, query: str, **kwargs):
        assert query == "邮费是多少"
        assert kwargs["strategy"] == "hybrid_rerank"
        assert kwargs["limit"] == 5
        return SimpleNamespace(
            matches=(
                SimpleNamespace(chunk=knowledge_row(status="done"), score=0.91),
            ),
            strategy=SimpleNamespace(value="hybrid_rerank"),
            confident=True,
            query_plan=SimpleNamespace(
                normalized_query="运费是多少",
                synonyms=("邮费",),
            ),
        )


def settings() -> Settings:
    return Settings(
        chat_base_url="http://localhost:11434/v1",
        openai_model="qwen3",
        openai_api_key="ollama",
        database_url="mysql+asyncmy://pawhelp:pawhelp@127.0.0.1:3306/pawhelp",
    )


def test_knowledge_console_page_and_complete_api_workflow() -> None:
    service = FakeKnowledgeAdmin()
    app = create_app(
        settings=settings(),
        model=FakeCombinedModel(),
        session_factory=UnusedSessionFactory(),  # type: ignore[arg-type]
        knowledge_admin_service=service,  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        page = client.get("/kb")
        materials = client.get("/v1/knowledge/materials")
        preview = client.post(
            "/v1/knowledge/preview", json={"path": "policies/refund.md"}
        )
        overview = client.get("/v1/knowledge/overview")
        markdown_preview = client.post(
            "/v1/knowledge/preview-markdown",
            json={"markdown": "# 发票\n\n## 开票时效\n\n三个工作日内开具。", "content_type": "policy"},
        )
        markdown_ingest = client.post(
            "/v1/knowledge/ingest-markdown",
            json={"markdown": "# 发票\n\n## 开票时效\n\n三个工作日内开具。", "content_type": "policy"},
        )
        mining_stats = client.get("/v1/knowledge/mining/stats")
        mining = client.post(
            "/v1/knowledge/mining/run",
            json={"confirm_external_model_usage": True, "vectorize_after": True},
        )
        created = client.post(
            "/v1/knowledge/chunks",
            json={
                "category": "配送服务",
                "questions": "邮费在哪里看？",
                "answer": "提交订单前可在结算页查看实际运费。",
            },
        )
        updated = client.put(
            "/v1/knowledge/chunks/9",
            json={
                "category": "配送服务",
                "questions": "最新邮费在哪里看？",
                "answer": "请以结算页最新费用为准。",
            },
        )
        deleted = client.delete("/v1/knowledge/chunks/9")
        synced = client.post("/v1/knowledge/sync-materials", json={})
        chunks = client.get(
            "/v1/knowledge/chunks?status=pending&page=1&page_size=25"
        )
        vectorized = client.post("/v1/knowledge/vectorize-pending", json={})
        searched = client.post(
            "/v1/knowledge/search", json={"query": "邮费是多少"}
        )

    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    assert page.content == KNOWLEDGE_PAGE_PATH.read_bytes()
    assert "知识库工作台" in page.text
    assert "切块预览" in page.text
    assert "重建离线库" in page.text
    assert "编辑" in page.text
    assert "删除" in page.text
    assert materials.json()[0]["path"] == "policies/refund.md"
    assert preview.json()[0]["questions"] == "退货条件"
    assert overview.json()["dual_write_consistent"] is True
    assert markdown_preview.json()["ingestible_count"] == 1
    assert markdown_ingest.json()["vectorized"] == 1
    assert mining_stats.json()["unprocessed_conversations"] == 2
    assert mining.json()["kept"] == 2
    assert created.status_code == 201
    assert created.json()["vectorize_status"] == "pending"
    assert service.created.questions == "邮费在哪里看？"
    assert updated.json()["vectorize_status"] == "done"
    assert service.updated[0] == 9
    assert service.updated[1].questions == "最新邮费在哪里看？"
    assert deleted.json() == {"deleted": 9}
    assert service.deleted == 9
    assert synced.json() == {"replaced": 8, "ingested": 10, "vectorized": 10}
    assert chunks.json()["items"][0]["id"] == 9
    assert chunks.json()["total"] == 1
    assert chunks.json()["total_pages"] == 1
    assert chunks.json()["all_total"] == 3
    assert chunks.json()["pending_total"] == 1
    assert chunks.json()["done_total"] == 2
    assert service.list_status == ("pending", 1, 25)
    assert vectorized.json() == {"vectorized": 2}
    assert searched.json()["matches"][0]["vectorize_status"] == "done"
    assert searched.json()["matches"][0]["score"] == 0.91
    assert searched.json()["normalized_query"] == "运费是多少"


def test_material_preview_is_structured_and_confined_to_knowledge_root(
    tmp_path: Path,
) -> None:
    policies = tmp_path / "policies"
    policies.mkdir()
    material = policies / "refund.md"
    material.write_text(
        "# 售后政策\n\n## 退货条件\n\n符合条件的商品可以申请退货。",
        encoding="utf-8",
    )
    service = KnowledgeAdminService(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        documents_root=tmp_path,
        vectorize_batch_size=16,
    )

    materials = service.list_materials()
    preview = service.preview("policies/refund.md")

    assert [row.path for row in materials] == ["policies/refund.md"]
    assert materials[0].character_count == len(material.read_text(encoding="utf-8"))
    assert preview[0].category == "售后政策"
    assert preview[0].questions == "退货条件"
    with pytest.raises(KnowledgeMaterialNotFound):
        service.preview("../outside.md")


def test_dual_write_consistency_requires_no_pending_rows() -> None:
    assert dual_write_is_consistent({"pending": 0, "done": 3}, 3) is True
    assert dual_write_is_consistent({"pending": 1, "done": 3}, 3) is False
    assert dual_write_is_consistent({"pending": 0, "done": 3}, None) is None


class _AsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def begin(self):
        return _AsyncContext()


class _DuplicateSessionFactory:
    def __call__(self):
        return _AsyncContext()


@pytest.mark.asyncio
async def test_manual_create_rejects_an_existing_exact_identity(
    monkeypatch, tmp_path: Path
) -> None:
    class DuplicateRepository:
        async def list_identities(self, *, exclude_id=None):
            return {("邮费是多少", "请以结算页为准。")}

        async def add_drafts(self, *args, **kwargs):
            raise AssertionError("duplicate knowledge must not be inserted")

    monkeypatch.setattr(
        knowledge_admin_module,
        "KnowledgeRepository",
        lambda session: DuplicateRepository(),
    )
    service = KnowledgeAdminService(
        _DuplicateSessionFactory(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        documents_root=tmp_path,
        vectorize_batch_size=16,
    )

    with pytest.raises(knowledge_admin_module.KnowledgeDuplicateError):
        await service.create_manual(
            ManualKnowledgeRequest(
                category="配送服务",
                questions="邮费是多少",
                answer="请以结算页为准。",
            )
        )


@pytest.mark.asyncio
async def test_material_sync_scope_never_replaces_manual_or_conversation_knowledge(
    mysql_session,
) -> None:
    rows = [
        KnowledgeChunk(
            category="同步保护测试",
            questions=f"来源 {source_kind}",
            answer="仅用于验证同步范围。",
            content_type="conversation" if source_kind == "conversation" else "faq",
            source_kind=source_kind,
            vectorize_status="done",
        )
        for source_kind in ("markdown", "faq_table", "manual", "conversation")
    ]
    mysql_session.add_all(rows)
    await mysql_session.flush()

    replaceable = await KnowledgeRepository(mysql_session).list_replaceable_imports()
    replaceable_ids = {row.id for row in replaceable}

    assert rows[0].id in replaceable_ids
    assert rows[1].id in replaceable_ids
    assert rows[2].id not in replaceable_ids
    assert rows[3].id not in replaceable_ids
