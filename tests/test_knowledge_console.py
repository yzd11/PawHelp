"""Public knowledge-console page and API workflow."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.database.models import KnowledgeChunk
from app.knowledge.chunking import KnowledgeDraft
from app.knowledge.consistency import KnowledgeConsistencyChecker
from app.main import KNOWLEDGE_PAGE_PATH, create_app
from app.repositories.knowledge import KnowledgeRepository
from app.repositories.knowledge_mining import KnowledgeCandidate, KnowledgeMiningRepository, Transcript
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
    assert synced.json() == {"replaced": 8, "ingested": 10, "vectorized": 10,
                             "rechecked": 0, "flagged": [], "recheck_unavailable": False}
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


class _ScriptedConsistencyModel:
    """Judges a candidate by its question; an exception stands for an unavailable provider."""

    def __init__(self, decisions):
        self.decisions, self.payloads = decisions, []

    def with_structured_output(self, schema, **kwargs):
        return self

    async def ainvoke(self, messages):
        payload = json.loads(messages[-1].content.split("\n", 1)[1])
        self.payloads.append(payload)
        decision = self.decisions[payload["candidate"]["question"]]
        if isinstance(decision, Exception):
            raise decision
        return decision


class _OfficialShippingRuleSearch:
    async def retrieve(self, query, **kwargs):
        chunk = SimpleNamespace(id=1002, source_kind="markdown", section_path="商品与购物 FAQ / 运费怎么算",
                                questions="运费怎么算", answer="单笔订单满 99 元包邮,未满收取 10 元运费。")
        return SimpleNamespace(matches=[SimpleNamespace(chunk=chunk, score=0.8)])


SHIPPING_CONFLICT = {"verdict": "conflict", "existing_id": 1002, "reason": "文档规定满 99 元包邮、未满收 10 元"}


def _checked_service(tmp_path, decisions, session_factory=None):
    model = _ScriptedConsistencyModel(decisions)
    checker = KnowledgeConsistencyChecker(
        _OfficialShippingRuleSearch(), model, method="json_schema")
    service = KnowledgeAdminService(
        session_factory or _DuplicateSessionFactory(), None, None, None,  # type: ignore[arg-type]
        documents_root=tmp_path, vectorize_batch_size=16, consistency_checker=checker,
        conversation_extractor=SimpleNamespace(extract=None),  # type: ignore[arg-type]
    )
    return service, model


@pytest.mark.parametrize(("decision", "status", "code"), [
    (SHIPPING_CONFLICT, 409, "knowledge_conflict"),
    (ConnectionError("test-only provider outage"), 503, "knowledge_check_unavailable"),
])
def test_manual_entry_contradicting_a_document_is_refused_before_any_write(
    monkeypatch, tmp_path: Path, decision, status, code
) -> None:
    class NoInsertRepository:
        async def list_identities(self, *, exclude_id=None):
            return set()

        async def add_drafts(self, *args, **kwargs):
            raise AssertionError("refused knowledge must not be inserted")

    monkeypatch.setattr(knowledge_admin_module, "KnowledgeRepository", lambda session: NoInsertRepository())
    service, model = _checked_service(tmp_path, {"邮费是多少": decision})
    app = create_app(settings=settings(), model=FakeCombinedModel(),
                     session_factory=UnusedSessionFactory(),  # type: ignore[arg-type]
                     knowledge_admin_service=service)

    with TestClient(app) as client:
        response = client.post("/v1/knowledge/chunks", json={
            "category": "配送服务", "questions": "邮费是多少", "answer": "没有固定的邮费金额，以结算页为准。"})

    assert response.status_code == status
    detail = response.json()["detail"]
    assert detail["code"] == code
    assert model.payloads[0]["existing"][0]["source"] == "document"
    if status == 409:
        assert "#1002" in detail["message"] and "运费怎么算" in detail["message"]


@pytest.mark.asyncio
async def test_mined_entry_contradicting_a_document_is_staged_as_discarded_with_its_reason(
    monkeypatch, tmp_path: Path
) -> None:
    staged = []

    class RecordingSession(_AsyncContext):
        def add_all(self, rows):
            staged.extend(rows)

        async def flush(self):
            return None

    async def transcripts(self, *, limit):
        return [Transcript(7, ())]

    async def promote(self):
        return sum(row.status == "extracted" for row in staged), 0

    async def no_empty_sources(self, batch_no, conversation_ids):
        return None

    repository = KnowledgeMiningRepository
    monkeypatch.setattr(repository, "load_unstaged_transcripts", transcripts)
    monkeypatch.setattr(repository, "deduplicate_and_promote", promote)
    monkeypatch.setattr(repository, "add_empty_sources", no_empty_sources)
    mined = [KnowledgeCandidate(7, "邮费是多少", "没有固定的邮费金额。"),
             KnowledgeCandidate(7, "猫窝能机洗吗", "可以，冷水轻柔模式。")]

    async def extract(rows):
        return mined

    new = {"verdict": "new", "existing_id": None, "reason": "文档未涉及"}
    service, _ = _checked_service(tmp_path, {"邮费是多少": SHIPPING_CONFLICT, "猫窝能机洗吗": new},
                                  session_factory=lambda: RecordingSession())
    service._conversation_extractor = SimpleNamespace(extract=extract)

    result = await service.mine_conversations(vectorize_after=False)

    assert (result.kept, result.discarded) == (1, 1)
    assert [(row.question, row.status) for row in staged] == [("邮费是多少", "discarded"), ("猫窝能机洗吗", "extracted")]
    assert "#1002" in staged[0].discard_reason and staged[1].discard_reason is None

    staged.clear()
    service, _ = _checked_service(tmp_path, {"邮费是多少": ConnectionError("outage"), "猫窝能机洗吗": new},
                                  session_factory=lambda: RecordingSession())
    service._conversation_extractor = SimpleNamespace(extract=extract)
    with pytest.raises(knowledge_admin_module.KnowledgeCheckUnavailable):
        await service.mine_conversations(vectorize_after=False)
    assert staged == []


@pytest.mark.asyncio
@pytest.mark.parametrize("shipping_decision", [SHIPPING_CONFLICT, ConnectionError("test-only provider outage")])
async def test_rebuild_reports_kept_entries_the_rebuilt_documents_contradict(
    monkeypatch, tmp_path: Path, shipping_decision
) -> None:
    kept = [SimpleNamespace(id=111, questions="邮费是多少", answer="没有固定的邮费金额。"),
            SimpleNamespace(id=514, questions="猫窝能机洗吗", answer="可以，冷水轻柔模式。")]
    deleted = []

    class RebuildRepository:
        async def list_replaceable_imports(self):
            return []

        async def mark_pending(self, rows):
            return None

        async def delete_ids(self, ids):
            deleted.extend(ids)

        async def list_identities(self, *, exclude_id=None):
            return set()

        async def list_kept_entries(self):
            return kept

    async def nothing_pending(*args, **kwargs):
        return 0

    monkeypatch.setattr(knowledge_admin_module, "KnowledgeRepository", lambda session: RebuildRepository())
    monkeypatch.setattr(knowledge_admin_module, "vectorize_pending", nothing_pending)
    new = {"verdict": "new", "existing_id": None, "reason": "文档未涉及"}
    class CommittingSession(_AsyncContext):
        async def commit(self):
            return None

    service, _ = _checked_service(tmp_path, {"邮费是多少": shipping_decision, "猫窝能机洗吗": new},
                                  session_factory=lambda: CommittingSession())
    service._vector_store = SimpleNamespace(delete=lambda ids: None)

    result = await service.sync_materials()

    assert result.rechecked == 2 and deleted == []
    if isinstance(shipping_decision, Exception):
        assert result.recheck_unavailable and result.flagged == []
    else:
        assert not result.recheck_unavailable
        assert [(flag.chunk_id, flag.verdict) for flag in result.flagged] == [(111, "conflict")]
        assert "#1002" in result.flagged[0].message
