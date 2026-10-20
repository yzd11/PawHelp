"""Observability and knowledge-flywheel contracts. Real MySQL, offline models."""
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import Settings
from app.database.models import Conversation, Message, LowConfidenceQuestion, ReviewQueue, KnowledgeChunk, EvalRun, RequestObservation
from app.flywheel import Flywheel
from app.observability import UsageCollector, Observability
from app.knowledge.confidence import calibrate, EvidenceGate
from app.evaluation.pipeline import metrics_from_report, cost_by_intent, CallBudget, ModelCallBudgetExceeded
from app.main import create_app
from app.api.operations import operations_router
from tests.fakes import FakeChatModel, fake_settings as config
from tests.workflow_fakes import run_example


class GapModel:
    def __init__(self): self.calls, self.question = 0, "测试接口 " + uuid4().hex
    def with_structured_output(self, schema, **kwargs): return self
    async def ainvoke(self, messages):
        self.calls += 1
        payload = json.loads(messages[-1].content)
        return {"normalized_question": self.question, "ai_suggested_answer": "证据中没有说明，请人工核准。", "matched_review_id": next(iter(payload["pending_candidates"]), {}).get("id")}


class Publishing:
    def __init__(self, sessions): self.sessions, self.fail, self.chunk_id = sessions, True, None
    async def assert_consistent(self, question, answer, **_): pass
    async def vectorize_pending(self):
        if self.fail: raise ConnectionError("test-only vector failure")
        async with self.sessions() as session, session.begin():
            rows = await session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id == self.chunk_id))
            for row in rows: row.vectorize_status = "done"


async def test_feedback_exact_turn_owned_idempotent_and_snapshot(mysql_session):
    sessions = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    user = str(uuid4())
    snapshot = [{"chunk_id": 123, "question": "旧问", "answer": "当轮原文，不是更新后的知识", "score": .9}]
    async with sessions() as session, session.begin():
        conversation = Conversation(user_id=user)
        session.add(conversation); await session.flush()
        q1=Message(conversation_id=conversation.id, role="user", content="第一轮问题")
        session.add(q1); await session.flush()
        answer=Message(conversation_id=conversation.id, role="assistant", content="第一轮回答", reply_to_message_id=q1.id, retrieved_chunks=snapshot)
        session.add(answer); await session.flush()
        session.add(Message(conversation_id=conversation.id, role="user", content="不要错误获取的第二轮问题"))
    app = create_app(settings=config(), model=FakeChatModel(), session_factory=sessions)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            path = f"/v1/conversations/{conversation.id}/messages/{answer.id}/feedback"
            forged = await client.post(path, headers={"X-Demo-User-Id": str(uuid4())}, json={"vote":"down"})
            assert forged.status_code == 404
            first = await client.post(path, headers={"X-Demo-User-Id": user}, json={"vote":"down"})
            repeat = await client.post(path, headers={"X-Demo-User-Id": user}, json={"vote":"down"})
            assert first.status_code == repeat.status_code == 200
            assert repeat.json()["already_recorded"] is True
            assert (await client.get("/api/review")).status_code == 401
            assert (await client.get("/api/review", headers={"X-Admin-Token":"test-only-admin"})).status_code == 200
        async with sessions() as session:
            rows = list(await session.scalars(select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == conversation.id)))
            assert len(rows) == 1 and rows[0].raw_question == "第一轮问题"
            assert rows[0].retrieved_chunks == snapshot
    finally:
        async with sessions() as session, session.begin():
            await session.execute(delete(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == conversation.id))
            await session.execute(delete(Message).where(Message.conversation_id == conversation.id))
            await session.execute(delete(Conversation).where(Conversation.id == conversation.id))


async def test_dedup_atomic_mapping_approve_sync_failure_retry(mysql_session):
    sessions = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    publication = Publishing(sessions)
    model = GapModel()
    flywheel = Flywheel(config(), sessions, model, publication)
    app = FastAPI()
    app.include_router(operations_router(config(), sessions, flywheel, SimpleNamespace()))
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers={"X-Admin-Token":"test-only-admin"})
    # Constrain this test's batch to its own raw rows, so existing user problems are untouched.
    async with sessions() as session, session.begin():
        a = LowConfidenceQuestion(raw_question="test-"+uuid4().hex, source="self_check")
        b = LowConfidenceQuestion(raw_question="test-"+uuid4().hex, source="user_feedback")
        session.add_all([a,b]); await session.flush()
    ids = [a.id,b.id]
    review_id = chunk_id = None
    try:
        # Exercise the same processor via a session adapter filtering the work query only.
        from unittest.mock import patch
        real_scalars = __import__('sqlalchemy.ext.asyncio', fromlist=['AsyncSession']).AsyncSession.scalars
        async def scoped_scalars(session, statement, *args, **kwargs):
            if statement.column_descriptions[0].get("name") == "id" and statement.column_descriptions[0].get("entity") is LowConfidenceQuestion:
                statement = statement.where(LowConfidenceQuestion.id.in_(ids))
            elif statement.column_descriptions[0].get("entity") is ReviewQueue:
                statement = statement.where(ReviewQueue.normalized_question == model.question)
            return await real_scalars(session, statement, *args, **kwargs)
        with patch("sqlalchemy.ext.asyncio.AsyncSession.scalars", scoped_scalars):
            response = await client.post("/api/review/process")
            assert response.status_code == 200
            result = response.json()
            assert result["processed"] == 2 and model.calls == 2
            assert (await client.post("/api/review/process")).json()["processed"] == 0
        async with sessions() as session:
            pool_rows = list(await session.scalars(select(LowConfidenceQuestion).where(LowConfidenceQuestion.id.in_(ids))))
            assert pool_rows[0].matched_review_id == pool_rows[1].matched_review_id
            review_id = pool_rows[0].matched_review_id
            row = await session.get(ReviewQueue, review_id)
            assert row.occurrence_count == 2
        assert (await client.post(f"/api/review/{review_id}/approve", json={"approved_answer":" "})).status_code == 409
        response = await client.post(f"/api/review/{review_id}/approve", json={"approved_answer":"仅测试核准答案，不作为真实政策"})
        assert response.status_code == 200
        result = response.json()
        chunk_id = result["published_chunk_id"]
        assert result["review_status"] == "通过" and result["publication_status"] == "待同步"
        assert result["publication_error"]
        assert (await client.post(f"/api/review/{review_id}/approve", json={"approved_answer":"重复审核"})).status_code == 409
        publication.fail = False
        publication.chunk_id = chunk_id
        response = await client.post(f"/api/review/{review_id}/publish")
        assert response.status_code == 200
        result = response.json()
        assert result["publication_status"] == "已同步" and result["published_chunk_id"] == chunk_id
        assert result["publication_error"] is None
    finally:
        await client.aclose()
        async with sessions() as session, session.begin():
            await session.execute(delete(LowConfidenceQuestion).where(LowConfidenceQuestion.id.in_(ids)))
            if review_id: await session.execute(delete(ReviewQueue).where(ReviewQueue.id == review_id))
            if chunk_id: await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.id == chunk_id))


async def test_usage_actual_and_unknown_are_not_zero():
    collector = UsageCollector("test")
    run = uuid4()
    await collector.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="答案", usage_metadata={"input_tokens":13,"output_tokens":7,"total_tokens":20}))]]), run_id=run)
    await collector.on_llm_end(LLMResult(generations=[]), run_id=run)
    await collector.on_llm_error(ConnectionError(), run_id=uuid4())
    assert len(collector.rows)==2 and collector.rows[0]["input_tokens"]==13
    assert collector.rows[1]["input_tokens"] is None
    budget=CallBudget(1)
    await budget.on_chat_model_start({},[])
    with pytest.raises(ModelCallBudgetExceeded): await budget.on_chat_model_start({},[])


def test_selfhost_never_defaults_or_falls_back_to_cloud():
    with pytest.raises(ValueError):
        Observability(config(observability_enabled=True,langfuse_enabled=True, langfuse_base_url="https://cloud.langfuse.com", langfuse_public_key="x", langfuse_secret_key="y"), None)


async def test_disabled_observability_neither_exports_nor_persists():
    from unittest.mock import Mock, patch
    settings=config(observability_enabled=False,langfuse_enabled=True,
        langfuse_base_url="http://127.0.0.1:3000",langfuse_public_key="test-public",langfuse_secret_key="test-secret")
    sessions=Mock(side_effect=AssertionError("disabled observations must not open DB sessions"))
    with patch("langfuse.Langfuse",side_effect=AssertionError("disabled observations must not export traces")):
        observation=Observability(settings,sessions)
        with observation.request(SimpleNamespace(raw_question="离线测试",conversation_id=1,user_id="test")) as record:
            await observation.save(record,SimpleNamespace(conversation_id=1,user_message_id=1))
        await observation.close()
    sessions.assert_not_called()


async def test_observability_explicit_opt_in_persists_owned_record_and_costs(mysql_session):
    from app.observability import record_intent
    sessions=async_sessionmaker(mysql_session.bind,expire_on_commit=False)
    user=str(uuid4())
    intent="观测测试-"+uuid4().hex[:12]
    observation=Observability(config(observability_enabled=True,langfuse_enabled=False),sessions)
    async with sessions() as session,session.begin():
        conversation=Conversation(user_id=user)
        session.add(conversation);await session.flush()
        question=Message(conversation_id=conversation.id,role="user",content="仅测试观测，无模型调用")
        session.add(question);await session.flush()
    try:
        turn=SimpleNamespace(raw_question=question.content,conversation_id=conversation.id,user_message_id=question.id,user_id=user)
        with observation.request(turn) as record:
            record_intent(intent)
            await observation.save(record,turn)
        report=await cost_by_intent(sessions,config())
        row=next(item for item in report["items"] if item["intent"]==intent)
        assert row["requests"]==1 and row["model_calls"]==0
        assert row["unobserved_requests"]==1 and not row["usage_complete"]
        async with sessions() as session:
            assert await session.get(RequestObservation,record["id"]) is not None
    finally:
        async with sessions() as session,session.begin():
            await session.execute(delete(RequestObservation).where(RequestObservation.conversation_id==conversation.id))
            await session.execute(delete(Message).where(Message.conversation_id==conversation.id))
            await session.execute(delete(Conversation).where(Conversation.id==conversation.id))
        await observation.close()


def test_calibration_holdout_and_version_guard(tmp_path):
    rows=[{"case_id": f"case{i}", "sufficient":bool(i%2), "evidence":[{"answer":"事实", "score":.95 if i%2 else .05}]} for i in range(80)]
    from app.knowledge.confidence import corpus_fingerprint
    corpus = tmp_path/"knowledge";corpus.mkdir()
    artifact=calibrate(rows,reranker_model="reranker",corpus_hash=corpus_fingerprint(corpus),dataset_hash="labels")
    path=tmp_path/"calibration.json";path.write_text(json.dumps(artifact),encoding="utf-8")
    settings=config(evidence_calibration_path=str(path),reranker_model="reranker")
    gate=EvidenceGate(settings,corpus)
    assert gate.check(rows[1]["evidence"],legacy_confident=False).accepted
    assert not gate.check(rows[0]["evidence"],legacy_confident=True).accepted
    settings.reranker_model="changed"
    assert not EvidenceGate(settings,corpus).check([],legacy_confident=False).calibrated


def test_trend_partial_not_fake_zero_and_config_comparison():
    report={"run":{"cases_sha256":"dataset", "corpus_sha256":"corpus"}, "strategies":{"hybrid_rerank":{"recall_at_k":.8,"mrr":.7}},
        "cases":[{"case_id":"A1","strategy":"hybrid_rerank", "generation_status":"completed", "coverage_status":"completed", "faithfulness_status":"failed", "faithfulness":None}]}
    metrics=metrics_from_report(report)
    assert not metrics["complete"] and metrics["faithfulness"] is None and metrics["error_cases"]==1
    report["run"]["corpus_sha256"]="changed"
    assert metrics_from_report(report)["comparison_key"] != metrics["comparison_key"]
    before = metrics_from_report(report)["comparison_key"]
    report["run"]["retrieval_configuration"] = {"rerank_limit": 5}
    assert metrics_from_report(report)["comparison_key"] != before


def test_absent_refusal_is_complete_without_fake_faithfulness():
    report = {"cases": [
        {"case_id": "A1", "strategy": "hybrid_rerank", "expected_answerable": True,
         "generation_status": "completed", "coverage_status": "completed", "faithfulness_status": "completed", "faithfulness": .9},
        {"case_id": "D1", "strategy": "hybrid_rerank", "expected_answerable": False,
         "generation_status": "completed", "coverage_status": "not_run", "faithfulness_status": "not_run", "faithfulness": None},
    ]}
    metrics = metrics_from_report(report)
    assert metrics["complete"] and metrics["faithfulness_sample_count"] == 1 and metrics["faithfulness"] == .9


async def test_generation_insufficiency_marker_is_not_shown_and_enters_pool():
    service, model, search, events = await run_example("商品咨询", "这个支持手机吗", ["当前材料不足。[证据不足:没有手机兼容性说明]"])
    assert events[-1].name == "done"
    assert "证据不足:" not in events[-1].data["content"]
    assert "self_check" in service.low_confidence


@pytest.mark.parametrize("source", ["retrieval_low_conf", "self_check"])
async def test_workflow_problem_entrances_persist_exact_snapshot_via_http(mysql_session, monkeypatch, source):
    from dataclasses import replace
    from random import Random
    from langchain_core.messages import SystemMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from app.api import create_api_router
    from app.context import PreparedContext
    from app.services.tool_chat import PreparedTurn
    from app.tools.business import ToolRuntime
    from app.workflow.graph import WorkflowChatService
    from tests.workflow_fakes import LabelledModel, LabelledSearch
    from tests.order_fakes import FakeOrderData, TestToolCatalog

    sessions = async_sessionmaker(mysql_session.bind, expire_on_commit=False)
    user = str(uuid4())
    async with sessions() as session, session.begin():
        conversation = Conversation(user_id=user)
        session.add(conversation); await session.flush()
        question = Message(conversation_id=conversation.id, role="user", content="测试知识是否充分")
        session.add(question); await session.flush()
    model = LabelledModel("商品咨询", ["无法确认。[证据不足:本次材料缺少说明]"])
    search = LabelledSearch(True)
    if source == "retrieval_low_conf":
        original_retrieve = search.retrieve
        async def weak_retrieve(*args, **kwargs):
            result = await original_retrieve(*args, **kwargs)
            return replace(result, matches=(replace(result.matches[0], score=.01),), confident=False)
        monkeypatch.setattr(search, "retrieve", weak_retrieve)
    settings = config(evidence_calibration_path=".scratch/observability-flywheel/nonexistent-calibration.json")
    service = WorkflowChatService(settings=settings, model=model, session_factory=sessions,
        knowledge_search=search, checkpointer=InMemorySaver(), order_data=FakeOrderData(), catalog=TestToolCatalog())
    prepared = PreparedTurn(conversation_id=conversation.id, user_id=user, user_message_id=question.id,
        raw_question=question.content, prepared_context=PreparedContext((SystemMessage(content="安全客服"), HumanMessage(content=question.content)), 20, 0),
        tool_runtime=ToolRuntime(conversation.id, sessions, Random(0), search, user_id=user, order_data=service._orders))
    async def prepare(request): return prepared
    monkeypatch.setattr(service, "prepare_turn", prepare)
    app = FastAPI()
    app.include_router(create_api_router(settings, model, sessions, service, None))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/chat/stream", json={"user_id":user,"message":question.content})
            assert response.status_code == 200 and "event: done" in response.text
        async with sessions() as session:
            rows = list(await session.scalars(select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == conversation.id)))
            assert len(rows) == 1 and rows[0].source == source and rows[0].raw_question == question.content
            assert rows[0].retrieved_chunks[0]["answer"] == "签收后七天内可申请退货。"
            answers = list(await session.scalars(select(Message).where(Message.conversation_id == conversation.id, Message.role == "assistant")))
            assert answers[-1].reply_to_message_id == question.id
            assert answers[-1].retrieved_chunks == rows[0].retrieved_chunks
    finally:
        await service.close()
        async with sessions() as session, session.begin():
            await session.execute(delete(RequestObservation).where(RequestObservation.conversation_id == conversation.id))
            await session.execute(delete(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == conversation.id))
            await session.execute(delete(Message).where(Message.conversation_id == conversation.id))
            await session.execute(delete(Conversation).where(Conversation.id == conversation.id))


def test_no_hidden_provider_retry_is_set_before_client_construction():
    from app.llm import create_chat_model
    model = create_chat_model(config(), max_retries=0)
    assert model.root_client.max_retries == model.root_async_client.max_retries == 0
    with pytest.raises(ValueError):
        Flywheel(config(), None, create_chat_model(config()), None)
