"""Dashboard seams with isolated SQL and frozen synthetic labels; no paid calls."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database.models import RequestObservation
from app.evaluation.dashboard import calibration_report, trend_report
from app.evaluation.pipeline import cost_by_intent
from app.knowledge.confidence import scoring_fingerprint


async def test_cost_window_partial_usage_and_no_fake_share():
    engine = create_async_engine("sqlite+aiosqlite://")
    sessions = async_sessionmaker(engine)
    now = datetime(2026, 10, 13, 10, tzinfo=timezone.utc)
    settings = SimpleNamespace(input_price_per_million=1, output_price_per_million=2, langfuse_enabled=False)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE conversations (id BIGINT PRIMARY KEY)"))
            await connection.execute(text("CREATE TABLE messages (id BIGINT PRIMARY KEY, conversation_id BIGINT)"))
            await connection.execute(text("""CREATE TABLE request_observations (
                id VARCHAR(36) PRIMARY KEY, conversation_id BIGINT, user_message_id BIGINT,
                intent VARCHAR(32), trace_id VARCHAR(64), model_usage JSON, duration_ms INTEGER,
                status VARCHAR(32), created_at DATETIME)"""))
        async with sessions() as session, session.begin():
            for key, age, intent, usage in (
                ("recent", 1, "knowledge", [{"status": "completed", "input_tokens": 100, "output_tokens": 50}]),
                ("unknown", 2, "logistics", [{"status": "failed", "input_tokens": 20, "output_tokens": None}]),
                ("old", 8, "old", [{"status": "completed", "input_tokens": 9000, "output_tokens": 9000}]),
                ("future", -1, "future", [{"status": "completed", "input_tokens": 9000, "output_tokens": 9000}]),
            ):
                session.add(RequestObservation(id=key, intent=intent, model_usage=usage, status="completed",
                    duration_ms=1, created_at=(now-timedelta(days=age)).replace(tzinfo=None)))
        result = await cost_by_intent(sessions, settings, now=now)
        assert result["summary"] == {"requests": 2, "total_tokens": 170, "intent_count": 2,
            "highest_intent": "knowledge", "unknown_usage_calls": 1, "failed_calls": 1, "model_calls": 2,
            "unobserved_requests": 0, "excluded_unlinked_empty_requests": 0}
        assert result["items"][0]["token_share"] == 150/170
        assert result["items"][0]["estimated_cost"] == .0002
        assert result["items"][1]["estimated_cost"] is None
        assert not result["items"][1]["usage_complete"]
        empty = await cost_by_intent(sessions, settings, days=1, now=now-timedelta(days=20))
        assert empty["summary"]["highest_intent"] is None and empty["items"] == []
    finally:
        await engine.dispose()


async def test_cost_quarantines_empty_orphans_but_preserves_historical_cost_and_unknown_requests():
    engine=create_async_engine("sqlite+aiosqlite://")
    sessions=async_sessionmaker(engine)
    settings=SimpleNamespace(input_price_per_million=1,output_price_per_million=2,langfuse_enabled=False)
    now=datetime(2026,10,13,10,tzinfo=timezone.utc)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE conversations (id BIGINT PRIMARY KEY)"))
            await connection.execute(text("CREATE TABLE messages (id BIGINT PRIMARY KEY, conversation_id BIGINT)"))
            await connection.execute(text("""CREATE TABLE request_observations (
                id VARCHAR(36) PRIMARY KEY, conversation_id BIGINT, user_message_id BIGINT,
                intent VARCHAR(32), trace_id VARCHAR(64), model_usage JSON, duration_ms INTEGER,
                status VARCHAR(32), created_at DATETIME)"""))
            await connection.execute(text("INSERT INTO conversations VALUES (1)"))
            await connection.execute(text("INSERT INTO messages VALUES (1,1)"))
        async with sessions() as session,session.begin():
            for key,intent,linked,usage in (
                ("test-residue","售后",False,[]),
                ("real-unknown","售后",True,[]),
                ("real-zero","零用量",True,[{"status":"success","input_tokens":0,"output_tokens":0}]),
                ("historical-paid","历史",False,[{"status":"success","input_tokens":4,"output_tokens":6}]),
                ("json-null","异常空值",False,None),
                ("empty-object","异常空值",False,{}),
            ):
                session.add(RequestObservation(id=key,intent=intent,conversation_id=1 if linked else 999,
                    user_message_id=1 if linked else 999,model_usage=usage,status="success",duration_ms=1,
                    created_at=now.replace(tzinfo=None)))
            await session.execute(text("""INSERT INTO request_observations
                (id,conversation_id,user_message_id,intent,model_usage,duration_ms,status,created_at)
                VALUES ('sql-null',999,999,'异常空值',NULL,1,'success',:created)"""),
                {"created":now.replace(tzinfo=None)})
        result=await cost_by_intent(sessions,settings,now=now)
        items={r["intent"]:r for r in result["items"]}
        assert result["summary"]["excluded_unlinked_empty_requests"]==1
        assert result["summary"]["requests"]==6
        assert result["summary"]["unobserved_requests"]==4
        assert items["售后"]["requests"]==1 and items["售后"]["unobserved_requests"]==1
        assert not items["售后"]["usage_complete"] and items["售后"]["estimated_cost"] is None
        assert items["售后"]["token_share"] is None
        assert items["零用量"]["usage_complete"] and items["零用量"]["estimated_cost"]==0
        assert items["历史"]["total_tokens"]==10
        assert items["异常空值"]["requests"]==3 and not items["异常空值"]["usage_complete"]
        async with sessions() as session:
            assert await session.scalar(text("SELECT COUNT(*) FROM request_observations"))==7
    finally:
        await engine.dispose()


def test_trend_skips_partial_and_different_configuration_and_missing_scores():
    def run(id, key, complete=True, mrr=.8, faith=.9):
        return {"id": id, "dataset_size": 5, "created_at": "2026-10-13", "triggered_by": "手动",
            "metrics": {"comparison_key": key, "complete": complete, "mrr": mrr, "faithfulness": faith,
                "recall_at_k": .8, "strategies": {"hybrid_rerank": {"refusal_accuracy": .9}}}}
    report = trend_report([run(1,"a"), run(2,"b",mrr=.1), run(3,"a",False,mrr=.1), run(4,"a",mrr=.7,faith=None)])
    a = next(g for g in report["groups"] if g["key"]=="a")
    assert [r["id"] for r in a["rows"]] == [4,3,1]
    assert a["latest"]["previous_id"] == 1
    assert a["latest"]["deltas"]["mrr"] == pytest.approx(-.1)
    assert a["latest"]["deltas"]["faithfulness"] is None
    assert a["declining"] == ["mrr"]
    assert a["rows"][1]["previous_id"] is None
    assert all(v is None for v in a["rows"][1]["deltas"].values())
    assert trend_report([]) == {"total_runs":0,"groups":[]}


@pytest.fixture
def calibration_files(tmp_path):
    knowledge = tmp_path/"knowledge"
    knowledge.mkdir()
    (knowledge/"policy.md").write_text("policy",encoding="utf-8")
    path=tmp_path/"calibration.json"
    settings=SimpleNamespace(evidence_calibration_path=str(path),reranker_model="test-reranker")
    artifact={"formula_version":2,"purpose":"relevance_prefilter","threshold":.3,"validation":{"passed":True},
        "reranker_model":"test-reranker", "scoring_fingerprint":scoring_fingerprint(knowledge, settings),"dataset_sha256":"test-dataset",
        "reviewer":"Synthetic test fixture, not business approval"}
    directory=tmp_path/"reports/confidence-calibration"
    directory.mkdir(parents=True)
    for name,size in (("training",40),("validation",20)):
        rows=[{"case_id":f"{name}-{i}","sufficient":i%2==0,
            "evidence":[{"score":.8 if i%2==0 else .02,"answer":"test"}]} for i in range(size)]
        data={**artifact,"rows":rows,"label_scope":"relevance_prefilter"}
        raw=json.dumps(data).encode()
        (directory/f"{name}.json").write_bytes(raw)
        artifact[f"{name}_labels_sha256"]=hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(artifact),encoding="utf-8")
    return tmp_path,path,settings


def test_offline_scan_uses_bound_snapshots_and_never_publishes(calibration_files):
    root,path,settings=calibration_files
    before=path.read_bytes()
    report=calibration_report(settings,root)
    assert report["available"], report["message"]
    # The historical v2 report remains usable even with the committed v1 gate;
    # only the running gate determines whether its threshold is actually active.
    assert report["active_threshold"] == (.3 if report["active"] else None)
    assert report["training_size"] == 40 and report["validation_size"] == 20
    assert report["validation_at_recommended"]["irrelevant_pass_rate"] == 0
    assert report["validation_at_recommended"]["relevant_pass_rate"] == 1
    assert path.read_bytes()==before
    assert len(report["scan"])>=101 and len(report["distributions"])==4
    assert report["recommended_threshold"] in [r["threshold"] for r in report["scan"]]


def test_historical_v2_report_works_with_v1_serving_gate(calibration_files, monkeypatch):
    root,_,settings=calibration_files
    class LegacyGate:
        def __init__(self,*args): pass
        def check(self,*args,**kwargs):
            return SimpleNamespace(calibrated=False,threshold=None,reason="校准版本与当前门控不一致")
    monkeypatch.setattr("app.evaluation.dashboard.EvidenceGate",LegacyGate)
    report=calibration_report(settings,root)
    assert report["available"] and not report["active"]
    assert report["active_threshold"] is None and report["recommended_threshold"] is not None
    assert report["published_threshold"]==.3


def test_calibration_missing_snapshot_and_changed_corpus_do_not_claim_active(calibration_files):
    root,path,settings=calibration_files
    (root/"knowledge/policy.md").write_text("changed",encoding="utf-8")
    report=calibration_report(settings,root)
    assert report["available"] and not report["active"]
    assert report["active_threshold"] is None
    (root/"reports/confidence-calibration/training.json").unlink()
    report=calibration_report(settings,root)
    assert not report["available"] and "快照" in report["message"]
    path.unlink()
    assert not calibration_report(settings,root)["available"]
