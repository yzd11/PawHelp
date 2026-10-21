"""Jev evidence judge: modes, fallbacks and calibration binding, with a fake System One client."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.knowledge.confidence import ConfidenceDecision
from app.knowledge.evidence import EvidenceDecision, EvidenceSufficiencyChecker
from app.knowledge.jev_evidence import (
    INSTRUCTIONS_SHA256,
    JEV_QUESTION,
    JevEvidenceJudge,
    JevRoutedChecker,
    build_evidence_checker,
)
from app.knowledge.pipeline import KnowledgeEvidencePipeline
from app.knowledge.search import KnowledgeMatch, RetrievalResult, RetrievalStrategy
from tests.fakes import fake_settings

EVIDENCE = [{"n": 1, "answer": "订单实付满 99 元包邮，未满收取 10 元运费。"}]


class FakeJev:
    def __init__(self, p_yes=0.5, *, model="jev-1.14.0", error=None, started=None, release=None):
        self.p_yes, self.model, self.error, self.calls = p_yes, model, error, []
        self.started, self.release = started, release

    async def system_one(self, state, questions, *, model=None, timeout=None):
        self.calls.append((state, questions, model))
        if self.started:
            self.started.set()
        if self.release:
            await self.release.wait()
        if self.error:
            raise self.error
        return SimpleNamespace(model=self.model, nouls={JEV_QUESTION: SimpleNamespace(noul=self.p_yes)},
                               usage=SimpleNamespace(input_tokens=321))


class FakeLlm:
    def __init__(self, useful=True, *, started=None, release=None):
        self.useful, self.calls, self.started, self.release = useful, [], started, release

    async def check(self, question, evidence):
        self.calls.append(question)
        if self.started:
            self.started.set()
        if self.release:
            await self.release.wait()
        return EvidenceDecision(useful=self.useful, reason="大模型判定")


def calibration(tmp_path, **overrides):
    data = {"purpose": "jev_evidence_sufficiency", "model": "jev-1.14.0", "instructions_sha256": INSTRUCTIONS_SHA256,
            "lower": 0.2, "upper": 0.8, "validation": {"passed": True}, **overrides}
    path = tmp_path / "jev.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def routed(tmp_path, jev, llm, *, mode="jev", **overrides):
    return JevRoutedChecker(llm, JevEvidenceJudge(jev, model="jev-1.14.0", timeout=1), mode=mode,
                            calibration_path=calibration(tmp_path, **overrides))


@pytest.mark.parametrize("p_yes,useful,llm_called", [(0.9, True, False), (0.1, False, False), (0.5, True, True)])
async def test_jev_mode_decides_outside_the_band_and_defers_inside(tmp_path, p_yes, useful, llm_called):
    jev, llm = FakeJev(p_yes), FakeLlm(useful=True)

    decision = await routed(tmp_path, jev, llm).check("满多少包邮", EVIDENCE)

    assert decision.useful is useful
    assert bool(llm.calls) is llm_called
    assert decision.judge == ("llm" if llm_called else "jev")
    assert decision.jev["p_yes"] == p_yes and decision.jev["input_tokens"] == 321
    if p_yes == 0.1:
        assert decision.reason == "Jev 判定证据不足以回答（P=0.10）"
    state, questions, model = jev.calls[0]
    assert json.loads(state) == {"question": "满多少包邮", "evidence": EVIDENCE}
    assert questions[JEV_QUESTION]["type"] == "noul" and model == "jev-1.14.0"


@pytest.mark.parametrize("error", [RuntimeError("boom"), asyncio.TimeoutError()])
async def test_jev_failure_hands_the_decision_to_the_llm(tmp_path, error):
    llm = FakeLlm(useful=False)

    decision = await routed(tmp_path, FakeJev(error=error), llm).check("满多少包邮", EVIDENCE)

    assert llm.calls and decision.judge == "llm" and decision.useful is False
    assert decision.jev["error"] == type(error).__name__


@pytest.mark.parametrize("overrides,served_model", [
    ({"model": "jev-1.13.0"}, "jev-1.14.0"),
    ({}, "jev-1.15.0"),
    ({"instructions_sha256": "stale"}, "jev-1.14.0"),
    ({"validation": {"passed": False}}, "jev-1.14.0"),
])
async def test_mismatched_calibration_defers_to_the_llm(tmp_path, overrides, served_model):
    llm = FakeLlm(useful=True)

    decision = await routed(tmp_path, FakeJev(0.01, model=served_model), llm, **overrides).check("q", EVIDENCE)

    assert llm.calls and decision.judge == "llm" and decision.useful is True
    assert decision.jev["calibration"]


async def test_shadow_mode_keeps_the_llm_decision_and_runs_jev_concurrently(tmp_path):
    jev_started, llm_started = asyncio.Event(), asyncio.Event()
    # Each side waits for the other to start: sequential calls would deadlock.
    jev = FakeJev(0.01, started=jev_started, release=llm_started)
    llm = FakeLlm(useful=True, started=llm_started, release=jev_started)

    decision = await asyncio.wait_for(routed(tmp_path, jev, llm, mode="shadow").check("q", EVIDENCE), 2)

    assert decision.useful is True and decision.judge == "llm" and decision.reason == "大模型判定"
    assert decision.jev["p_yes"] == 0.01


async def test_shadow_mode_does_not_wait_for_a_slower_jev(tmp_path):
    jev = FakeJev(0.9, release=asyncio.Event())  # never released: Jev hangs

    decision = await asyncio.wait_for(routed(tmp_path, jev, FakeLlm(useful=False), mode="shadow").check("q", EVIDENCE), 1)

    assert decision.useful is False and decision.jev == {"error": "SlowerThanLlm"}


def test_llm_mode_needs_no_key_and_builds_the_plain_checker():
    checker = build_evidence_checker(fake_settings(), model=object())

    assert type(checker) is EvidenceSufficiencyChecker


@pytest.mark.parametrize("mode", ["jev", "shadow"])
def test_jev_modes_refuse_to_start_without_a_key(mode):
    with pytest.raises(ValidationError, match="TYPESAFE_API_KEY"):
        fake_settings(evidence_judge=mode)


async def test_pipeline_serves_jev_verdict_and_logs_it(tmp_path, caplog):
    chunk = SimpleNamespace(id=7, category="policy", section_path="运费", questions="包邮", answer=EVIDENCE[0]["answer"])
    result = RetrievalResult((KnowledgeMatch(chunk, 0.9),), None, RetrievalStrategy.HYBRID_RERANK, True)
    search = SimpleNamespace(retrieve=lambda q, **_: asyncio.sleep(0, result))
    gate = SimpleNamespace(check=lambda evidence, **_: ConfidenceDecision(True, True, {}, 0.17, "已校准证据评分"))
    splitter = SimpleNamespace(split=lambda q: asyncio.sleep(0, [q]))
    pipeline = KnowledgeEvidencePipeline(search, gate, routed(tmp_path, FakeJev(0.05), FakeLlm()), splitter, top_k=5)

    with caplog.at_level("INFO", logger="app.knowledge.jev_evidence"):
        retrieval = await pipeline.retrieve("满多少包邮")

    assert retrieval.result.confident is False
    assert retrieval.confidence["reason"] == "Jev 判定证据不足以回答（P=0.05）"
    assert "judge=jev" in caplog.text and '"p_yes": 0.05' in caplog.text


def test_offline_eval_picks_bounds_on_train_and_compares_with_the_llm_on_validation():
    from scripts.run_jev_evidence_evals import dataset_stamp, load_cases, summarize

    def case(case_id, answerable, served, llm):
        return {"case_id": case_id, "query": "q", "strategy": "hybrid_rerank", "generation_status": "completed",
                "citations": EVIDENCE, "expected_answerable": answerable,
                "evidence_confidence": {"relevance_accepted": served, "complete": llm}}
    report = {"run": {"cases_sha256": "x", "generated_at": "t"}, "cases": [
        case("A1", True, True, True), case("A2", True, True, True), case("D1", False, True, False),
        case("D2", False, False, None), case("A5", True, True, False), case("D5", False, True, False),
        case("A10", True, True, True), {"case_id": "A3", "strategy": "dense"},
    ]}
    cases = load_cases(report)
    for c, p in zip(cases, [0.9, 0.6, 0.3, 0.7, 0.95, 0.02, 0.65]):
        c.update(p_yes=p, model="jev-1.14.0", latency_ms=100, input_tokens=10)

    summary = summarize(cases, model="jev-1.14.0", dataset=dataset_stamp(cases, report))

    assert [c["case_id"] for c in cases if c["validation"]] == ["A5", "D5", "A10"]
    # Train: negatives D1=0.3, D2=0.7 (prefiltered, still a negative); lowest positive 0.6.
    assert summary["upper"] == pytest.approx(0.7, abs=1e-5) and summary["lower"] == pytest.approx(0.6, abs=1e-5)
    gate = summary["validation"]["served_gate"]
    # A5: Jev accepts what the LLM refused; D5: Jev refuses; A10 falls in the band, so the LLM decides.
    assert (gate["llm_false_refusals"], gate["policy_false_refusals"]) == (1, 0)
    assert (gate["llm_false_accepts"], gate["policy_false_accepts"]) == (0, 0)
    assert gate["jev_coverage"] == pytest.approx(2 / 3)
    assert summary["validation"]["passed"] is True


async def test_intent_eval_maps_jev_labels_and_picks_the_review_threshold_on_train():
    from scripts.run_jev_intent_evals import JEV_QUESTION, jev_choice, summarize

    class FakeChoiceJev:
        async def system_one(self, state, questions, *, model=None, timeout=None):
            assert json.loads(state) == {"customer_message": "转人工"} and questions[JEV_QUESTION]["type"] == "choice"
            answer = SimpleNamespace(choice="human_agent", probabilities={"human_agent": 0.9, "complaint": 0.1})
            return SimpleNamespace(model="jev-1.14.0", choices={JEV_QUESTION: answer}, usage=SimpleNamespace(input_tokens=50))

    jev = await jev_choice(FakeChoiceJev(), "转人工", model="jev-1.14.0", timeout=1)
    assert jev["top"] == "人工" and jev["probabilities"] == {"人工": 0.9, "投诉": 0.1}

    def case(case_id, split, intents, llm, jev_p):
        runs = [{"intent": i, "confidence": c, "mismatches": [] if i in intents else ["intent"]} for i, c in llm]
        probabilities = {i: jev_p.get(i, 0.0) for i in ("物流", "售后", "其他")}
        return {"id": case_id, "split": split, "intents": intents, "llm": runs,
                "jev": {"top": max(probabilities, key=probabilities.get), "probabilities": probabilities, "latency_ms": 5}}
    # The LLM is equally sure when right and wrong; Jev's probability for its pick tells them apart.
    cases = [case("a", "train", ["物流"], [("物流", 0.9)], {"物流": 0.8}),
             case("b", "train", ["售后"], [("物流", 0.9)], {"物流": 0.2, "售后": 0.7}),
             case("c", "train", ["其他"], [("其他", 0.3)], {"其他": 0.9}),
             case("d", "validation", ["物流"], [("物流", 0.9)], {"物流": 0.85}),
             case("e", "validation", ["售后"], [("物流", 0.95)], {"物流": 0.1, "售后": 0.8})]

    summary = summarize(cases, model="llm", jev_model="jev-1.14.0", runs_per_case=1)

    rules = summary["validation"]["rules"]
    assert rules["current"] == {"score": "confidence", "threshold": 0.55, "served_wrong": 1, "clarified_clear": 0}
    assert rules["jev_tuned"]["threshold"] == 0.8  # flags b, keeps a
    assert (rules["jev_tuned"]["served_wrong"], rules["jev_tuned"]["clarified_clear"]) == (0, 1)
    assert summary["validation"]["auc"]["jev_p"]["auc"] == 1.0 and summary["validation"]["auc"]["confidence"]["auc"] == 0.0
    # Catching the wrong answer cost a clarification on a clear question, so the rule is not strictly better.
    assert summary["passed"] is False
