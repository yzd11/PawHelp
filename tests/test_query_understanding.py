"""Query normalization must be structured, bounded, and safely degradable."""

from app.knowledge.query_understanding import QueryUnderstandingService
from app.prompts import (
    render_evidence_sufficiency_messages,
    render_query_understanding_messages,
)
from tests.fakes import FakeStructuredModel


async def test_query_understanding_normalizes_and_deduplicates_expansions() -> None:
    model = FakeStructuredModel(
        result={
            "normalized_query": "X100 手机支持什么充电器？",
            "synonyms": ["充电头", " 充电头 ", "电源适配器"],
            "categories": ["手机", "手机"],
            "content_types": ["faq"],
        }
    )
    service = QueryUnderstandingService(model, method="json_mode")

    plan = await service.understand("X100 用啥充电头")

    assert plan.synonyms == ["充电头", "电源适配器"]
    assert model.structured_kwargs == {"method": "json_mode"}


async def test_query_understanding_falls_back_to_raw_query() -> None:
    service = QueryUnderstandingService(
        FakeStructuredModel(error=RuntimeError("provider unavailable")),
        method="json_mode",
    )

    plan = await service.understand("  邮费咋算  ")

    assert plan.normalized_query == "邮费咋算"
    assert plan.synonyms == []


def test_json_mode_structured_prompts_explicitly_request_json() -> None:
    query_text = " ".join(
        str(message.content) for message in render_query_understanding_messages("邮费")
    )
    evidence_text = " ".join(
        str(message.content) for message in render_evidence_sufficiency_messages("{}")
    )

    assert "JSON" in query_text
    assert "JSON" in evidence_text
