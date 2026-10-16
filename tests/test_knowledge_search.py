"""Hybrid retrieval behavior at the Milvus-to-MySQL authority seam."""

from types import SimpleNamespace

from app.database.models import KnowledgeChunk
from app.knowledge.query_understanding import QueryPlan, QueryUnderstandingService
from app.knowledge.search import (
    HybridKnowledgeSearch,
    RetrievalStrategy,
    build_filter_expression,
)
from tests.fakes import FakeStructuredModel


class FixedEmbedder:
    def embed(self, texts):
        return [[0.0] * 1024 for _ in texts]


class FixedPlanner:
    async def understand(self, query: str) -> QueryPlan:
        return QueryPlan(
            normalized_query="配送费用如何计算",
            synonyms=["邮费", "运费"],
        )


class RankedStore:
    def __init__(self, existing_id: str) -> None:
        self.existing_id = existing_id
        self.query = ""
        self.filter_expression = ""

    def search_hybrid(self, vector, query, **kwargs):
        self.query = query
        self.filter_expression = kwargs["filter_expression"]
        return [("999999999", 0.99), (self.existing_id, 0.88)]


class FixedReranker:
    def __init__(self, score: float) -> None:
        self.score = score

    def rerank(self, query, chunks, *, limit):
        return [SimpleNamespace(chunk=row, score=self.score) for row in chunks[:limit]]


class ReuseSession:
    def __init__(self, session) -> None:
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class ReuseSessionFactory:
    def __init__(self, session) -> None:
        self.session = session

    def __call__(self):
        return ReuseSession(self.session)


def test_filter_expression_is_built_only_from_typed_values() -> None:
    expression = build_filter_expression(["手机", 'a"b'], ["faq"])

    assert expression == 'category in ["手机","a\\\"b"] and content_type in ["faq"]'


async def test_hybrid_search_expands_query_filters_metadata_and_reranks(mysql_session) -> None:
    row = KnowledgeChunk(
        category="配送服务",
        questions="配送费用如何计算？",
        answer="结算页会展示实际运费。",
        content_type="faq",
        vectorize_status="done",
    )
    mysql_session.add(row)
    await mysql_session.flush()
    row.vector_id = str(row.id)
    await mysql_session.flush()
    store = RankedStore(str(row.id))
    search = HybridKnowledgeSearch(
        ReuseSessionFactory(mysql_session),
        FixedEmbedder(),
        store,
        FixedReranker(0.87),
        FixedPlanner(),
        confidence_threshold=0.5,
    )

    result = await search.retrieve("邮费咋算", strategy=RetrievalStrategy.HYBRID_RERANK)

    assert [match.chunk.id for match in result.matches] == [row.id]
    assert result.matches[0].score == 0.87
    assert result.confident is True
    assert store.query == "配送费用如何计算 邮费 运费"


async def test_rerank_score_below_threshold_is_not_confident(mysql_session) -> None:
    row = KnowledgeChunk(
        category="其他",
        questions="无关问题",
        answer="无关答案",
        content_type="faq",
        vectorize_status="done",
    )
    mysql_session.add(row)
    await mysql_session.flush()
    row.vector_id = str(row.id)
    await mysql_session.flush()
    search = HybridKnowledgeSearch(
        ReuseSessionFactory(mysql_session),
        FixedEmbedder(),
        RankedStore(str(row.id)),
        FixedReranker(0.1),
        FixedPlanner(),
        confidence_threshold=0.5,
    )

    result = await search.retrieve("不存在的问题")

    assert result.matches
    assert result.confident is False


async def test_guessed_category_and_type_do_not_narrow_recall(mysql_session) -> None:
    """A planner cannot know which document type holds a fact, so its guess
    must not hide policies and manuals behind a FAQ-only Milvus filter."""

    row = KnowledgeChunk(
        category="积分与优惠券细则",
        questions="签到获取积分",
        answer="每日签到可获得 5 积分。",
        content_type="policy",
        vectorize_status="done",
    )
    mysql_session.add(row)
    await mysql_session.flush()
    row.vector_id = str(row.id)
    await mysql_session.flush()
    store = RankedStore(str(row.id))
    planner = QueryUnderstandingService(
        FakeStructuredModel(
            result={
                "normalized_query": "每天签到能获得多少积分",
                "categories": ["会员等级与积分规则"],
                "content_types": ["faq"],
            }
        ),
        method="json_mode",
    )
    search = HybridKnowledgeSearch(
        ReuseSessionFactory(mysql_session),
        FixedEmbedder(),
        store,
        FixedReranker(0.87),
        planner,
        confidence_threshold=0.5,
    )

    result = await search.retrieve("每天签到能获得多少积分")

    assert store.filter_expression == ""
    assert [match.chunk.id for match in result.matches] == [row.id]
