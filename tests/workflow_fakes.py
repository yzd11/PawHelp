"""Network-free model, search and service doubles for workflow tests.

Each double scripts only the customer-understanding decisions a test cares
about; knowledge-pipeline helpers (question split, evidence sufficiency) get
neutral defaults so tests do not break when those stages are added or tuned.
"""

from random import Random

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings
from app.context import PreparedContext
from app.database.models import KnowledgeChunk
from app.knowledge.query_understanding import QueryPlan
from app.knowledge.search import KnowledgeMatch, RetrievalResult, RetrievalStrategy
from app.services.tool_chat import PreparedTurn
from app.tools.business import ToolRuntime
from app.workflow.graph import WorkflowChatService
from tests.order_fakes import FakeOrderData, TestToolCatalog


def tool_call(name: str, args: str, call_id: str) -> AIMessageChunk:
    return AIMessageChunk(content="", tool_call_chunks=[
        {"name": name, "args": args, "id": call_id, "index": 0, "type": "tool_call_chunk"}
    ])


def pipeline_default(schema_name: str, messages):
    """Neutral answers for knowledge-pipeline schemas; None for everything else."""

    if schema_name == "KnowledgeSubquestions":
        return {"queries": [str(messages[-1].content)]}
    if schema_name == "EvidenceDecision":
        return {"useful": True, "reason": "证据可回答问题"}
    return None


class LabelledModel:
    """Classifies every turn as one fixed intent and streams scripted agent outputs."""

    def __init__(self, intent: str, outputs=(), decision=None) -> None:
        self.intent = intent
        self.decision = decision or {}
        self.outputs = list(outputs)
        self.classifier_calls = 0
        self.resolution_calls = 0
        self.expansion_calls = 0
        self.agent_calls = 0
        self.agent_histories = []
        self.tool_names: list[str] = []

    def with_structured_output(self, schema, *, method):
        parent = self

        class Runnable:
            async def ainvoke(self, messages):
                default = pipeline_default(schema.__name__, messages)
                if default is not None:
                    return default
                if schema.__name__ == "ResolvedQuestion":
                    parent.resolution_calls += 1
                    return {"question": messages[-1].content}
                if schema.__name__ == "ExpandedQueries":
                    parent.expansion_calls += 1
                    return {"queries": [messages[-1].content]}
                parent.classifier_calls += 1
                return {"intent": parent.intent, "confidence": 0.95, **parent.decision}

        return Runnable()

    def bind_tools(self, tools, **kwargs):
        self.tool_names = [tool.name for tool in tools]
        return self

    async def astream(self, messages):
        self.agent_calls += 1
        self.agent_histories.append(list(messages))
        assert self.outputs, "unexpected agent model call"
        item = self.outputs.pop(0)
        yield item if isinstance(item, AIMessageChunk) else AIMessageChunk(content=item)


class ScriptedUnderstandingModel(LabelledModel):
    """Returns a scripted sequence of intent decisions and an optional rewrite."""

    def __init__(self, decisions, outputs=(), resolution=None):
        super().__init__("其他", outputs)
        self.decisions = list(decisions)
        self.resolution = resolution
        self.inputs = []

    def with_structured_output(self, schema, *, method):
        if schema.__name__ == "ExpandedQueries":
            return super().with_structured_output(schema, method=method)
        parent = self

        class Runnable:
            async def ainvoke(self, messages):
                default = pipeline_default(schema.__name__, messages)
                if default is not None:
                    return default
                parent.inputs.append((schema.__name__, messages))
                if schema.__name__ == "ResolvedQuestion":
                    return parent.resolution or {"question": messages[-1].content}
                parent.classifier_calls += 1
                assert parent.decisions, "classifier exceeded its bounded call count"
                return parent.decisions.pop(0)

        return Runnable()


class KeywordModel:
    """Routes by keywords in the question, like a small but plausible classifier."""

    def __init__(self):
        self.structured: list[tuple[str, str]] = []
        self.agent_calls = 0

    def with_structured_output(self, schema, *, method):
        parent = self

        class Runnable:
            async def ainvoke(self, messages):
                default = pipeline_default(schema.__name__, messages)
                if default is not None:
                    return default
                question = str(messages[-1].content)
                parent.structured.append((schema.__name__, question))
                if schema.__name__ == "ResolvedQuestion":
                    if question == "它能退吗":
                        return {"question": "订单1001的猫爬架能退吗"}
                    if question == "能退吗":
                        return {"question": "订单9999能退吗"}
                    return {"question": question}
                if schema.__name__ == "ExpandedQueries":
                    return {"queries": ["猫爬架退货政策", "退货办理条件", "猫爬架退货政策"]}
                if question == "怪问题":
                    return {"intent": "物流", "confidence": "high"}
                if "退" in question:
                    return {"intent": "退款退货", "confidence": 0.95}
                if "售后" in question:
                    return {"intent": "售后", "confidence": 0.94}
                if "物流" in question:
                    return {"intent": "物流", "confidence": 0.96}
                return {"intent": "其他", "confidence": 0.2}

        return Runnable()

    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages):
        self.agent_calls += 1
        yield AIMessageChunk(content="可以查看本次政策依据，具体能否退还要核对订单条件。[1]")


class LabelledSearch:
    """Returns one return-policy chunk, or nothing when not confident."""

    def __init__(self, confident: bool = True) -> None:
        self.confident = confident
        self.calls: list[dict[str, object]] = []
        self.queries: list[tuple[str, dict[str, object]]] = []

    async def retrieve(self, query, **kwargs):
        self.calls.append(kwargs)
        self.queries.append((query, kwargs))
        chunk = KnowledgeChunk(
            id=41, category="退款退货", questions="七天无理由退货如何申请",
            answer="签收后七天内可申请退货。", section_path="退货政策 / 无理由退货",
            content_type="policy", vectorize_status="done",
        )
        return RetrievalResult(
            (KnowledgeMatch(chunk, 0.9),) if self.confident else (),
            QueryPlan(normalized_query=query), RetrievalStrategy.HYBRID_RERANK,
            self.confident,
        )


class InMemoryWorkflow(WorkflowChatService):
    """The real workflow graph with persistence replaced by in-memory lists."""

    def __init__(self, model, search):
        settings = Settings(
            chat_base_url="http://localhost:11434/v1", openai_model="fake",
            openai_api_key="fake", database_url="mysql+asyncmy://user:pass@localhost/db",
        )
        super().__init__(
            settings=settings, model=model, session_factory=lambda: None,
            knowledge_search=search, checkpointer=InMemorySaver(),
            order_data=FakeOrderData(),
            catalog=TestToolCatalog(),
        )
        self.saved = []
        self.low_confidence = []

    async def _persist_workflow_answer(self, conversation_id, answer, actions, sources, *, related_order_id=None, tool_trace=None, reply_to_message_id=None, retrieved_chunks=None):
        self.saved.append((answer, actions))
        return 20 + len(self.saved)

    async def _persist_assistant(self, conversation_id, content, *, tool_calls=None):
        self.saved.append(("tool_call", tool_calls))
        return 1

    async def _persist_tool(self, conversation_id, *, content, tool_call_id):
        self.saved.append(("tool_result", tool_call_id))
        return 2

    async def _record_low_confidence(self, turn, *, source, reason, retrieved_chunks=None):
        self.low_confidence.append(source)


def turn(question: str, history=(), *, user_id: str = "demo") -> PreparedTurn:
    messages = [SystemMessage(content="安全客服"), *history, HumanMessage(content=question)]
    return PreparedTurn(
        conversation_id=1, user_id=user_id, user_message_id=1, raw_question=question,
        prepared_context=PreparedContext(tuple(messages), 50, 0),
        tool_runtime=ToolRuntime(1, lambda: None, Random(0), user_id=user_id, order_data=FakeOrderData()),
    )


async def run_example(intent, question, outputs=(), confident=True, decision=None):
    model = LabelledModel(intent, outputs, decision)
    search = LabelledSearch(confident)
    service = InMemoryWorkflow(model, search)
    prepared = PreparedTurn(
        conversation_id=1, user_id="example-user", user_message_id=1,
        raw_question=question,
        prepared_context=PreparedContext(
            (SystemMessage(content="You are a safe assistant."), HumanMessage(content=question)), 30, 0,
        ),
        tool_runtime=ToolRuntime(1, lambda: None, Random(0), search, user_id="example-user", order_data=service._orders),
    )
    events = [event async for event in service.stream_turn(prepared)]
    return service, model, search, events


async def run_script(question, decisions, *, outputs=(), resolution=None, history=(), confident=True):
    model = ScriptedUnderstandingModel(decisions, outputs, resolution)
    search = LabelledSearch(confident)
    service = InMemoryWorkflow(model, search)
    events = [e async for e in service.stream_turn(turn(question, history))]
    return service, model, search, events
