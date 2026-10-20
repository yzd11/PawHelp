"""Request and response models for customer-service operations."""

from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StrictBool, field_validator


class ChatRequest(BaseModel):
    """One browser-authored turn for the server-owned persistence workflow."""

    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    conversation_id: PositiveInt | None = None
    message: str

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        """Normalize a customer turn and reject empty content."""

        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        return value


class PublicToolTrace(BaseModel):
    """Registry-approved metadata for one tool badge in hydrated history."""

    tool_call_id: str
    name: str
    label: str
    status: Literal["succeeded", "failed"] = "succeeded"
    source: Literal["builtin", "mcp"] = "builtin"
    mcp_server: str | None = None


class PublicSource(BaseModel):
    """Browser-safe citation metadata resolved from a private tool result."""

    number: PositiveInt
    chunk_id: PositiveInt
    category: str
    section_path: str | None = None


class PublicMessage(BaseModel):
    """Browser-safe hydrated conversation content."""

    id: int
    role: Literal["user", "assistant"]
    content: str
    tool_trace: list[PublicToolTrace] = Field(default_factory=list)
    sources: list[PublicSource] = Field(default_factory=list)
    suggested_actions: list[Literal["转人工", "建工单", "申请退款"]] = Field(default_factory=list)
    related_order_id: str | None = None
    feedback: Literal["up", "down"] | None = None


class TicketActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    message_id: PositiveInt
    ticket_type: Literal["售后", "投诉", "咨询"]
    description: str = Field(min_length=1, max_length=2000)

    @field_validator("description")
    @classmethod
    def normalize_ticket_description(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("description must not be blank")
        return value


class OrderSelectionRequest(BaseModel):
    """A user click on an explicitly synthetic order card."""

    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    message_id: PositiveInt
    order_id: str = Field(pattern=r"^\d{4,20}$")


class TicketConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: UUID
    message_id: PositiveInt
    proposal_id: str = Field(min_length=1, max_length=64)
    confirmed: StrictBool


class RefundRequestAction(BaseModel):
    """Explicitly submitted form, separate from model-generated chat text."""

    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    message_id: PositiveInt
    order_id: str = Field(pattern=r"^\d{4,20}$")
    reason: Literal["七天无理由", "商品破损", "质量问题", "发错或漏发", "其他"]
    description: str | None = Field(default=None, max_length=1000)

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        return (value.strip() or None) if value is not None else None


class RequestType(str, Enum):
    """Supported categories for an after-sales customer request."""

    REFUND = "refund"
    RETURN = "return"
    EXCHANGE = "exchange"
    REPAIR = "repair"
    LOGISTICS = "logistics"
    COMPLAINT = "complaint"
    INVOICE = "invoice"
    OTHER = "other"


class AfterSalesExtraction(BaseModel):
    """The normalized facts extracted from one after-sales description."""

    order_id: str | None = Field(
        default=None, description="Customer-provided order identifier, if available."
    )
    request_type: RequestType = Field(description="After-sales request category.")
    expected_solution: str | None = Field(
        default=None, description="Resolution requested by the customer, if available."
    )


class ExtractionRequest(BaseModel):
    """One customer description to normalize into after-sales facts."""

    description: str = Field(description="Customer-provided after-sales description.")

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, value: str) -> str:
        """Normalize and reject whitespace-only customer descriptions."""

        description = value.strip()
        if not description:
            raise ValueError("description must not be empty")
        return description


KnowledgeContentType = Literal["faq", "policy", "manual", "conversation"]
ManualKnowledgeContentType = Literal["faq", "policy", "manual"]
VectorizeStatus = Literal["pending", "done"]


class KnowledgeMaterial(BaseModel):
    """One local Markdown source available for chunk preview."""

    path: str
    name: str
    content_type: KnowledgeContentType
    size: int
    character_count: int = 0
    line_count: int = 0
    chunk_count: int = 0
    key_clause_count: int = 0
    table_chunk_count: int = 0


class KnowledgePreviewRequest(BaseModel):
    """Select one repository-owned Markdown source for a dry-run preview."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=512)


class MarkdownKnowledgeRequest(BaseModel):
    """A full Markdown document to preview or ingest through the real chunker."""

    model_config = ConfigDict(extra="forbid")

    markdown: str = Field(min_length=1, max_length=200_000)
    content_type: ManualKnowledgeContentType = "manual"
    vectorize_after: bool = True

    @field_validator("markdown")
    @classmethod
    def normalize_markdown(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("markdown must not be empty")
        return normalized


class KnowledgeDraftView(BaseModel):
    """One structure-aware chunk before persistence."""

    model_config = ConfigDict(from_attributes=True)

    category: str
    questions: str
    answer: str
    section_path: str | None
    content_type: KnowledgeContentType | None
    is_key_clause: bool


class MarkdownKnowledgePreviewItem(KnowledgeDraftView):
    """One preview row annotated with table and duplicate signals."""

    is_table_block: bool
    duplicate: bool


class MarkdownKnowledgePreviewResult(BaseModel):
    items: list[MarkdownKnowledgePreviewItem]
    total: int
    key_clause_count: int
    table_block_count: int
    duplicate_count: int
    ingestible_count: int


class MarkdownKnowledgeIngestResult(BaseModel):
    inserted: int
    skipped_duplicates: int
    skipped_conflicts: int = 0
    vectorized: int


class ManualKnowledgeRequest(BaseModel):
    """A manually authored knowledge entry."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1, max_length=255)
    questions: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    section_path: str | None = Field(default=None, max_length=512)
    content_type: ManualKnowledgeContentType = "manual"
    is_key_clause: bool = False

    @field_validator("category", "questions", "answer", "section_path")
    @classmethod
    def normalize_knowledge_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("knowledge text must not be empty")
        return normalized


class KnowledgeChunkView(KnowledgeDraftView):
    """Browser-safe authoritative knowledge state."""

    id: int
    vector_id: str | None
    vectorize_status: VectorizeStatus
    source_kind: str
    source_path: str | None


class KnowledgeChunkPage(BaseModel):
    """One server-paginated slice plus whole-library status totals."""

    items: list[KnowledgeChunkView]
    page: PositiveInt
    page_size: PositiveInt
    total: int
    total_pages: PositiveInt
    all_total: int
    pending_total: int
    done_total: int


class KnowledgeOverview(BaseModel):
    materials: int
    mysql_total: int
    pending_total: int
    done_total: int
    milvus_total: int | None
    milvus_collection: str
    dual_write_consistent: bool | None
    # None: no published calibration records the non-document entries it was collected with.
    calibration_entries_changed: bool | None = None


class KnowledgeMiningStats(BaseModel):
    unprocessed_conversations: int
    extracted: int
    kept: int
    discarded: int
    batches: int


class KnowledgeMiningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_external_model_usage: Literal[True]
    vectorize_after: bool = True


class KnowledgeMiningResult(BaseModel):
    conversations: int
    extracted: int
    kept: int
    discarded: int
    vectorized: int


class KnowledgeVectorizeResult(BaseModel):
    """Result of one recoverable pending-vectorization action."""

    vectorized: int


class KnowledgeDeleteResult(BaseModel):
    """Result of deleting one authoritative chunk and its vector."""

    deleted: int


class KnowledgeRecheckFlag(BaseModel):
    """A kept entry that duplicates or contradicts the rebuilt knowledge."""

    chunk_id: int
    verdict: Literal["duplicate", "conflict"]
    message: str


class KnowledgeSyncResult(BaseModel):
    """Result of replacing imported sources with the current server files."""

    replaced: int
    ingested: int
    vectorized: int
    rechecked: int = 0
    flagged: list[KnowledgeRecheckFlag] = Field(default_factory=list)
    recheck_unavailable: bool = False


class KnowledgeSearchRequest(BaseModel):
    """A semantic retrieval self-test query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    strategy: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = (
        "hybrid_rerank"
    )
    top_k: int = Field(default=5, ge=1, le=20)
    categories: list[str] = Field(default_factory=list, max_length=10)
    content_types: list[KnowledgeContentType] = Field(default_factory=list, max_length=4)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized


class KnowledgeSearchResult(BaseModel):
    """Retrieval diagnostics for the local knowledge console."""

    matches: list["KnowledgeSearchMatch"]
    strategy: Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
    confident: bool
    normalized_query: str
    synonyms: list[str]


class KnowledgeSearchMatch(KnowledgeChunkView):
    score: float


class RagEvalRunRequest(BaseModel):
    """Explicit confirmation for one potentially billable evaluation run."""

    model_config = ConfigDict(extra="forbid")

    confirm_external_model_usage: Literal[True]
    full_generation: bool = True


class FaithCaseReviewRequest(BaseModel):
    """Human decision for one model-flagged answer, guarded against stale forms."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["未解决", "已解决", "无需解决"]
    resolution: str | None = Field(default=None, max_length=300)
    expected_seen_count: PositiveInt
    expected_status: Literal["未解决", "已解决", "无需解决"]
    expected_resolution: str | None = Field(default=None, max_length=300)

    @field_validator("resolution")
    @classmethod
    def normalize_resolution(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None
