"""ORM declarations for the customer-service and knowledge schemas."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import ForeignKey, ForeignKeyConstraint, Index, Text, UniqueConstraint, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, DECIMAL, ENUM, INTEGER, JSON, TINYINT, VARCHAR
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.schema import FetchedValue

from app.database.base import Base


def _created_at() -> Mapped[datetime]:
    return mapped_column(
        DATETIME(), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )


def _updated_at() -> Mapped[datetime]:
    return mapped_column(
        DATETIME(),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        server_onupdate=FetchedValue(),
    )


class ToolAuditLog(Base):
    """Best-effort tool audit; deliberately no foreign keys."""
    __tablename__ = "tool_audit_logs"
    __table_args__ = (Index("idx_conversation_id", "conversation_id"), Index("idx_tool_name", "tool_name"), Index("idx_status", "status"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(VARCHAR(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(VARCHAR(128), nullable=False)
    tool_source: Mapped[str] = mapped_column(ENUM("builtin", "mcp"), nullable=False)
    mcp_server: Mapped[str | None] = mapped_column(VARCHAR(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(ENUM("成功", "失败", "超时", "校验拦下", "权限拒绝"), nullable=False)
    error_message: Mapped[str | None] = mapped_column(VARCHAR(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(TINYINT(unsigned=True), nullable=False, server_default=text("0"))
    duration_ms: Mapped[int | None] = mapped_column(INTEGER(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = _created_at()


class Conversation(Base):
    """A customer-service conversation owned by one anonymous browser user."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), primary_key=True, autoincrement=True
    )
    user_id: Mapped[str] = mapped_column(VARCHAR(64), nullable=False)
    status: Mapped[str] = mapped_column(
        ENUM("进行中", "已转人工", "已结束"),
        nullable=False,
        server_default=text("'进行中'"),
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    title: Mapped[str | None] = mapped_column(VARCHAR(32), nullable=True)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class ConversationSummary(Base):
    """Immutable facts from one previously unsummarized message interval."""

    __tablename__ = "conversation_summaries"
    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),
        Index("idx_conv_upto", "conversation_id", "upto_msg_id"),
    )
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), ForeignKey("conversations.id"), nullable=False)
    seq: Mapped[int] = mapped_column(INTEGER(unsigned=True), nullable=False)
    from_msg_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), nullable=False)
    upto_msg_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class Message(Base):
    """One ordered user, assistant, or tool record in a conversation."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(
        ENUM("user", "assistant", "tool"), nullable=False
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[dict[str, object] | list[object] | None] = mapped_column(
        JSON, nullable=True
    )
    tool_call_id: Mapped[str | None] = mapped_column(VARCHAR(64), nullable=True)
    tool_trace: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    suggested_actions: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    source_refs: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    related_order_id: Mapped[str | None] = mapped_column(VARCHAR(64), nullable=True)
    reply_to_message_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    retrieved_chunks: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    feedback: Mapped[str | None] = mapped_column(ENUM("up", "down"), nullable=True)
    created_at: Mapped[datetime] = _created_at()


class FAQ(Base):
    """A manually entered FAQ that can be imported into the knowledge base."""

    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), primary_key=True, autoincrement=True
    )
    question: Mapped[str] = mapped_column(VARCHAR(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(VARCHAR(64), nullable=False)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class Ticket(Base):
    """A human-support escalation created from a trusted tool call."""

    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(VARCHAR(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id"), nullable=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(
        ENUM("售后", "投诉", "咨询"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        ENUM("待处理", "已处理"), nullable=False, server_default=text("'待处理'")
    )
    created_at: Mapped[datetime] = _created_at()


class Order(Base):
    """A stable order owned by one demo browser identity."""

    __tablename__ = "orders"

    user_id: Mapped[str] = mapped_column(VARCHAR(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(VARCHAR(64), primary_key=True)
    product: Mapped[str] = mapped_column(VARCHAR(200), nullable=False)
    amount: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    status: Mapped[str] = mapped_column(
        ENUM("pending_payment", "processing", "shipped", "delivered"), nullable=False,
    )
    is_demo: Mapped[bool] = mapped_column(TINYINT(1), nullable=False, server_default=text("1"))
    created_at: Mapped[datetime] = mapped_column(DATETIME(), nullable=False)
    shipped_at: Mapped[datetime | None] = mapped_column(DATETIME(), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DATETIME(), nullable=True)


class Shipment(Base):
    """One shipment linked to the owner's order, with a current status."""

    __tablename__ = "shipments"
    __table_args__ = (
        ForeignKeyConstraint(["user_id", "order_id"], ["orders.user_id", "orders.order_id"], name="fk_shipment_order"),
        UniqueConstraint("user_id", "order_id", name="uq_shipment_order"),
    )

    tracking_no: Mapped[str] = mapped_column(VARCHAR(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(VARCHAR(64), nullable=False)
    order_id: Mapped[str] = mapped_column(VARCHAR(64), nullable=False)
    carrier: Mapped[str] = mapped_column(VARCHAR(64), nullable=False)
    status: Mapped[str] = mapped_column(
        ENUM("accepted", "in_transit", "out_for_delivery", "delivered"), nullable=False,
    )
    location: Mapped[str] = mapped_column(VARCHAR(200), nullable=False)
    estimated_delivery_at: Mapped[datetime | None] = mapped_column(DATETIME(), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(), nullable=False)


class ShipmentEvent(Base):
    """An ordered, timestamped shipment observation."""

    __tablename__ = "shipment_events"

    tracking_no: Mapped[str] = mapped_column(VARCHAR(64), ForeignKey("shipments.tracking_no"), primary_key=True)
    sequence: Mapped[int] = mapped_column(INTEGER(unsigned=True), primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DATETIME(), nullable=False)
    location: Mapped[str] = mapped_column(VARCHAR(200), nullable=False)
    description: Mapped[str] = mapped_column(VARCHAR(300), nullable=False)


class RefundRequest(Base):
    """User-submitted application, never an executed payment refund."""

    __tablename__ = "refund_requests"
    __table_args__ = (
        ForeignKeyConstraint(["order_user_id", "order_id"], ["orders.user_id", "orders.order_id"], name="fk_refund_order"),
    )

    request_no: Mapped[str] = mapped_column(VARCHAR(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id"), nullable=False
    )
    source_message_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("messages.id"), unique=True, nullable=False
    )
    order_id: Mapped[str] = mapped_column(VARCHAR(64), nullable=False)
    order_user_id: Mapped[str | None] = mapped_column(VARCHAR(64), nullable=True)
    reason: Mapped[str] = mapped_column(VARCHAR(32), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        ENUM("待人工审核", "已受理", "已驳回"), nullable=False,
        server_default=text("'待人工审核'"),
    )
    created_at: Mapped[datetime] = _created_at()


class KnowledgeChunk(Base):
    """The authoritative text and vectorization state for one knowledge chunk."""

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), primary_key=True, autoincrement=True
    )
    category: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str | None] = mapped_column(VARCHAR(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(VARCHAR(32), nullable=True)
    source_kind: Mapped[str] = mapped_column(
        VARCHAR(24), nullable=False, server_default=text("'legacy'")
    )
    source_path: Mapped[str | None] = mapped_column(VARCHAR(512), nullable=True)
    is_key_clause: Mapped[bool] = mapped_column(
        TINYINT(1), nullable=False, server_default=text("0")
    )
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    vector_id: Mapped[str | None] = mapped_column(VARCHAR(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        ENUM("pending", "done"),
        nullable=False,
        server_default=text("'pending'"),
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class QAExtractionStaging(Base):
    """A conversation-derived QA candidate awaiting global deduplication."""

    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), primary_key=True, autoincrement=True
    )
    batch_no: Mapped[str] = mapped_column(VARCHAR(64), nullable=False)
    source_ref: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        ENUM("extracted", "kept", "discarded"),
        nullable=False,
        server_default=text("'extracted'"),
    )
    discard_reason: Mapped[str | None] = mapped_column(VARCHAR(500), nullable=True)
    created_at: Mapped[datetime] = _created_at()


class LowConfidenceQuestion(Base):
    """A question rejected because the available knowledge was not useful."""

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id"), nullable=True
    )
    raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(
        ENUM("retrieval_low_conf", "self_check", "user_feedback"), nullable=False
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_chunks: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    matched_review_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), ForeignKey("review_queue.id", ondelete="SET NULL"), nullable=True, index=True)
    event_key: Mapped[str | None] = mapped_column(VARCHAR(128), nullable=True, unique=True)
    created_at: Mapped[datetime] = _created_at()


class TopicClassification(Base):
    """Current offline topic result; never written by realtime chat."""
    __tablename__ = "topic_classifications"
    __table_args__ = (UniqueConstraint("question_id", name="uk_question_id"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), ForeignKey("low_confidence_questions.id", name="fk_topic_question"), nullable=False)
    labels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    classified_at: Mapped[datetime] = _created_at()


class ReviewQueue(Base):
    __tablename__ = "review_queue"
    __table_args__ = (Index("idx_review_status", "review_status"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    normalized_question: Mapped[str] = mapped_column(VARCHAR(512), nullable=False)
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(INTEGER(unsigned=True), nullable=False, server_default=text("1"))
    review_status: Mapped[str] = mapped_column(ENUM("待审", "通过", "驳回"), nullable=False, server_default=text("'待审'"))
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_chunk_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    publication_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class EvalRun(Base):
    __tablename__ = "eval_runs"
    __table_args__ = (Index("idx_created_at", "created_at"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    triggered_by: Mapped[str] = mapped_column(ENUM("定时", "手动"), nullable=False, server_default=text("'定时'"))
    dataset_size: Mapped[int] = mapped_column(INTEGER(unsigned=True), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class RequestObservation(Base):
    """Actual usage for one graph invocation, including interrupted/resumed invocations."""
    __tablename__ = "request_observations"
    __table_args__ = (Index("idx_observation_intent", "intent"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})
    id: Mapped[str] = mapped_column(VARCHAR(36), primary_key=True)
    conversation_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    user_message_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    intent: Mapped[str] = mapped_column(VARCHAR(32), nullable=False)
    trace_id: Mapped[str | None] = mapped_column(VARCHAR(64), nullable=True)
    model_usage: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    duration_ms: Mapped[int] = mapped_column(INTEGER(unsigned=True), nullable=False)
    status: Mapped[str] = mapped_column(VARCHAR(32), nullable=False)
    created_at: Mapped[datetime] = _created_at()


class FaithCase(Base):
    """A persistent ledger row for one evaluation case judged unfaithful."""

    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), primary_key=True, autoincrement=True
    )
    eval_id: Mapped[str] = mapped_column(VARCHAR(16), nullable=False, unique=True)
    bucket: Mapped[str] = mapped_column(VARCHAR(24), nullable=False)
    query: Mapped[str] = mapped_column(VARCHAR(512), nullable=False)
    strategy: Mapped[str] = mapped_column(
        VARCHAR(24), nullable=False, server_default=text("'hybrid_rerank'")
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(VARCHAR(64), nullable=True)
    status: Mapped[str] = mapped_column(
        ENUM("未解决", "已解决", "无需解决"),
        nullable=False,
        server_default=text("'未解决'"),
    )
    seen_count: Mapped[int] = mapped_column(
        INTEGER(unsigned=True), nullable=False, server_default=text("1")
    )
    first_seen_at: Mapped[datetime] = _created_at()
    last_seen_at: Mapped[datetime] = _created_at()
    resolution: Mapped[str | None] = mapped_column(VARCHAR(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DATETIME(), nullable=True)


# Preserve the approved DDL names and avoid ORM-generated index names.
Index("idx_user_id", Conversation.user_id)
Index("idx_conversation_id", Message.conversation_id)
Index("idx_category", FAQ.category)
Index("idx_conversation_id", Ticket.conversation_id)
Index("idx_refund_conversation_id", RefundRequest.conversation_id)
Index("idx_category", KnowledgeChunk.category)
Index("idx_vectorize_status", KnowledgeChunk.vectorize_status)
Index("idx_source_kind", KnowledgeChunk.source_kind)
Index("idx_batch_no", QAExtractionStaging.batch_no)
Index("idx_status", QAExtractionStaging.status)
Index("idx_source", LowConfidenceQuestion.source)
Index("idx_created_at", LowConfidenceQuestion.created_at)
Index("idx_status", FaithCase.status)
Index("idx_last_seen_at", FaithCase.last_seen_at)
