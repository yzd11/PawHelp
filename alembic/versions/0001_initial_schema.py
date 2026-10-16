"""Create the approved commerce and knowledge schema.

Revision ID: 0001_initial_schema
Revises: None
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at",
        mysql.DATETIME(),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
        comment="创建时间",
    )


def _updated_at() -> sa.Column:
    return sa.Column(
        "updated_at",
        mysql.DATETIME(),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        comment="更新时间",
    )


def upgrade() -> None:
    op.execute("SET NAMES utf8mb4")
    op.create_table(
        "conversations",
        sa.Column("id", mysql.BIGINT(unsigned=True), autoincrement=True, nullable=False, comment="会话主键"),
        sa.Column("user_id", mysql.VARCHAR(64), nullable=False, comment="用户标识"),
        sa.Column("status", mysql.ENUM("进行中", "已转人工", "已结束"), nullable=False, server_default=sa.text("'进行中'"), comment="处理状态"),
        _created_at(),
        _updated_at(),
        sa.PrimaryKeyConstraint("id"),
        comment="客服会话",
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_user_id", "conversations", ["user_id"])

    op.create_table(
        "messages",
        sa.Column("id", mysql.BIGINT(unsigned=True), autoincrement=True, nullable=False, comment="消息主键"),
        sa.Column("conversation_id", mysql.BIGINT(unsigned=True), nullable=False, comment="所属会话"),
        sa.Column("role", mysql.ENUM("user", "assistant", "tool"), nullable=False, comment="角色:用户/助手/工具结果"),
        sa.Column("content", sa.Text(), nullable=True, comment="消息正文,assistant 纯工具调用时可为空"),
        sa.Column("tool_calls", mysql.JSON(), nullable=True, comment="assistant 消息带的工具调用申请单"),
        sa.Column("tool_call_id", mysql.VARCHAR(64), nullable=True, comment="tool 消息对应的申请单 id,回灌时对号入座"),
        _created_at(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], name="fk_messages_conversation"),
        sa.PrimaryKeyConstraint("id"),
        comment="会话消息流水",
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_conversation_id", "messages", ["conversation_id"])

    op.create_table(
        "faq",
        sa.Column("id", mysql.BIGINT(unsigned=True), autoincrement=True, nullable=False, comment="FAQ 主键"),
        sa.Column("question", mysql.VARCHAR(512), nullable=False, comment="问题"),
        sa.Column("answer", sa.Text(), nullable=False, comment="答案"),
        sa.Column("category", mysql.VARCHAR(64), nullable=False, comment="分类"),
        _created_at(),
        _updated_at(),
        sa.PrimaryKeyConstraint("id"),
        comment="常见问答",
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_category", "faq", ["category"])

    op.create_table(
        "tickets",
        sa.Column("ticket_no", mysql.VARCHAR(32), nullable=False, comment="工单号"),
        sa.Column("conversation_id", mysql.BIGINT(unsigned=True), nullable=False, comment="关联会话"),
        sa.Column("description", sa.Text(), nullable=False, comment="问题描述"),
        sa.Column("ticket_type", mysql.ENUM("售后", "投诉", "咨询"), nullable=False, comment="工单类型"),
        sa.Column("status", mysql.ENUM("待处理", "已处理"), nullable=False, server_default=sa.text("'待处理'"), comment="处理状态"),
        _created_at(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], name="fk_tickets_conversation"),
        sa.PrimaryKeyConstraint("ticket_no"),
        comment="人工工单",
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_conversation_id", "tickets", ["conversation_id"])

    op.create_table(
        "knowledge_chunks",
        sa.Column("id", mysql.BIGINT(unsigned=True), autoincrement=True, nullable=False, comment="chunk 主键,与 Milvus 集合主键对齐"),
        sa.Column("category", mysql.VARCHAR(255), nullable=False, comment="分类 / 上级标题路径,进向量化文本"),
        sa.Column("questions", sa.Text(), nullable=False, comment="问法或本节标题,多个问法换行分隔,进向量化文本"),
        sa.Column("answer", sa.Text(), nullable=False, comment="正文答案,进向量化文本"),
        sa.Column("section_path", mysql.VARCHAR(512), nullable=True, comment="章节路径,元数据,溯源用,不进向量"),
        sa.Column("content_type", mysql.VARCHAR(32), nullable=True, comment="内容类型:faq / policy / manual 等,元数据"),
        sa.Column("is_key_clause", mysql.TINYINT(1), nullable=False, server_default=sa.text("0"), comment="是否关键条款,0 否 1 是,元数据"),
        sa.Column("prev_chunk_id", mysql.BIGINT(unsigned=True), nullable=True, comment="前一块指针,元数据"),
        sa.Column("next_chunk_id", mysql.BIGINT(unsigned=True), nullable=True, comment="后一块指针,元数据"),
        sa.Column("vector_id", mysql.VARCHAR(64), nullable=True, comment="Milvus 集合 knowledge 里的主键,写入后回填"),
        sa.Column("vectorize_status", mysql.ENUM("pending", "done"), nullable=False, server_default=sa.text("'pending'"), comment="待向量化 / 已向量化,双写幂等靠它"),
        _created_at(),
        _updated_at(),
        sa.ForeignKeyConstraint(["prev_chunk_id"], ["knowledge_chunks.id"], name="fk_chunks_prev", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["next_chunk_id"], ["knowledge_chunks.id"], name="fk_chunks_next", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        comment="知识库 chunk 原文权威源",
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_category", "knowledge_chunks", ["category"])
    op.create_index("idx_vectorize_status", "knowledge_chunks", ["vectorize_status"])

    op.create_table(
        "qa_extraction_staging",
        sa.Column("id", mysql.BIGINT(unsigned=True), autoincrement=True, nullable=False, comment="暂存行主键"),
        sa.Column("batch_no", mysql.VARCHAR(64), nullable=False, comment="抽取批次号"),
        sa.Column("source_ref", mysql.VARCHAR(255), nullable=True, comment="来源会话 / 导出文件标识"),
        sa.Column("question", sa.Text(), nullable=False, comment="LLM 从会话抽出的用户问法"),
        sa.Column("answer", sa.Text(), nullable=False, comment="LLM 从会话抽出的客服答案"),
        sa.Column("status", mysql.ENUM("extracted", "kept", "discarded"), nullable=False, server_default=sa.text("'extracted'"), comment="抽取与去重状态"),
        _created_at(),
        sa.PrimaryKeyConstraint("id"),
        comment="历史对话抽 QA 的离线中转暂存表",
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_batch_no", "qa_extraction_staging", ["batch_no"])
    op.create_index("idx_status", "qa_extraction_staging", ["status"])


def downgrade() -> None:
    op.drop_table("qa_extraction_staging")
    op.drop_table("knowledge_chunks")
    op.drop_table("tickets")
    op.drop_table("faq")
    op.drop_table("messages")
    op.drop_table("conversations")
