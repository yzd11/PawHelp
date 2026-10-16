"""Create retrieval low-confidence and faithfulness ledgers.

Revision ID: 0002_rag_quality_tables
Revises: 0001_initial_schema
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


revision: str = "0002_rag_quality_tables"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET NAMES utf8mb4")
    op.create_table(
        "low_confidence_questions",
        sa.Column("id", mysql.BIGINT(unsigned=True), autoincrement=True, nullable=False, comment="主键"),
        sa.Column("conversation_id", mysql.BIGINT(unsigned=True), nullable=True, comment="来源会话"),
        sa.Column("raw_question", sa.Text(), nullable=False, comment="用户原话,带情绪口语"),
        sa.Column("source", mysql.ENUM("retrieval_low_conf", "self_check", "user_feedback"), nullable=False, comment="入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决"),
        sa.Column("reason", sa.Text(), nullable=True, comment="判不能的原因,留作复盘"),
        sa.Column("created_at", mysql.DATETIME(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP"), comment="入池时间"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], name="fk_lcq_conversation"),
        sa.PrimaryKeyConstraint("id"),
        comment="低置信度问题池",
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_source", "low_confidence_questions", ["source"])
    op.create_index("idx_created_at", "low_confidence_questions", ["created_at"])

    op.create_table(
        "faith_cases",
        sa.Column("id", mysql.BIGINT(unsigned=True), autoincrement=True, nullable=False, comment="主键"),
        sa.Column("eval_id", mysql.VARCHAR(16), nullable=False, comment="评估集题号,如 A43;一题一行"),
        sa.Column("bucket", mysql.VARCHAR(24), nullable=False, comment="题目所属桶:A_policy / B_model / C_colloquial / E_multi"),
        sa.Column("query", mysql.VARCHAR(512), nullable=False, comment="用户问题原文"),
        sa.Column("strategy", mysql.VARCHAR(24), nullable=False, server_default=sa.text("'hybrid_rerank'"), comment="产出这条答案的检索策略"),
        sa.Column("answer", sa.Text(), nullable=False, comment="被判编造的那版生成答案原文"),
        sa.Column("reason", sa.Text(), nullable=False, comment="裁判给的理由:编在哪一句"),
        sa.Column("citations", mysql.JSON(), nullable=True, comment="这一轮喂给模型的 Top-K 证据全集快照"),
        sa.Column("judge_model", mysql.VARCHAR(64), nullable=True, comment="判这条的裁判模型"),
        sa.Column("status", mysql.ENUM("未解决", "已解决", "无需解决"), nullable=False, server_default=sa.text("'未解决'"), comment="处置状态,人工点按钮改"),
        sa.Column("seen_count", mysql.INTEGER(unsigned=True), nullable=False, server_default=sa.text("1"), comment="被判编造的累计次数(跨轮)"),
        sa.Column("first_seen_at", mysql.DATETIME(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP"), comment="第一次被判编造的时间"),
        sa.Column("last_seen_at", mysql.DATETIME(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP"), comment="最近一次被判编造的时间"),
        sa.Column("resolution", mysql.VARCHAR(300), nullable=True, comment="处置说明"),
        sa.Column("resolved_at", mysql.DATETIME(), nullable=True, comment="最近一次被标为已解决/无需解决的时间;复发后仍保留"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("eval_id", name="uk_eval_id"),
        comment="retrieval 忠实度编造个案台账",
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_status", "faith_cases", ["status"])
    op.create_index("idx_last_seen_at", "faith_cases", ["last_seen_at"])


def downgrade() -> None:
    op.drop_table("faith_cases")
    op.drop_table("low_confidence_questions")
