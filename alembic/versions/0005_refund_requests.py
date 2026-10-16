"""Create explicit refund applications and bind offers to one demo order.

Revision ID: 0005_refund_requests
Revises: 0004_workflow_suggested_actions
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


revision: str = "0005_refund_requests"
down_revision: str | None = "0004_workflow_suggested_actions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("related_order_id", mysql.VARCHAR(64), nullable=True))
    op.create_table(
        "refund_requests",
        sa.Column("request_no", mysql.VARCHAR(32), primary_key=True),
        sa.Column("conversation_id", mysql.BIGINT(unsigned=True), nullable=False),
        sa.Column("source_message_id", mysql.BIGINT(unsigned=True), nullable=False),
        sa.Column("order_id", mysql.VARCHAR(64), nullable=False),
        sa.Column("reason", mysql.VARCHAR(32), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", mysql.ENUM("待人工审核", "已受理", "已驳回"), nullable=False, server_default="待人工审核"),
        sa.Column("created_at", mysql.DATETIME(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["source_message_id"], ["messages.id"]),
        sa.UniqueConstraint("source_message_id", name="uq_refund_source_message_id"),
    )
    op.create_index("idx_refund_conversation_id", "refund_requests", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("idx_refund_conversation_id", table_name="refund_requests")
    op.drop_table("refund_requests")
    op.drop_column("messages", "related_order_id")
