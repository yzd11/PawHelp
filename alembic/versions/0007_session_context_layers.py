"""Append-only session summaries and safe final-answer tool badges."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision = "0007_session_context_layers"
down_revision = "0006_persistent_demo_orders"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Accept existing instances that applied the documented DDL; never erase data.
    inspector = sa.inspect(op.get_bind())
    columns = {item["name"] for item in inspector.get_columns("conversations")}
    for name, kind in [("summary", sa.Text()), ("summary_upto_msg_id", mysql.BIGINT(unsigned=True)), ("layer1_from_msg_id", mysql.BIGINT(unsigned=True))]:
        if name not in columns:
            op.add_column("conversations", sa.Column(name, kind, nullable=True))
    if "conversation_summaries" not in inspector.get_table_names():
        op.create_table(
            "conversation_summaries",
            sa.Column("id", mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True),
            sa.Column("conversation_id", mysql.BIGINT(unsigned=True), nullable=False),
            sa.Column("seq", mysql.INTEGER(unsigned=True), nullable=False),
            sa.Column("from_msg_id", mysql.BIGINT(unsigned=True), nullable=False),
            sa.Column("upto_msg_id", mysql.BIGINT(unsigned=True), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("created_at", mysql.DATETIME(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
            sa.UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),
            mysql_charset="utf8mb4",
        )
        op.create_index("idx_conv_upto", "conversation_summaries", ["conversation_id", "upto_msg_id"])
    if "tool_trace" not in {item["name"] for item in inspector.get_columns("messages")}:
        op.add_column("messages", sa.Column("tool_trace", mysql.JSON(), nullable=True))


def downgrade() -> None:
    # Explicit downgrade is destructive; normal upgrade only adds nullable data.
    op.drop_column("messages", "tool_trace")
    op.drop_table("conversation_summaries")
    for name in ("layer1_from_msg_id", "summary_upto_msg_id", "summary"):
        op.drop_column("conversations", name)
