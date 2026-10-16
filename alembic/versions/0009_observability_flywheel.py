"""Non-destructive observability, exact-turn feedback and review persistence."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import BIGINT, JSON, VARCHAR, ENUM
from app.database.models import EvalRun, ReviewQueue, RequestObservation

revision = "0009_observability_flywheel"
down_revision = "0008_tool_system"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    bind.execute(sa.text("SET NAMES utf8mb4"))
    for table in (ReviewQueue.__table__, EvalRun.__table__, RequestObservation.__table__):
        if table.name not in sa.inspect(bind).get_table_names():
            table.create(bind)
        else:
            missing = {c.name for c in table.columns} - {c["name"] for c in sa.inspect(bind).get_columns(table.name)}
            # The user may have applied the published observability DDL already.
            for name in missing:
                col = table.columns[name]
                if not col.nullable:
                    raise RuntimeError(f"existing {table.name} missing required column {name}; preserve data")
                op.add_column(table.name, sa.Column(name, col.type, nullable=True))
    columns = {
        "messages": [sa.Column("reply_to_message_id", BIGINT(unsigned=True)), sa.Column("retrieved_chunks", JSON), sa.Column("feedback", ENUM("up", "down"))],
        "low_confidence_questions": [sa.Column("retrieved_chunks", JSON), sa.Column("matched_review_id", BIGINT(unsigned=True)), sa.Column("event_key", VARCHAR(128))],
    }
    for name, additions in columns.items():
        existing = {c["name"] for c in sa.inspect(bind).get_columns(name)}
        for column in additions:
            if column.name not in existing:
                op.add_column(name, column)
    inspector = sa.inspect(bind)
    if not any(f["constrained_columns"] == ["matched_review_id"] for f in inspector.get_foreign_keys("low_confidence_questions")):
        op.create_foreign_key("fk_lcq_review", "low_confidence_questions", "review_queue", ["matched_review_id"], ["id"], ondelete="SET NULL")
    if not any(i["column_names"] == ["matched_review_id"] for i in inspector.get_indexes("low_confidence_questions")):
        op.create_index("idx_matched_review_id", "low_confidence_questions", ["matched_review_id"])
    if not any(i["column_names"] == ["event_key"] and i["unique"] for i in inspector.get_indexes("low_confidence_questions")):
        op.create_index("uk_lcq_event", "low_confidence_questions", ["event_key"], unique=True)


def downgrade():
    raise RuntimeError("observability contains review and usage history; downgrade requires an explicit backup/export plan")
