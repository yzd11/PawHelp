"""Offline topic result table, compatible with existing installations."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, JSON

revision = "0010_topic_classifications"
down_revision = "0009_observability_flywheel"
branch_labels = None
depends_on = None


def upgrade():
    bind=op.get_bind()
    bind.execute(sa.text("SET NAMES utf8mb4"))
    inspector=sa.inspect(bind)
    if "topic_classifications" in inspector.get_table_names():
        columns={c["name"]:c for c in inspector.get_columns("topic_classifications")}
        if not {"id","question_id","labels","classified_at"} <= columns.keys():
            raise RuntimeError("existing classification table differs; preserve it and review schema")
        if not any(c["column_names"]==["question_id"] for c in inspector.get_unique_constraints("topic_classifications")):
            raise RuntimeError("existing classification table lacks unique question_id; do not silently alter populated data")
        return
    op.create_table("topic_classifications",
        sa.Column("id",BIGINT(unsigned=True),primary_key=True,autoincrement=True),
        sa.Column("question_id",BIGINT(unsigned=True),nullable=False),
        sa.Column("labels",JSON,nullable=False),
        sa.Column("classified_at",DATETIME,nullable=False,server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("question_id",name="uk_question_id"),
        sa.ForeignKeyConstraint(["question_id"],["low_confidence_questions.id"],name="fk_topic_question"),
        mysql_engine="InnoDB",mysql_charset="utf8mb4")


def downgrade():
    op.drop_table("topic_classifications")
