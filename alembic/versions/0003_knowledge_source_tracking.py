"""Track where each knowledge chunk came from.

Revision ID: 0003_knowledge_source_tracking
Revises: 0002_rag_quality_tables
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


revision: str = "0003_knowledge_source_tracking"
down_revision: str | None = "0002_rag_quality_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_chunks",
        sa.Column(
            "source_kind",
            mysql.VARCHAR(24),
            nullable=False,
            server_default=sa.text("'legacy'"),
            comment="来源类型:markdown / faq_table / manual / conversation / legacy",
        ),
    )
    op.add_column(
        "knowledge_chunks",
        sa.Column(
            "source_path",
            mysql.VARCHAR(512),
            nullable=True,
            comment="Markdown 相对路径等来源标识",
        ),
    )
    op.create_index("idx_source_kind", "knowledge_chunks", ["source_kind"])
    op.execute(
        "UPDATE knowledge_chunks SET source_kind = 'conversation' "
        "WHERE content_type = 'conversation'"
    )


def downgrade() -> None:
    op.drop_index("idx_source_kind", table_name="knowledge_chunks")
    op.drop_column("knowledge_chunks", "source_path")
    op.drop_column("knowledge_chunks", "source_kind")
