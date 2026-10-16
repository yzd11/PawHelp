"""Persist suggested workflow actions on assistant messages.

Revision ID: 0004_workflow_suggested_actions
Revises: 0003_knowledge_source_tracking
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


revision: str = "0004_workflow_suggested_actions"
down_revision: str | None = "0003_knowledge_source_tracking"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("suggested_actions", mysql.JSON(), nullable=True))
    op.add_column("messages", sa.Column("source_refs", mysql.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "source_refs")
    op.drop_column("messages", "suggested_actions")
