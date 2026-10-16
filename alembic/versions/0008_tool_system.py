"""Unified tool audit, no foreign keys and no destructive upgrade."""
from alembic import op
import sqlalchemy as sa
from app.database.models import ToolAuditLog

revision = "0008_tool_system"
down_revision = "0007_session_context_layers"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "tool_audit_logs" not in inspector.get_table_names():
        ToolAuditLog.__table__.create(bind)
    else:
        expected = {c.name for c in ToolAuditLog.__table__.columns}
        if expected - {c["name"] for c in inspector.get_columns("tool_audit_logs")}:
            raise RuntimeError("existing tool_audit_logs schema incomplete; preserve data and repair schema first")
        if inspector.get_foreign_keys("tool_audit_logs"):
            raise RuntimeError("tool_audit_logs must not have foreign keys")


def downgrade():
    op.drop_table("tool_audit_logs")
