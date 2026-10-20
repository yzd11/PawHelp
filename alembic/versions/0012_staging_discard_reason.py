"""Why a mined QA candidate was discarded by the write-time consistency check."""
from alembic import op
import sqlalchemy as sa

revision = "0012_staging_discard_reason"
down_revision = "0011_conversation_titles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable: exact-duplicate discards and earlier rows carry no reason.
    columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("qa_extraction_staging")}
    if "discard_reason" not in columns:
        op.add_column("qa_extraction_staging", sa.Column("discard_reason", sa.VARCHAR(500), nullable=True))


def downgrade() -> None:
    op.drop_column("qa_extraction_staging", "discard_reason")
