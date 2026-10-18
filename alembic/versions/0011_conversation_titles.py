"""Short model-generated conversation titles for the sidebar."""
from alembic import op
import sqlalchemy as sa

revision = "0011_conversation_titles"
down_revision = "0010_topic_classifications"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable: existing conversations keep showing their first question.
    if "title" not in {item["name"] for item in sa.inspect(op.get_bind()).get_columns("conversations")}:
        op.add_column("conversations", sa.Column("title", sa.VARCHAR(32), nullable=True))


def downgrade() -> None:
    op.drop_column("conversations", "title")
