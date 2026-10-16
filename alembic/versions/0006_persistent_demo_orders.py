"""Persist owned demo orders and logistics; preserve legacy refund records."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision = "0006_persistent_demo_orders"
down_revision = "0005_refund_requests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orders",
        sa.Column("user_id", mysql.VARCHAR(64), primary_key=True),
        sa.Column("order_id", mysql.VARCHAR(64), primary_key=True),
        sa.Column("product", mysql.VARCHAR(200), nullable=False),
        sa.Column("amount", mysql.DECIMAL(10, 2), nullable=False),
        sa.Column("status", mysql.ENUM("pending_payment", "processing", "shipped", "delivered"), nullable=False),
        sa.Column("is_demo", mysql.TINYINT(1), nullable=False, server_default=sa.text("1")),
        sa.Column("created_at", mysql.DATETIME(), nullable=False),
        sa.Column("shipped_at", mysql.DATETIME(), nullable=True),
        sa.Column("delivered_at", mysql.DATETIME(), nullable=True),
    )
    op.create_table(
        "shipments",
        sa.Column("tracking_no", mysql.VARCHAR(64), primary_key=True),
        sa.Column("user_id", mysql.VARCHAR(64), nullable=False),
        sa.Column("order_id", mysql.VARCHAR(64), nullable=False),
        sa.Column("carrier", mysql.VARCHAR(64), nullable=False),
        sa.Column("status", mysql.ENUM("accepted", "in_transit", "out_for_delivery", "delivered"), nullable=False),
        sa.Column("location", mysql.VARCHAR(200), nullable=False),
        sa.Column("estimated_delivery_at", mysql.DATETIME(), nullable=True),
        sa.Column("updated_at", mysql.DATETIME(), nullable=False),
        sa.ForeignKeyConstraint(["user_id", "order_id"], ["orders.user_id", "orders.order_id"], name="fk_shipment_order"),
        sa.UniqueConstraint("user_id", "order_id", name="uq_shipment_order"),
    )
    op.create_table(
        "shipment_events",
        sa.Column("tracking_no", mysql.VARCHAR(64), primary_key=True),
        sa.Column("sequence", mysql.INTEGER(unsigned=True), primary_key=True),
        sa.Column("occurred_at", mysql.DATETIME(), nullable=False),
        sa.Column("location", mysql.VARCHAR(200), nullable=False),
        sa.Column("description", mysql.VARCHAR(300), nullable=False),
        sa.ForeignKeyConstraint(["tracking_no"], ["shipments.tracking_no"]),
    )
    # Existing applications remain valid with NULL; all new submissions fill this owner.
    op.add_column("refund_requests", sa.Column("order_user_id", mysql.VARCHAR(64), nullable=True))
    op.create_foreign_key("fk_refund_order", "refund_requests", "orders", ["order_user_id", "order_id"], ["user_id", "order_id"])


def downgrade() -> None:
    op.drop_constraint("fk_refund_order", "refund_requests", type_="foreignkey")
    op.drop_column("refund_requests", "order_user_id")
    op.drop_table("shipment_events")
    op.drop_table("shipments")
    op.drop_table("orders")
