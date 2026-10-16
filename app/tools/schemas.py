"""Explicit model-visible schemas for customer-service business tools."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OrderArgs(BaseModel):
    """Identify an order for order or logistics demonstration data."""

    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(min_length=1, max_length=64)


class LogisticsArgs(BaseModel):
    """Prefer the tracking number from query_order; keep order_id compatibility."""

    model_config = ConfigDict(extra="forbid")
    tracking_no: str | None = Field(default=None, min_length=1, max_length=64, description="运单号，由查询订单返回")
    order_id: str | None = Field(default=None, min_length=1, max_length=64, description="兼容旧调用的订单号")

    @model_validator(mode="after")
    def require_identifier(self):
        if not self.tracking_no and not self.order_id:
            raise ValueError("tracking_no or order_id is required")
        return self


class ProductArgs(BaseModel):
    """Identify a product for demonstration catalogue data."""

    product: str = Field(min_length=1, max_length=200)


class FAQArgs(BaseModel):
    """Supply a customer question for knowledge search."""

    keyword: str = Field(min_length=1, max_length=100)


class TicketArgs(BaseModel):
    """Describe a request for human customer-support follow-up."""

    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    ticket_type: Literal["售后", "投诉", "咨询"]
