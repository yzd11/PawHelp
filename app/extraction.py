"""Structured extraction of after-sales customer requests."""

import logging
from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage

from app.prompts import render_after_sales_extraction_messages
from app.schemas import AfterSalesExtraction

logger = logging.getLogger(__name__)


class StructuredRunnable(Protocol):
    """Minimal structured runnable boundary used by the extractor."""

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> object: ...


class StructuredOutputModel(Protocol):
    """Model capability needed to construct a structured runnable."""

    def with_structured_output(
        self, schema: type[AfterSalesExtraction], *, method: str
    ) -> StructuredRunnable: ...


class StructuredOutputError(Exception):
    """Safe public failure for unavailable or malformed structured output."""


class AfterSalesExtractor:
    """Normalize customer after-sales descriptions with one configured runnable."""

    def __init__(self, model: StructuredOutputModel, *, method: str) -> None:
        self._structured_model = model.with_structured_output(
            AfterSalesExtraction,
            method=method,
        )

    async def extract(self, description: str) -> AfterSalesExtraction:
        """Extract normalized facts without exposing upstream details."""

        try:
            result = await self._structured_model.ainvoke(
                render_after_sales_extraction_messages(description)
            )
            return AfterSalesExtraction.model_validate(result)
        except Exception as error:
            logger.warning(
                "structured_output_error exception_class=%s", type(error).__name__
            )
            raise StructuredOutputError() from None
