"""Network-free model doubles for chat-stream tests."""

from collections.abc import AsyncIterator, Callable, Sequence

from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessageChunk, BaseMessage

from app.config import Settings


StreamItem = object | BaseException


def fake_settings(**overrides: object) -> Settings:
    """Offline settings that never read the developer's .env or reach a real service."""

    values = dict(
        _env_file=None, chat_base_url="http://localhost/v1", openai_model="fake",
        openai_api_key="fake", database_url="mysql+asyncmy://unused", admin_token="test-only-admin",
    )
    values.update(overrides)
    return Settings(**values)


def _chunk_from_item(item: StreamItem) -> AIMessageChunk:
    """Mirror a provider stream while allowing a controlled terminal failure."""

    if isinstance(item, BaseException):
        raise item
    if isinstance(item, AIMessageChunk):
        return item
    return AIMessageChunk(content=item)


class FakeChatModel:
    """A deterministic streaming model that records the supplied context."""

    def __init__(
        self, chunks: Sequence[object] = (), error: BaseException | None = None
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.seen_messages: tuple[BaseMessage, ...] = ()
        self.structured_schema: object | None = None
        self.structured_kwargs: dict[str, object] | None = None
        self.structured_runnable: FakeStructuredRunnable | None = None

    async def astream(
        self, messages: Sequence[BaseMessage]
    ) -> AsyncIterator[AIMessageChunk]:
        self.seen_messages = tuple(messages)
        if self.error is not None:
            raise self.error
        for content in self.chunks:
            yield _chunk_from_item(content)

    def with_structured_output(
        self, schema: object, **kwargs: object
    ) -> "FakeStructuredRunnable":
        self.structured_schema = schema
        self.structured_kwargs = kwargs
        self.structured_runnable = FakeStructuredRunnable(None, None)
        return self.structured_runnable


class FakeStructuredRunnable:
    """Deterministic structured-output runnable for extraction tests."""

    def __init__(self, result: object | None, error: Exception | None) -> None:
        self.result = result
        self.error = error
        self.seen_messages: tuple[BaseMessage, ...] = ()

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> object:
        self.seen_messages = tuple(messages)
        if self.error is not None:
            raise self.error
        return self.result


class FakeStructuredModel:
    """Records structured-output configuration and returns a fake runnable."""

    def __init__(
        self, result: object | None = None, error: Exception | None = None
    ) -> None:
        self.result = result
        self.error = error
        self.structured_schema: object | None = None
        self.structured_kwargs: dict[str, object] | None = None
        self.structured_runnable: FakeStructuredRunnable | None = None

    def with_structured_output(
        self, schema: object, **kwargs: object
    ) -> FakeStructuredRunnable:
        self.structured_schema = schema
        self.structured_kwargs = kwargs
        self.structured_runnable = FakeStructuredRunnable(self.result, self.error)
        return self.structured_runnable


class FakeCombinedModel(FakeChatModel):
    """A shared model double supporting streaming chat and extraction."""

    def __init__(
        self,
        chunks: Sequence[object] = (),
        error: Exception | None = None,
        structured_result: object | None = None,
        structured_error: Exception | None = None,
    ) -> None:
        FakeChatModel.__init__(self, chunks=chunks, error=error)
        self._structured_result = structured_result
        self._structured_error = structured_error

    def with_structured_output(
        self, schema: object, **kwargs: object
    ) -> FakeStructuredRunnable:
        self.structured_schema = schema
        self.structured_kwargs = kwargs
        self.structured_runnable = FakeStructuredRunnable(
            self._structured_result, self._structured_error
        )
        return self.structured_runnable


class FakeBoundChatModel:
    """The tool-bound selector returned by ``FakeToolCallingChatModel``."""

    def __init__(self, parent: "FakeToolCallingChatModel") -> None:
        self._parent = parent

    async def astream(
        self, messages: Sequence[BaseMessage]
    ) -> AsyncIterator[AIMessageChunk]:
        self._parent.selector_calls += 1
        self._parent.selector_seen_messages = tuple(messages)
        if self._parent.on_selector_stream is not None:
            self._parent.on_selector_stream()
        for item in self._parent.selector_items:
            yield _chunk_from_item(item)


class FakeToolCallingChatModel:
    """Record bound selection separately from the unbound convergence stream."""

    def __init__(
        self,
        *,
        selector_items: Sequence[StreamItem] = (),
        final_items: Sequence[StreamItem] = (),
        bind_error: BaseException | None = None,
        on_selector_stream: Callable[[], None] | None = None,
        on_final_stream: Callable[[], None] | None = None,
    ) -> None:
        self.selector_items = tuple(selector_items)
        self.final_items = tuple(final_items)
        self.bind_error = bind_error
        self.on_selector_stream = on_selector_stream
        self.on_final_stream = on_final_stream
        self.bound_tool_names: tuple[str, ...] = ()
        self.bind_kwargs: dict[str, object] = {}
        self.bind_calls = 0
        self.selector_calls = 0
        self.final_calls = 0
        self.selector_seen_messages: tuple[BaseMessage, ...] = ()
        self.final_seen_messages: tuple[BaseMessage, ...] = ()

    def bind_tools(
        self, tools: Sequence[BaseTool], **kwargs: object
    ) -> FakeBoundChatModel:
        self.bind_calls += 1
        if self.bind_error is not None:
            raise self.bind_error
        self.bound_tool_names = tuple(tool.name for tool in tools)
        self.bind_kwargs = kwargs
        return FakeBoundChatModel(self)

    async def astream(
        self, messages: Sequence[BaseMessage]
    ) -> AsyncIterator[AIMessageChunk]:
        self.final_calls += 1
        self.final_seen_messages = tuple(messages)
        if self.on_final_stream is not None:
            self.on_final_stream()
        for item in self.final_items:
            yield _chunk_from_item(item)
