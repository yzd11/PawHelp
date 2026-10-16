"""One graph-level tracing seam; actual usage is independent of trace export."""
import asyncio
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import logging
from time import perf_counter
from uuid import uuid4
from urllib.parse import urlparse
import ipaddress

from langchain_core.callbacks import AsyncCallbackHandler
from app.database.models import RequestObservation

logger = logging.getLogger(__name__)
_active = ContextVar("pawhelp_observation", default=None)


class UsageCollector(AsyncCallbackHandler):
    def __init__(self, model):
        self.model, self.rows, self.seen = model, [], set()

    async def on_llm_end(self, response, *, run_id, **kwargs):
        if str(run_id) in self.seen:
            return
        self.seen.add(str(run_id))
        usage = None
        for group in response.generations:
            for generation in group:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or usage
        output = response.llm_output or {}
        usage = usage or output.get("token_usage") or output.get("usage")
        self.rows.append({"run_id": str(run_id), "model": output.get("model_name", self.model),
                          "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")) if usage else None,
                          "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")) if usage else None,
                          "status": "success"})

    async def on_llm_error(self, error, *, run_id, **kwargs):
        if str(run_id) not in self.seen:
            self.seen.add(str(run_id))
            self.rows.append({"run_id": str(run_id), "model": self.model, "input_tokens": None, "output_tokens": None, "status": "failed"})


class Observability:
    def __init__(self, settings, session_factory):
        self.settings, self.sessions, self.client = settings, session_factory, None
        self.enabled = settings.observability_enabled
        if self.enabled and settings.langfuse_enabled:
            url = urlparse(settings.langfuse_base_url)
            try:
                local = url.hostname == "localhost" or ipaddress.ip_address(url.hostname or "").is_private or ipaddress.ip_address(url.hostname or "").is_loopback
            except ValueError:
                local = False
            if url.scheme not in {"http", "https"} or not local or url.username or url.password:
                raise ValueError("LANGFUSE_BASE_URL must explicitly point to your local/private self-hosted server")
            if not settings.langfuse_public_key.get_secret_value() or not settings.langfuse_secret_key.get_secret_value():
                raise ValueError("self-hosted Langfuse keys are required when enabled")
            from langfuse import Langfuse
            self.client = Langfuse(public_key=settings.langfuse_public_key.get_secret_value(), secret_key=settings.langfuse_secret_key.get_secret_value(), base_url=settings.langfuse_base_url)

    @contextmanager
    def request(self, turn):
        record = {"id": str(uuid4()), "intent": "未识别", "collector": UsageCollector(self.settings.openai_model), "status": "success", "started": perf_counter()}
        callbacks = [record["collector"]] if self.enabled else []
        if self.client:
            from langfuse.langchain import CallbackHandler
            callbacks.append(CallbackHandler(public_key=self.settings.langfuse_public_key.get_secret_value()))
        record["callbacks"] = callbacks
        ctx = self.client.start_as_current_observation(name="customer-service", as_type="agent", input={"question": turn.raw_question}) if self.client else nullcontext(None)
        token = _active.set(record)
        try:
            with ctx as span:
                record["span"] = span
                record["trace_id"] = span.trace_id if span else None
                if self.client:
                    from langfuse import propagate_attributes
                    with propagate_attributes(session_id=str(turn.conversation_id), user_id=turn.user_id, trace_name="customer-service"):
                        yield record
                else:
                    yield record
        finally:
            _active.reset(token)

    async def save(self, record, turn):
        if not self.enabled:
            return
        try:
            async with self.sessions() as session, session.begin():
                session.add(RequestObservation(id=record["id"], conversation_id=turn.conversation_id, user_message_id=turn.user_message_id,
                    intent=record["intent"], trace_id=record.get("trace_id"), model_usage=record["collector"].rows,
                    duration_ms=int((perf_counter() - record["started"]) * 1000), status=record["status"]))
        except Exception:
            logger.exception("observation persistence failed request=%s", record["id"])

    async def close(self):
        if self.client:
            await asyncio.to_thread(self.client.flush)


def record_intent(intent):
    record = _active.get()
    if record is not None:
        record["intent"] = intent
        if record.get("span"):
            record["span"].update(metadata={"intent": intent})


def finish_trace(record, state):
    record_intent(state.get("intent", record["intent"]))
    if record.get("span"):
        from langfuse import propagate_attributes
        # Runs on the root request observation, so intent becomes trace metadata too.
        with propagate_attributes(metadata={"intent": record["intent"]}):
            record["span"].update(output={"answer": state.get("answer"), "intent": record["intent"]})
