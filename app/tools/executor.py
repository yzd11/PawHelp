"""Validate, authorize, execute, format and audit at one seam."""
import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
from time import monotonic
from datetime import datetime, timezone

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from app.tools.registry import ToolRegistry, UnknownToolError

logger = logging.getLogger(__name__)


class InvalidToolArguments(Exception):
    public_code = "invalid_tool_arguments"
    public_message = "The requested information is invalid."
    def __init__(self): super().__init__(self.public_message)


class TransientToolError(Exception):
    """Explicitly retryable transport error, not an empty business result."""


class ToolExecutionFailed(Exception):
    public_code = "tool_failed"
    public_message = "The requested service could not complete."
    def __init__(self): super().__init__(self.public_message)


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_call_id: str
    name: str
    content: str
    attempts: int
    status: str = "成功"
    code: str | None = None
    @property
    def ok(self): return self.status == "成功"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def format_result(value) -> str:
    if isinstance(value, list):
        texts = [b.get("text", "") for b in value if isinstance(b, dict) and b.get("type") == "text"]
        value = "\n".join(texts) if texts else value
    if isinstance(value, str):
        try: value = json.loads(value)
        except ValueError: return _json({"message": value})
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if not k.startswith("_") and k not in {"debug", "internal", "raw_payload"}}
        labels = {"shipped": "已发货", "delivered": "已签收", "pending_payment": "待付款", "in_transit": "运输中",
                  "out_for_delivery": "派送中", "in_stock": "有货", "limited": "库存有限", "preorder": "预售"}
        for key in ("status", "availability"):
            if isinstance(value.get(key), str) and value[key] in labels: value.setdefault(key + "_label", labels[value[key]])
    return _json(value)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, *, timeout_seconds: float = 3, max_attempts: int = 2, runtime=None):
        if timeout_seconds <= 0 or max_attempts < 1: raise ValueError("tool execution bounds must be positive")
        self._registry = registry
        self.runtime = runtime or registry.runtime
        self.timeout_seconds, self.max_attempts = timeout_seconds, max_attempts

    async def execute(self, call: Mapping[str, object], *, write_requested: bool = False, confirmed: bool = False,
                      deny_reason: str | None = None, argument_error: str | None = None) -> ToolExecutionResult:
        start = monotonic()
        call_id = str(call.get("id") or "invalid-call")[:64]
        name = str(call.get("name") or "invalid-tool")[:128]
        args = call.get("args", {})
        spec = None
        attempts = 0
        status, code, error, content = "失败", None, None, ""
        try:
            spec = self._registry.spec(name)
            if not isinstance(call.get("id"), str) or not call.get("id") or not isinstance(args, dict): raise InvalidToolArguments()
            try:
                errors = sorted(Draft202012Validator(spec.schema).iter_errors(args), key=lambda e: str(e.path))
            except Exception as schema_error:
                logger.warning("tool schema evaluation failed name=%s", name, exc_info=True)
                error = "工具参数定义无法解析，未执行；请联系管理员检查工具 Schema。"
                raise InvalidToolArguments() from schema_error
            if errors:
                error = "; ".join(f"{'.'.join(map(str, e.path)) or '参数'}: {e.validator} 校验未通过" for e in errors[:5])
                raise InvalidToolArguments()
            if argument_error:
                error = argument_error
                raise InvalidToolArguments()
            if not isinstance(spec.tool.args_schema, dict):
                try: args = spec.tool.args_schema.model_validate(args).model_dump()
                except (ValidationError, TypeError, AttributeError): raise InvalidToolArguments()
            if deny_reason or spec.permission == "deny" or (spec.permission == "write" and not (name == "create_ticket" and spec.source == "builtin" and write_requested and confirmed)):
                status, code, error = "权限拒绝", "permission_denied", deny_reason or "未获本地授权，或客户尚未确认提交；不得声称已完成写入。"
            else:
                limit = 1 if spec.permission == "write" else self.max_attempts
                for attempts in range(1, limit + 1):
                    try:
                        async with asyncio.timeout(self.timeout_seconds): value = await spec.tool.ainvoke(args)
                        content = format_result(value)
                        payload = json.loads(content)
                        if isinstance(payload, dict) and (payload.get("error") or payload.get("code") in {"order_not_found", "not_shipped", "not_found"}):
                            status, code, error = "失败", "not_found", str(payload.get("error") or payload.get("message") or "查询暂无结果")[:512]
                        else: status = "成功"
                        break
                    except asyncio.CancelledError: raise
                    except (TimeoutError, TransientToolError, ConnectionError) as exc:
                        if attempts == limit:
                            status, code, error = ("超时" if isinstance(exc, TimeoutError) else "失败"), "tool_timeout" if isinstance(exc, TimeoutError) else "tool_unavailable", "依赖服务暂时不可用；未能取得结果，不得猜测。"
                            if spec.permission == "write":
                                error = "写入结果未知，未自动重试；请核对工单记录或联系人工，不得声称创建成功或失败。"
                    except Exception:
                        logger.exception("tool fault name=%s source=%s server=%s", name, spec.source, spec.server)
                        status, code, error = "失败", "tool_failed", "工具服务故障；不得把故障当成业务否定或猜测结果。"
                        break
        except InvalidToolArguments:
            status, code, error = "校验拦下", "invalid_tool_arguments", error or "参数格式不合法，请补齐必填项或调整类型/取值后重试。"
        except UnknownToolError:
            status, code, error = "权限拒绝", "unknown_tool", "工具未登记，不能执行。"
        except asyncio.CancelledError:
            cancelled = ToolExecutionResult(call_id, name, _json({"code": "cancelled", "error": "调用被取消，结果未知"}), attempts, "失败", "cancelled")
            await asyncio.shield(self._audit(cancelled, spec, call.get("args"), "调用被取消，结果未知", int((monotonic() - start) * 1000)))
            raise
        if not content or (status != "成功" and code != "not_found"): content = _json({"code": code, "error": error, "status": status})
        result = ToolExecutionResult(call_id, name, content, attempts, status, code)
        await self._audit(result, spec, call.get("args"), error, int((monotonic() - start) * 1000))
        return result

    async def _audit(self, result, spec, args, error, duration):
        if self.runtime is None: return
        try:
            from app.database.models import ToolAuditLog
            async with self.runtime.session_factory() as session, session.begin():
                session.add(ToolAuditLog(conversation_id=self.runtime.conversation_id or None,
                    tool_call_id=result.tool_call_id, tool_name=result.name,
                    tool_source=spec.source if spec else "builtin", mcp_server=spec.server if spec else None,
                    arguments=args if isinstance(args, dict) else {"invalid_shape": type(args).__name__},
                    result_summary=result.content[:4000], status=result.status, error_message=(error or "")[:512] or None,
                    retry_count=max(0, result.attempts - 1), duration_ms=duration, created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        except Exception: logger.exception("tool_audit write failed call_id=%s; execution result unchanged", result.tool_call_id)


__all__ = ["InvalidToolArguments", "ToolExecutionFailed", "ToolExecutionResult", "ToolExecutor", "TransientToolError", "UnknownToolError"]
