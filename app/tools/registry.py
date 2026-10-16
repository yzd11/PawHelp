"""One trusted registry for built-in and discovered tools."""
from collections.abc import Iterable
from dataclasses import dataclass
import re

from jsonschema import Draft202012Validator
from langchain_core.tools import BaseTool


class UnknownToolError(Exception):
    public_code = "unknown_tool"
    public_message = "The requested service is unavailable."
    def __init__(self): super().__init__(self.public_message)


_SAFE_LABELS = {"query_order": "查询订单", "query_product": "查询商品", "query_logistics": "查询物流",
                "query_faq": "查询常见问题", "create_ticket": "创建人工工单",
                "query_warranty": "查询保修", "query_return_progress": "查询退货进度"}


@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool
    source: str = "builtin"
    server: str | None = None
    permission: str = "read"
    label: str | None = None

    @property
    def schema(self) -> dict:
        return self.tool.args_schema if isinstance(self.tool.args_schema, dict) else self.tool.args_schema.model_json_schema()


class ToolRegistry:
    """Registration is a trusted local action; MCP metadata never grants rights."""
    def __init__(self, tools: Iterable[BaseTool | ToolSpec] = (), *, runtime=None):
        self.runtime = runtime
        self._specs: dict[str, ToolSpec] = {}
        for tool in tools: self.register(tool)

    def register(self, tool: BaseTool | ToolSpec) -> None:
        spec = tool if isinstance(tool, ToolSpec) else ToolSpec(tool, permission="write" if tool.name == "create_ticket" else "read")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", spec.tool.name) or not spec.tool.description.strip():
            raise ValueError("tool needs a safe name and description")
        if spec.tool.name in self._specs: raise ValueError("tool names must be unique")
        if spec.permission not in {"read", "write", "deny"} or spec.source not in {"builtin", "mcp"}: raise ValueError("invalid local tool policy")
        schema = spec.schema
        Draft202012Validator.check_schema(schema)
        def check(value):
            if isinstance(value, dict):
                if "$ref" in value and not str(value["$ref"]).startswith("#/"): raise ValueError("external schema references are forbidden")
                if "$ref" in value:
                    target = schema
                    try:
                        for token in value["$ref"][2:].split("/"):
                            token = token.replace("~1", "/").replace("~0", "~")
                            target = target[int(token)] if isinstance(target, list) else target[token]
                    except (KeyError, IndexError, TypeError, ValueError) as error:
                        raise ValueError("unresolved local schema reference") from error
                for item in value.values(): check(item)
            elif isinstance(value, list):
                for item in value: check(item)
        check(schema)
        self._specs[spec.tool.name] = spec

    @property
    def names(self) -> set[str]: return set(self._specs)

    def spec(self, name: str) -> ToolSpec:
        try: return self._specs[name]
        except KeyError as error: raise UnknownToolError() from error

    def require(self, name: str) -> BaseTool: return self.spec(name).tool

    def safe_label(self, name: str) -> str:
        spec = self.spec(name)
        return (spec.label or _SAFE_LABELS.get(name) or name)[:80]

    @property
    def tools(self) -> list[BaseTool]: return [spec.tool for spec in self._specs.values() if spec.permission != "deny"]
