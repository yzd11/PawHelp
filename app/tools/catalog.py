"""Live tool snapshots: local registration + per-server MCP discovery."""
import asyncio
import base64
import hashlib
import hmac
import importlib.util
import json
import logging
from pathlib import Path

import httpx
from mcp.types import CallToolResult, TextContent
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.repositories.orders import OrderNotFound
from app.tools.business import build_business_tools
from app.tools.executor import TransientToolError
from app.tools.registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


class ToolCatalog:
    """A snapshot is fixed for a turn; discovery and policy refresh next turn."""
    def __init__(self, settings, *, builder=build_business_tools):
        self.settings, self.builder = settings, builder
        self.plugins = {}

    async def snapshot(self, runtime) -> ToolRegistry:
        registry = ToolRegistry(runtime=runtime)
        for tool in self.builder(runtime):
            # The public compatibility builder is not active in the conversation workflow.
            if tool.name not in {"query_logistics", "query_faq"}: registry.register(tool)
        plugin_dir = ROOT / self.settings.tool_plugins_path
        for path in sorted(plugin_dir.glob("*.py")):
            if path.name.startswith("_"): continue
            try:
                stamp = path.stat().st_mtime_ns
                cached = self.plugins.get(path)
                if cached is None or cached[0] != stamp:
                    definition = importlib.util.spec_from_file_location("pawhelp_plugin_" + path.stem, path)
                    module = importlib.util.module_from_spec(definition)
                    definition.loader.exec_module(module)
                    self.plugins[path] = (stamp, module)
                self.plugins[path][1].register(runtime, registry)
            except Exception: logger.exception("local tool plugin rejected path=%s", path.name)
        if not self.settings.mcp_enabled: return registry
        try:
            policy = json.loads((ROOT / self.settings.tool_policy_path).read_text(encoding="utf-8"))
            if not isinstance(policy, dict): raise ValueError("policy must be object")
        except (ValueError, OSError):
            logger.exception("tool policy unavailable; MCP tools denied")
            policy = {}

        async def inject_scenario(request, handler):
            # Private server headers are built from owned DB snapshots, never model arguments.
            args = request.args
            orders = runtime.order_data
            try:
                if request.server_name == "logistics":
                    scenario = await orders.get_logistics(runtime.user_id, tracking_no=args.get("tracking_no"), order_id=args.get("order_id"))
                else:
                    scenario = await orders.get_order(runtime.user_id, str(args.get("order_id", "")))
                if not scenario.get("is_demo", False):
                    raise ValueError("MCP demo servers cannot process real purchases")
            except OrderNotFound:
                return CallToolResult(content=[TextContent(type="text", text='{"code":"order_not_found","error":"没有找到您的这笔订单或运单"}')])
            payload = base64.urlsafe_b64encode(json.dumps(scenario, ensure_ascii=False, separators=(",", ":")).encode()).decode()
            signature = hmac.new(self.settings.mcp_demo_secret.get_secret_value().encode(), payload.encode(), hashlib.sha256).hexdigest()
            headers = {"x-pawhelp-scenario": payload, "x-pawhelp-signature": signature}
            try: return await handler(request.override(headers=headers))
            except Exception as exc:
                def transport_failure(error):
                    if isinstance(error, BaseExceptionGroup): return bool(error.exceptions) and all(transport_failure(e) for e in error.exceptions)
                    return isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError))
                if transport_failure(exc): raise TransientToolError() from exc
                raise

        client = MultiServerMCPClient({
            "logistics": {"url": self.settings.mcp_logistics_url, "transport": "streamable_http"},
            "aftersales": {"url": self.settings.mcp_aftersales_url, "transport": "streamable_http"},
        }, tool_interceptors=[inject_scenario])

        async def discover(server):
            try:
                async with asyncio.timeout(self.settings.mcp_discovery_timeout_seconds): return await client.get_tools(server_name=server)
            except Exception:
                logger.warning("MCP discovery unavailable server=%s; other tools remain usable", server, exc_info=True)
                return []
        servers = ("logistics", "aftersales")
        candidates = [ToolSpec(tool, "mcp", server, policy.get(f"mcp:{server}:{tool.name}", "deny"))
                      for server, tools in zip(servers, await asyncio.gather(*(discover(s) for s in servers))) for tool in tools]
        # An unknown denied name cannot shadow another Server's locally approved tool.
        for spec in sorted(candidates, key=lambda candidate: candidate.permission == "deny"):
            try:
                registry.register(spec)
            except Exception:
                logger.warning("MCP tool rejected server=%s name=%s", spec.server, spec.tool.name, exc_info=True)
        logger.info("tool_catalog conversation=%s tools=%s", runtime.conversation_id,
                    [(n, registry.spec(n).source, registry.spec(n).server, registry.spec(n).permission) for n in sorted(registry.names)])
        return registry
