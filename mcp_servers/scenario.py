"""Verify trusted demo snapshots; headers are NOT real customer authentication."""
import asyncio
import base64
import hashlib
import hmac
import json
import os
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DemoSettings(BaseSettings):
    mcp_demo_secret: SecretStr = SecretStr("local-demo-only-change-me")
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def read_scenario(ctx):
    headers = ctx.request_context.request.headers
    payload, signature = headers.get("x-pawhelp-scenario", ""), headers.get("x-pawhelp-signature", "")
    expected = hmac.new(DemoSettings().mcp_demo_secret.get_secret_value().encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not payload or len(payload) > 16000 or not hmac.compare_digest(expected, signature): raise ValueError("未提供可信演示场景")
    value = json.loads(base64.urlsafe_b64decode(payload))
    if not value.get("is_demo"): raise ValueError("仅支持演示订单")
    return value


async def delay():
    await asyncio.sleep(max(0, min(120, float(os.environ.get("MOCK_DELAY_SECONDS", "0")))))
