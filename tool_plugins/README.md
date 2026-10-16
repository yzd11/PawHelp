# 可信本地热注册

这里的 Python 文件相当于后端代码，有完整进程权限。只允许开发者放置，不能接收用户上传。
新增 `.py`，导出 `register(runtime, registry)`；只做注册即可，下轮对话生效，不重启客服。

```python
from langchain_core.tools import tool
from app.tools.registry import ToolSpec

def register(runtime, registry):
    @tool
    async def shop_hours() -> str:
        """查询演示客服值班时间。"""
        return '{"message":"演示客服：每天9至18点"}'
    registry.register(ToolSpec(shop_hours, permission="read", label="客服时间"))
```

外部工具另在 `config/tool-policy.json` 中按 `mcp:Server:工具名` 批准（仅 read/write/deny）。未知工具默认 deny，不因 Server 声明而授权。只支持 builtin create_ticket 写操作；给外部工具 write 也不会放行。
