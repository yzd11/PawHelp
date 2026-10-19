# PawHelp 运行说明

所有命令从仓库根目录执行。Python 3.12；依赖由 `pyproject.toml` 和 `uv.lock` 统一管理。不另建第二个 Python 项目。

## 一键启动

```powershell
uv run --locked python -m scripts.start
```

依次：`docker compose up -d --wait`（MySQL、etcd、MinIO、Milvus）→ `alembic upgrade head` → 物流 MCP（8101）、售后 MCP（8102）、主题分类器（8110，未训练出 `data/topic-classifier/model` 时跳过）、主应用（8000）。子进程直接用当前虚拟环境的 Python 启动，日志写入 `.runtime/logs/<服务>.log`；某个服务意外退出时打印其日志末尾并停止其余服务。端口已被占用时沿用正在运行的服务。Ctrl+C 停止本脚本启动的进程，容器继续运行（需要时 `docker compose stop`）。`--no-docker`、`--no-mcp`、`--no-classifier` 跳过对应部分。

主应用启动后在后台加载并预热 BGE-M3 与重排模型，页面先可用；`GET /health` 的 `knowledge_models` 为 `loading`、`ready` 或 `failed`。预热完成前到达的提问会等待加载完成，不会重复加载。

## 安装、迁移和主应用（分步）

```powershell
uv sync --locked --all-extras
# 仅首次：已有 .env 不要覆盖
Copy-Item .env.example .env
docker compose up -d mysql etcd minio milvus
uv run alembic upgrade head
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

已安装环境可以直接使用 `./.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000`。所有后台包括 `/topics` 和 `/acceptance` 都在 8000。

`uv.lock` 只记录 pypi.org 和 download.pytorch.org 的下载地址，本地和 CI 共用这一份，不要在 `pyproject.toml` 里加镜像 index。uv 按锁文件里的地址安装，`UV_DEFAULT_INDEX`、`UV_INDEX_URL` 或用户级 `uv.toml` 里配置的镜像（如清华源）不会加速 `uv sync`，反而会让 `--locked` 报锁文件需要更新（uv 0.11.18 实测）。在本仓库执行 uv 命令前先清掉这些变量；国内网络慢时用 `HTTPS_PROXY` 走代理。`uv lock` 同样要在没有镜像变量的环境里运行，否则锁文件会被改写成镜像地址：

```powershell
Remove-Item Env:UV_DEFAULT_INDEX, Env:UV_INDEX_URL -ErrorAction SilentlyContinue
uv sync --locked --all-extras
```

`CHAT_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY` 配置上游模型。`DATABASE_URL` 是数据库凭据，只留在本地 `.env`。`STRUCTURED_OUTPUT_METHOD` 必须由上游支持。示例数据库默认 Docker 3307，不是系统 MySQL 的默认 3306。

新安装默认业务库与账号均为 `pawhelp`。已有数据库只执行 Alembic 升级，不执行 DROP 或重新 seed。演示数据初始化命令都是显式离线操作：

```powershell
uv run python -m scripts.seed_db
uv run python -m scripts.seed_demo_orders
uv run python -m scripts.ingest_knowledge
uv run python -m scripts.vectorize_knowledge
```

固定演示订单不是实时交易；日期自然变旧，不应每天重置已有订单。

客户入口为 `/`，管理入口为 `/admin`。当前无登录、注册或角色权限，两个入口的分离仅用于区分客户和运营任务。浏览器首次打开迁移后的聊天页时，会先按本机 `.runtime/client-storage-migration.json` 迁移原 UUID、会话和反馈值，再初始化聊天；本地留有恢复副本。同源新旧身份冲突时停止初始化，不自动合并或生成新身份。`localhost`、`127.0.0.1` 和不同浏览器配置文件各自独立，须在原来源打开才能迁移；所有使用中的配置文件确认后，可归档本机策略文件，新的安装不需要该文件。更多安全限制见根目录 `SECURITY.md`。

## MCP 工具（可选）

`scripts.start` 已包含这两个服务；单独启动时分别在两个终端运行：

```powershell
uv run python -m mcp_servers.logistics_server
uv run python -m mcp_servers.aftersales_server
```

默认 8101、8102，对应 `.env.example` 的 MCP URL。它们和主服务使用同一个本地演示 secret；上线须替换可信身份与认证。无需 MCP 时在 `.env` 设置 `MCP_ENABLED=false`，重启主应用，不要删掉插件代码。

## Langfuse（可选）

参考 `.env.langfuse.example`，复制为本地 `.env.langfuse` 并配置，然后：

```powershell
docker compose --env-file .env.langfuse -f compose.langfuse.yaml up -d
```

在主应用 `.env` 中配置本地 Langfuse 地址与凭据，并设置 `LANGFUSE_ENABLED=true`。默认不连接云端。token 用量缺失、失败和已知 0 区分显示；没有配置单价时不虚构费用。

## 飞轮与评测

`FLYWHEEL_INTERVAL_SECONDS=0`、`EVALUATION_INTERVAL_SECONDS=0` 默认关闭收费周期任务。页面上的运行操作需确认，不能为了看效果自动触发模型调用。

```powershell
uv run python -m scripts.run_retrieval_evals --retrieval-only --sample-per-bucket 1
uv run python -m scripts.search_knowledge "邮费怎么算"
uv run python -m scripts.show_context_budget
uv run python -m scripts.run_intent_evals --runs 3
```

上述检索不调用回答生成/裁判；意图评测每题调用 3 次意图识别模型。完整评测、会话挖掘、问题标准化与分类器预标会调用上游模型，费用/数据发送需另行授权。

## 验证与安全

```powershell
uv run python -m scripts.check_sources
node scripts/check_frontend.cjs
```

以上只检查源码与配置，不访问数据库、不调用模型。完整开发测试已按维护者要求归档移出工作树，当前 CI 不提供功能回归覆盖；涉及业务行为的修改仍须在隔离数据上手动验证并记录结果。

本机后台控制不是公网认证。服务绑定回环地址，不把 8000/8110 直接暴露互联网。凭据与私人数据不上传 GitHub；先检查历史记录再发布。
