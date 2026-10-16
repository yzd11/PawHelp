# PawHelp 项目结构与运行边界

PawHelp 是一个 FastAPI 主应用加若干离线命令行任务。HTTP、业务流程、数据访问和离线分类分开组织。

## 代码与数据

| 位置 | 权威职责 | 不应该放入 |
|---|---|---|
| `app/main.py` | 主应用装配、生命周期、所有页面入口 | 训练循环、第二套主题后台 |
| `app/api/` | 按聊天、知识管理、评测、运营、页面拆分的 HTTP 接口 | 业务训练、模型权重 |
| `app/workflow/` | 实时会话、理解、分流、工具、上下文 | 旁路主题分类调用 |
| `app/knowledge/` | 知识建库、检索、重排、证据判断 | 私人训练集 |
| `app/topics/` | 离线分类领域（api/data/metrics/model/pool/taxonomy） | 临时验证入口 |
| `app/database/` / `app/repositories/` | ORM、连接、数据访问 | 模型训练 |
| `scripts/` | 有显式命令的离线/运维任务 | 浏览器页面 |
| `scripts/topics/` | 分类器数据准备、训练、评测及旁路归类 CLI | 重复主应用实现 |
| `config/` | 非秘密的版本化策略和训练配置 | API Key、数据库凭据 |
| `knowledge/` / `evals/` | 演示知识、RAG 与意图评测集 | 真实客户聊天记录 |
| `data/topic-classifier/` | 本地真实问题、标注、训练数据、权重和报告 | Git 发布内容 |
| `docs/` | 最终运行、结构、模型和数据接入说明 | 临时 spec、开发验收草稿 |

主应用路由保留；`/admin` 是后台入口。客户页与后台使用独立导航，所有页面位于 `app/static/`、共享 8000。`app/api/__init__.py` 保留 `create_api_router` 导入合同；`main.py` 负责装配和生命周期。训练模块按需加载 torch/transformers，主应用不会加载主题分类器权重。

路由文件职责：`chat.py` 处理会话/订单/明确售后操作，`knowledge.py` 处理知识维护，`evaluation.py` 处理 RAG 报告和复核，`operations.py` 处理飞轮/用量与周期任务，`pages.py` 处理页面。主题领域的只读接口仍在 `app/topics/api.py`，不额外建立一套转发层。

## 进程边界

```text
浏览器 → 8000 主应用 → MySQL（业务数据与知识权威源）
                     → Milvus（检索索引）
                     → 上游模型（理解/生成/摘要等，有费用）
                     → 8101/8102 MCP（可选工具）

离线 classify_pool CLI → 8110 分类器 → topic_classifications（MySQL）
                                               ↓
浏览器 → 8000 /topics ← 只读主题分布与已持久化归类
浏览器 → 8000 /acceptance ← 本地训练/评测产物
```

8110 的隔离是功能要求：低置信度问题积攒后按批处理，实时聊天不调它。

## 数据契约

- MySQL 保存完整消息、演示订单、知识、飞轮问题和归类；Milvus 不是原文权威源。
- SQLite checkpoint / `.runtime/` 服务于本地单实例会话，不替代 MySQL 消息历史。
- `config/evidence-calibration.json`：RAG 相关性粗筛校准；之后仍需模型核查完整证据。
- `data/topic-classifier/model/threshold.json`：主题分类器的逐标签判定阈值，由验证集选出。两种阈值不能互换。
- `topic_classifications.question_id` 关联原问题池；同一问题有多个主题，各主题各计一次；主题计数总和可能超过问题数。
- 原始问题与归并知识缺口不是同一种计数；页面分别提供原问题量和独立缺口量。

## Git 与本地工作材料

私人语料、模型权重和本机数据库不进入仓库。

`.idea`、`.aris`、`.venv`、`.uv-cache*`、`.runtime`、`log`、模型、私人语料与工作树状态属于本机运行/工具环境，不作为源码发布。不能为了整理目录删除 `.runtime`、数据库卷或活动 Git worktree。`tests/` 是离线功能回归，与源码检查一起在 CI 运行。

## 品牌与兼容标识

页面、OpenAPI 标题、MCP 服务和根目录使用 PawHelp，发行包及 MySQL 业务库/账号使用 `pawhelp`。Compose 显式指定项目和数据卷名称；浏览器存储键使用 `pawhelp.*`，后台与受信 MCP 请求头使用 `X-PawHelp-*`，JS 全局接口使用 `PawHelp*`。
