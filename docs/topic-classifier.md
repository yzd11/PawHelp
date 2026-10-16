# PawHelp 离线主题分类器

## 固定方案

`hfl/chinese-roberta-wwm-ext` 全参微调，17 类独立多标签，结果写入 `topic_classifications`。实时聊天不调用分类器。业务代码集中在 `app/topics/`，CLI 为 `scripts.topics.*`，本地数据目录为 `data/topic-classifier/`。目录更名不改变原始权重、冻结数据或模型版本。

术语权威表是 `app/topics/taxonomy.py`。四个领头类目是退换货、物流、尺码、发票；全部 17 类及边界在术语表中。维修与退货、运费与物流、价保与优惠不能靠关联词脑补。字面有效诉求全部标记，「其他」不与其他标签同时出现。

## 数据处理

真实问题池 → 脱敏/格式清洗 → 术语表预标与抽审 → 已知来源/近义组隔离 → 80/10/10 分层目标 → 只增强训练集。比例是整组分配目标，不保证拆开重复样本后精确凑比例。

各 CLI 的 `--help` 提供参数：`corpus`、`generate`、`review`、`near_groups`、`dataset`。其中预标/模拟生成会调用配置的大模型，不因为 README 提供命令就自动获得发送真实数据或追加费用的授权。原候选数据、修正意见和冻结测试集分开保存。

## 现有本地训练结果

- 模型：`63b4e0fccf53ac730747`；CUDA 全参训练，7 轮早停，加载第 5 轮最佳 checkpoint。
- 数据：训练 2750、验证 166、测试 167；训练增强前 1376。验证全部模拟，测试 166 模拟 + 1 真实。
- 阈值：验证集扫描 0.30～0.70、步长 0.05，选出 0.50；不是参考截图的 0.45。
- 测试 micro-F1 0.9527 / macro-F1 0.9579；17 类 P/R/F1 和各类 2×2 矩阵在本地报告。
- 短句「买大了想退」仍只命中退换货，尺码分数约 0.1603。10 条错例待复核，不声称全部验收通过，不降低阈值伪造通过。
- 实际批量归类已保存 7 个问题：商品信息 6、其他 1；另 2 个问题未归并，不进入本批。该快照不是所有部署都会有的初始数据。

此分数不代表生产真实问题准确率。权重、训练集和私人错例默认不进入 Git，克隆后缺少产物是正常状态。

## 后台页面

统一从主后台访问，不另起前端服务：

- `/topics`：问题记录、全体去重的独立问题、有记录主题和待归类数量；支持主题搜索、按记录/独立问题排序。默认隐藏零记录主题，可勾选显示全部 17 类。展开最多显示 3 条问法并进入明细。
- `/topics/questions`：关键词、来源、知识审核状态组合筛选；先筛选全部类目记录，再每页 10 条分页，翻页保留条件。展示归并问法，原话折叠查看；知识审核不代表分类标签已人工确认。归并频次不重复累加到问题记录数。
- `/acceptance`：分类服务、待处理数量、质量关注及最近批次，直达主题结果与知识审核。分开显示开始/完成时间，未结束日志保留结果未确认提示；运行诊断折叠，不将文件存在视为验收通过。
- `/acceptance/eval`：micro/macro P、R、F1 和每类验收结果；17 个 2×2 矩阵与已有验证集阈值扫描折叠查看。横条统一使用 0～1 刻度，不截断坐标放大差异。
- `/acceptance/data`：次级“技术资料”，按数据、划分、模型、训练、环境和文件分组折叠。文件清单只读取固定路径，不加载权重。
- `/acceptance/errors`：历史错例的关键词及漏标/多标/错位筛选、标签差异、相关分数和分页。不自动改标准答案或回流训练。

每类 F1 验收标准由维护者确认，配置在 `config/topic-acceptance.json`：严格档（前 5 类）≥0.90；标准档（运费至保修维修，共 8 类）≥0.80；观察档（账号、会员积分、评价、其他）不设线，不计为通过。support 为 0 的类目不可判为达标。此标准与模型分类阈值 0.50 无关，不是生产上线认证。

评测报告按模型版本、分类阈值、类目版本、训练时数据集清单摘要及测试集摘要核对；不匹配时不用于当前验收。现有错例与阈值扫描文件没有独立模型版本凭证：错例标记“来源待核对”，即使汇总数量一致也不宣称来源已认证；扫描只显示已有记录与当前配置的数值对照。后续如升级产物格式，应在生成阶段补充版本与内容摘要，不能靠展示层补造。

页面为只读，不启动训练、调用付费模型、覆盖报告或写数据库。批量归类由维护者按下方运行说明手动执行，不放在日常页面中。当前为 Transformers/PyTorch 推理，未实现 ONNX 导出与一致性校验；页面不会展示不存在的通过状态。缺少私有产物时显示空状态，损坏报告单独显示无效状态。

## 训练（不要覆盖已有权重）

训练依赖用 `uv sync --locked --extra classifier` 安装。环境声明在 `config/topic-training.json`，不依赖 `.scratch`。本机已完成训练，无须为目录整理重训。

如果准备重新训练，先准备审定并冻结的新数据、检查 GPU 和成本，使用全新的输出目录：

```powershell
. .\scripts\topics\utf8.ps1
.\.venv\Scripts\python.exe -m scripts.topics.runtime preflight --output data/topic-classifier/reports/environment-witness-new.json
# 仅首次、且 data/topic-classifier/base 尚未存在时：下载官方基座，不是微调。
.\.venv\Scripts\python.exe -m scripts.topics.runtime stage-base --output data/topic-classifier/base
.\.venv\Scripts\python.exe -m scripts.topics.runtime train --dataset data/topic-classifier/dataset --output data/topic-classifier/model-new --epochs 8 --batch 2
```

微批 2、梯度累积 8、FP16、梯度检查点、max_length 128、lr 2e-5、weight decay 0.01、早停 patience 2。验证集选模型和阈值，不用测试集调参。脚本拒绝覆盖非空训练目录；失败先诊断，不删除已有 checkpoint 或擅自改方案。

新克隆不包含基座权重，`train` 会读取 `data/topic-classifier/base/base-source.json`，所以先按上面的 `stage-base` 下载。已有基座目录时跳过这一步；脚本拒绝覆写非空目录。下载和初始化需要网络与已批准的 CUDA 环境，不调用 DeepSeek，也不代表已完成微调。

## 评测和推理服务

```powershell
. .\scripts\topics\utf8.ps1
.\.venv\Scripts\python.exe -m scripts.topics.runtime evaluate --dataset data/topic-classifier/dataset --model data/topic-classifier/model --output data/topic-classifier/reports
.\.venv\Scripts\python.exe -m uvicorn scripts.topics.serve:app --host 127.0.0.1 --port 8110
```

评测写 `eval-report.json`、`eval-report.md` 和 `error-samples.json`；再次评测会刷新这三个产物，先保存已做的人工错例复核记录。推理默认 CPU，启动时加载权重；无权重时启动失败，不回退成规则伪模型。可用 `TOPIC_MODEL_DIR` 指定另一份模型路径，但后台验收仍读取标准目录，应同步部署产物再比较。

## 旁路批量归类

另开终端，在项目根目录运行：

```powershell
Invoke-RestMethod http://127.0.0.1:8110/healthz
.\.venv\Scripts\python.exe -m scripts.topics.classify_pool --force
```

分类器服务和 MySQL 需就绪。默认攒满 10 条已归并问题；`--force` 仅允许不足一批时执行，不覆盖已归类结果。重复执行无待处理问题时幂等空跑。归类状态/失败报告放在本地 `data/topic-classifier/reports/batches/`。

后台使用主服务：`http://127.0.0.1:8000/topics`、`/acceptance/eval`、`/acceptance/errors`。8110 只有推理接口，不提供后台页面。
