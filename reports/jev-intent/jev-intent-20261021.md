# Jev 意图复核评测（大模型 deepseek-v4-flash，Jev jev-1.14.0）

每题大模型调用 3 次，Jev 1 次。阈值只用训练集选。

| 集合 | 次数 | 大模型意图准确率 | 大模型全字段通过 | Jev top-1 准确率 | AUC 大模型自报 | AUC Jev | 可被复核的错判 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 训练 | 489 | 0.943 | 0.912 | 0.920 | 0.868 | 0.842 | 17 |
| 验证 | 108 | 0.926 | 0.843 | 0.972 | 0.875 | 1.000 | 5 |

复核规则（意图为“其他”或分数低于阈值就请用户澄清）：

| 规则 | 分数 | 阈值 | 训练 错答/多问 | 验证 错答/多问 |
|---|---|---:|---:|---:|
| 现行 | confidence | 0.550 | 17/11 | 5/3 |
| 大模型自报重选阈值 | confidence | 0.000 | 17/11 | 5/3 |
| Jev | jev_p | 0.040 | 16/12 | 5/3 |

错答=没有澄清就按错误意图处理；多问=用户意图清楚却被要求澄清。

门槛：validation: Jev rule serves no more wrong intents and asks no more clear questions to clarify than the current 0.55 rule, and its AUC is higher than the LLM confidence's。结果：通过。

Jev 延迟 P50 257 ms，P95 972 ms；输入 134034 token，约 $0.0056。调用失败：无。
