# retrieval 四策略检索与生成质量报告

## 总体

| 策略 | Recall@5 | MRR | 证据覆盖 | 答案语义覆盖 | Faithfulness | 拒答正确率 | 未评上 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.980 | 0.859 | 0.998 | — | — | — | 0 |
| bm25 | 0.976 | 0.842 | 0.990 | — | — | — | 0 |
| hybrid | 0.983 | 0.866 | 0.998 | — | — | — | 0 |
| hybrid_rerank | 0.997 | 0.900 | 0.998 | 0.945 | 0.995 | 1.000 | 0 |

## 按 Query 类型

| 分桶 | Recall@K | MRR | Faithfulness |
|---|---:|---:|---:|
| dense:A_policy | 1.000 | 0.945 | — |
| bm25:A_policy | 0.990 | 0.935 | — |
| hybrid:A_policy | 1.000 | 0.953 | — |
| hybrid_rerank:A_policy | 1.000 | 0.973 | 0.995 |
| dense:B_model | 0.988 | 0.888 | — |
| bm25:B_model | 1.000 | 0.928 | — |
| hybrid:B_model | 1.000 | 0.912 | — |
| hybrid_rerank:B_model | 1.000 | 0.953 | 0.990 |
| dense:C_colloquial | 0.990 | 0.903 | — |
| bm25:C_colloquial | 0.970 | 0.842 | — |
| hybrid:C_colloquial | 0.980 | 0.905 | — |
| hybrid_rerank:C_colloquial | 1.000 | 0.956 | 0.995 |
| dense:D_absent | 0.000 | 0.000 | — |
| bm25:D_absent | 0.000 | 0.000 | — |
| hybrid:D_absent | 0.000 | 0.000 | — |
| hybrid_rerank:D_absent | 0.000 | 0.000 | — |
| dense:E_multi | 0.947 | 0.694 | — |
| bm25:E_multi | 0.947 | 0.652 | — |
| hybrid:E_multi | 0.952 | 0.693 | — |
| hybrid_rerank:E_multi | 0.992 | 0.728 | 1.000 |
| dense:F_identifier | 0.967 | 0.897 | — |
| bm25:F_identifier | 0.983 | 0.930 | — |
| hybrid:F_identifier | 1.000 | 0.906 | — |
| hybrid_rerank:F_identifier | 0.983 | 0.904 | 0.998 |

## 按难度

| 分桶 | Recall@K | MRR | Faithfulness |
|---|---:|---:|---:|
| dense:easy | 1.000 | 0.945 | — |
| bm25:easy | 0.990 | 0.935 | — |
| hybrid:easy | 1.000 | 0.953 | — |
| hybrid_rerank:easy | 1.000 | 0.973 | 0.995 |
| dense:medium | 0.986 | 0.897 | — |
| bm25:medium | 0.983 | 0.887 | — |
| hybrid:medium | 0.990 | 0.908 | — |
| hybrid_rerank:medium | 0.998 | 0.947 | 0.993 |
| dense:hard | 0.947 | 0.694 | — |
| bm25:hard | 0.947 | 0.652 | — |
| hybrid:hard | 0.952 | 0.693 | — |
| hybrid_rerank:hard | 0.992 | 0.728 | 1.000 |
