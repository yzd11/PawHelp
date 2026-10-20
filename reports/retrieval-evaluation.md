# retrieval 四策略检索与生成质量报告

## 总体

| 策略 | Recall@5 | MRR | 证据覆盖 | 答案语义覆盖 | Faithfulness | 拒答正确率 | 未评上 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.982 | 0.861 | 0.996 | — | — | — | 0 |
| bm25 | 0.975 | 0.855 | 0.992 | — | — | — | 0 |
| hybrid | 0.986 | 0.869 | 0.998 | — | — | — | 0 |
| hybrid_rerank | 0.996 | 0.893 | 0.998 | 0.945 | 0.995 | 1.000 | 1 |

## 按 Query 类型

| 分桶 | Recall@K | MRR | Faithfulness |
|---|---:|---:|---:|
| dense:A_policy | 1.000 | 0.945 | — |
| bm25:A_policy | 0.990 | 0.957 | — |
| hybrid:A_policy | 1.000 | 0.962 | — |
| hybrid_rerank:A_policy | 1.000 | 0.958 | 0.999 |
| dense:B_model | 1.000 | 0.906 | — |
| bm25:B_model | 0.994 | 0.926 | — |
| hybrid:B_model | 1.000 | 0.927 | — |
| hybrid_rerank:B_model | 1.000 | 0.947 | 0.994 |
| dense:C_colloquial | 0.990 | 0.900 | — |
| bm25:C_colloquial | 0.970 | 0.863 | — |
| hybrid:C_colloquial | 0.980 | 0.895 | — |
| hybrid_rerank:C_colloquial | 1.000 | 0.957 | 0.987 |
| dense:D_absent | 0.000 | 0.000 | — |
| bm25:D_absent | 0.000 | 0.000 | — |
| hybrid:D_absent | 0.000 | 0.000 | — |
| hybrid_rerank:D_absent | 0.000 | 0.000 | — |
| dense:E_multi | 0.947 | 0.693 | — |
| bm25:E_multi | 0.947 | 0.664 | — |
| hybrid:E_multi | 0.962 | 0.693 | — |
| hybrid_rerank:E_multi | 0.987 | 0.720 | 0.998 |
| dense:F_identifier | 0.967 | 0.895 | — |
| bm25:F_identifier | 0.983 | 0.933 | — |
| hybrid:F_identifier | 1.000 | 0.910 | — |
| hybrid_rerank:F_identifier | 0.983 | 0.903 | 0.997 |

## 按难度

| 分桶 | Recall@K | MRR | Faithfulness |
|---|---:|---:|---:|
| dense:easy | 1.000 | 0.945 | — |
| bm25:easy | 0.990 | 0.957 | — |
| hybrid:easy | 1.000 | 0.962 | — |
| hybrid_rerank:easy | 1.000 | 0.958 | 0.999 |
| dense:medium | 0.990 | 0.902 | — |
| bm25:medium | 0.981 | 0.897 | — |
| hybrid:medium | 0.990 | 0.909 | — |
| hybrid_rerank:medium | 0.998 | 0.945 | 0.991 |
| dense:hard | 0.947 | 0.693 | — |
| bm25:hard | 0.947 | 0.664 | — |
| hybrid:hard | 0.962 | 0.693 | — |
| hybrid_rerank:hard | 0.987 | 0.720 | 0.998 |
