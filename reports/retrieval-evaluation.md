# retrieval 四策略检索与生成质量报告

## 总体

| 策略 | Recall@5 | MRR | 证据覆盖 | 答案语义覆盖 | Faithfulness | 拒答正确率 | 未评上 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.976 | 0.859 | 0.993 | — | — | — | 0 |
| bm25 | 0.978 | 0.837 | 0.993 | — | — | — | 0 |
| hybrid | 0.985 | 0.870 | 0.998 | — | — | — | 0 |
| hybrid_rerank | 0.993 | 0.894 | 0.993 | 0.949 | 0.991 | 1.000 | 1 |

## 按 Query 类型

| 分桶 | Recall@K | MRR | Faithfulness |
|---|---:|---:|---:|
| dense:A_policy | 1.000 | 0.951 | — |
| bm25:A_policy | 1.000 | 0.943 | — |
| hybrid:A_policy | 1.000 | 0.960 | — |
| hybrid_rerank:A_policy | 1.000 | 0.963 | 0.992 |
| dense:B_model | 0.988 | 0.893 | — |
| bm25:B_model | 0.988 | 0.913 | — |
| hybrid:B_model | 1.000 | 0.927 | — |
| hybrid_rerank:B_model | 1.000 | 0.952 | 0.999 |
| dense:C_colloquial | 0.970 | 0.893 | — |
| bm25:C_colloquial | 0.980 | 0.809 | — |
| hybrid:C_colloquial | 0.980 | 0.888 | — |
| hybrid_rerank:C_colloquial | 0.990 | 0.950 | 0.981 |
| dense:D_absent | 0.000 | 0.000 | — |
| bm25:D_absent | 0.000 | 0.000 | — |
| hybrid:D_absent | 0.000 | 0.000 | — |
| hybrid_rerank:D_absent | 0.000 | 0.000 | — |
| dense:E_multi | 0.950 | 0.694 | — |
| bm25:E_multi | 0.947 | 0.668 | — |
| hybrid:E_multi | 0.957 | 0.703 | — |
| hybrid_rerank:E_multi | 0.992 | 0.721 | 0.990 |
| dense:F_identifier | 0.967 | 0.897 | — |
| bm25:F_identifier | 0.983 | 0.929 | — |
| hybrid:F_identifier | 1.000 | 0.923 | — |
| hybrid_rerank:F_identifier | 0.967 | 0.899 | 0.997 |

## 按难度

| 分桶 | Recall@K | MRR | Faithfulness |
|---|---:|---:|---:|
| dense:easy | 1.000 | 0.951 | — |
| bm25:easy | 1.000 | 0.943 | — |
| hybrid:easy | 1.000 | 0.960 | — |
| hybrid_rerank:easy | 1.000 | 0.963 | 0.992 |
| dense:medium | 0.976 | 0.894 | — |
| bm25:medium | 0.983 | 0.866 | — |
| hybrid:medium | 0.990 | 0.908 | — |
| hybrid_rerank:medium | 0.990 | 0.943 | 0.990 |
| dense:hard | 0.950 | 0.694 | — |
| bm25:hard | 0.947 | 0.668 | — |
| hybrid:hard | 0.957 | 0.703 | — |
| hybrid_rerank:hard | 0.992 | 0.721 | 0.990 |
