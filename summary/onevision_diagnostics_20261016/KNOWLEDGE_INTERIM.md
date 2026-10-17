# 16条知识干预：已完成结果

已完成17/17个预定干预。医学100题与KG267题重叠12题，分开报告，不合并样本量。

本组统一使用预先固定的both提示，未根据测试挑提示。无知识NLL11.169不能与原提示视觉消融的9.595混算；上下文比较只在本组同提示内进行。

| 干预 | 组 | 题数 | NLL↓ | EM匹配数 | Token F1↑ | ROUGE-L↑ |
|---|---|---:|---:|---:|---:|---:|
| context_base_none | medical | 100 | 11.169 | 0/100 | 12.445 | 12.435 |
| context_base_none | medical_KG | 267 | 9.055 | 0/267 | 7.144 | 7.014 |
| context_base_correct | medical | 100 | 2.983 | 9/100 | 16.948 | 16.948 |
| context_base_correct | medical_KG | 267 | 3.795 | 0/267 | 3.171 | 3.167 |
| context_base_unrelated | medical | 100 | 2.923 | 7/100 | 12.000 | 11.990 |
| context_base_unrelated | medical_KG | 267 | 3.828 | 1/267 | 2.461 | 2.386 |
| context_base_permuted | medical | 100 | 3.076 | 10/100 | 13.891 | 13.891 |
| context_base_permuted | medical_KG | 267 | 3.913 | 0/267 | 3.339 | 3.282 |
| context_shared_none | medical | 100 | 4.665 | 8/100 | 20.022 | 19.955 |
| context_shared_none | medical_KG | 267 | 5.294 | 0/267 | 8.978 | 8.756 |
| context_shared_correct | medical | 100 | 2.376 | 21/100 | 27.627 | 27.580 |
| context_shared_correct | medical_KG | 267 | 3.054 | 9/267 | 10.598 | 10.497 |
| context_shared_unrelated | medical | 100 | 2.376 | 21/100 | 29.591 | 29.478 |
| context_shared_unrelated | medical_KG | 267 | 3.230 | 3/267 | 4.328 | 4.193 |
| context_shared_permuted | medical | 100 | 2.452 | 22/100 | 29.071 | 29.042 |
| context_shared_permuted | medical_KG | 267 | 3.145 | 9/267 | 8.938 | 8.886 |
| residual_rank1_correct | medical | 100 | 3.925 | 9/100 | 18.841 | 18.774 |
| residual_rank1_correct | medical_KG | 267 | 4.601 | 0/267 | 7.299 | 7.168 |
| residual_rank1_unrelated | medical | 100 | 3.852 | 10/100 | 18.236 | 18.169 |
| residual_rank1_unrelated | medical_KG | 267 | 4.474 | 2/267 | 8.712 | 8.505 |
| residual_rank1_permuted | medical | 100 | 3.982 | 8/100 | 16.567 | 16.500 |
| residual_rank1_permuted | medical_KG | 267 | 4.598 | 0/267 | 7.645 | 7.493 |
| residual_rank2_correct | medical | 100 | 3.585 | 9/100 | 15.834 | 15.768 |
| residual_rank2_correct | medical_KG | 267 | 4.129 | 2/267 | 7.246 | 7.093 |
| residual_rank2_unrelated | medical | 100 | 3.495 | 10/100 | 16.501 | 16.434 |
| residual_rank2_unrelated | medical_KG | 267 | 3.954 | 2/267 | 6.252 | 6.081 |
| residual_rank2_permuted | medical | 100 | 3.620 | 9/100 | 16.209 | 16.143 |
| residual_rank2_permuted | medical_KG | 267 | 4.121 | 2/267 | 6.680 | 6.551 |
| continue_rank8_correct | medical | 100 | 2.762 | 16/100 | 22.804 | 22.804 |
| continue_rank8_correct | medical_KG | 267 | 3.131 | 3/267 | 4.055 | 4.016 |
| continue_rank8_unrelated | medical | 100 | 2.226 | 31/100 | 36.220 | 36.220 |
| continue_rank8_unrelated | medical_KG | 267 | 2.854 | 22/267 | 12.202 | 12.136 |
| continue_rank8_permuted | medical | 100 | 2.598 | 23/100 | 26.767 | 26.767 |
| continue_rank8_permuted | medical_KG | 267 | 2.942 | 4/267 | 6.760 | 6.622 |

注意：相关与低相关解剖知识都可能降低NLL；只有正确知识相对负对照的增量，才有助于定位知识内容作用。低相关组仍属医学，不能据其同样有效就断言所有专业知识均不需要。KG生成分数与NLL可能反向变化，不能以loss下降代替生成性能证据。

另有只读事实编码核验排在容量试验前：检查辅助16事实训练目标是否被记住及固定改写是否能回答。它不是额外医学测试，也不是独立事实泛化。

限制：

- Medical100 diagnostic and nativeKG267 overlap; never pool as independent367 QA.
- Unrelated is low-relevance author anatomy knowledge, not nonmedical unrelated text.
- Contexts differ in token lengths; no claim of strictly length-matched controls.
- Lower teacher-forced NLL need not improve generated medical answers.
- Only completed cases shown; no test-based prompt/capacity/hyperparameter selection.
- Knowledge encoding sanity and target-test generalization are separate; tiny-LoRA efficacy not assumed.
