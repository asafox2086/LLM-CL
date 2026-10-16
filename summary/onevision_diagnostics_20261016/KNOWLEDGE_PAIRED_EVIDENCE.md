# 知识干预的配对负对照与不确定性

只读已完成逐题输出。差值=相关知识−对应控制；EM/F1越大越好，NLL越小越好。按原图像路径组配对bootstrap一万次，区间是探索性95%分位区间，未做多重比较校正。

| 相关知识组 | 控制 | 评测组 | QA/图像组 | ΔEM [95%] | ΔF1 [95%] | ΔNLL [95%] | Δ回答字符数 |
|---|---|---|---:|---:|---:|---:|---:|
| context_base_correct | context_base_none | medical | 100/83 | +9.000 [+4.000, +14.679] | +4.503 [-1.142, +10.065] | -8.186 [-8.793, -7.599] | +3.57 |
| context_base_correct | context_base_none | medical_KG | 267/103 | +0.000 [+0.000, +0.000] | -3.973 [-5.887, -2.063] | -5.260 [-5.584, -4.919] | +18.06 |
| context_base_correct | context_base_unrelated | medical | 100/83 | +2.000 [-1.923, +6.122] | +4.948 [+0.839, +9.717] | +0.060 [-0.024, +0.148] | +1.01 |
| context_base_correct | context_base_unrelated | medical_KG | 267/103 | -0.375 [-1.200, +0.000] | +0.710 [-0.675, +2.016] | -0.034 [-0.104, +0.036] | +3.08 |
| context_base_correct | context_base_permuted | medical | 100/83 | -1.000 [-3.093, +0.000] | +3.057 [+0.873, +5.606] | -0.093 [-0.150, -0.046] | +6.55 |
| context_base_correct | context_base_permuted | medical_KG | 267/103 | +0.000 [+0.000, +0.000] | -0.168 [-0.744, +0.349] | -0.119 [-0.149, -0.089] | +7.45 |
| context_shared_correct | context_shared_none | medical | 100/83 | +13.000 [+6.250, +20.000] | +7.605 [-0.614, +15.854] | -2.290 [-2.729, -1.872] | -4.15 |
| context_shared_correct | context_shared_none | medical_KG | 267/103 | +3.371 [+1.493, +5.594] | +1.620 [-1.464, +4.812] | -2.240 [-2.458, -2.020] | +3.12 |
| context_shared_correct | context_shared_unrelated | medical | 100/83 | +0.000 [-5.825, +5.607] | -1.965 [-7.897, +3.728] | +0.000 [-0.085, +0.091] | -6.04 |
| context_shared_correct | context_shared_unrelated | medical_KG | 267/103 | +2.247 [+0.000, +4.622] | +6.269 [+3.506, +9.106] | -0.175 [-0.232, -0.117] | -12.60 |

相关知识优于“无上下文”不能单独证明事实正确性的作用；还需同时检查低相关与错配控制。同一图像多题不是独立样本，医学100与KG267亦不能合并为367题。区间不能替代更多种子／独立资料验证。

限制：

- Exploratory single-seed diagnostic intervals, no multiple-comparison correction or population-wide claim.
- Image paths define clusters; distinct files may still depict related patients/scenes.
- Medical100 and KG267 overlap12 QA; groups never pooled.
- Only completed negative controls admitted; no test-guided changes to prompts, training or capacity.
- A positive generated-score difference is needed alongside content-specific controls; NLL alone is insufficient.
- Prediction length in Unicode characters, not tokens; heterogeneous language and answer types remain.
