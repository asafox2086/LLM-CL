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
| context_shared_correct | context_shared_permuted | medical | 100/83 | -1.000 [-4.630, +2.041] | -1.444 [-5.500, +2.315] | -0.077 [-0.106, -0.049] | -1.70 |
| context_shared_correct | context_shared_permuted | medical_KG | 267/103 | +0.000 [+0.000, +0.000] | +1.660 [+0.170, +3.243] | -0.091 [-0.114, -0.067] | -5.47 |
| residual_rank1_correct | residual_rank1_unrelated | medical | 100/83 | -1.000 [-3.093, +0.000] | +0.605 [-2.324, +3.329] | +0.073 [+0.033, +0.119] | -0.43 |
| residual_rank1_correct | residual_rank1_unrelated | medical_KG | 267/103 | -0.749 [-1.901, +0.000] | -1.413 [-3.033, +0.068] | +0.127 [+0.100, +0.153] | -0.92 |
| residual_rank1_correct | residual_rank1_permuted | medical | 100/83 | +1.000 [+0.000, +3.158] | +2.274 [+0.000, +5.262] | -0.057 [-0.083, -0.033] | -0.82 |
| residual_rank1_correct | residual_rank1_permuted | medical_KG | 267/103 | +0.000 [+0.000, +0.000] | -0.346 [-1.128, +0.289] | +0.004 [-0.013, +0.020] | +0.78 |
| residual_rank2_correct | residual_rank2_unrelated | medical | 100/83 | -1.000 [-3.158, +0.000] | -0.667 [-3.486, +1.494] | +0.090 [+0.045, +0.138] | +1.03 |
| residual_rank2_correct | residual_rank2_unrelated | medical_KG | 267/103 | +0.000 [+0.000, +0.000] | +0.994 [-0.115, +2.291] | +0.175 [+0.141, +0.208] | -0.45 |
| residual_rank2_correct | residual_rank2_permuted | medical | 100/83 | +0.000 [+0.000, +0.000] | -0.375 [-1.184, +0.000] | -0.035 [-0.054, -0.015] | -0.20 |
| residual_rank2_correct | residual_rank2_permuted | medical_KG | 267/103 | +0.000 [+0.000, +0.000] | +0.566 [-0.255, +1.541] | +0.008 [-0.013, +0.030] | +0.09 |
| continue_rank8_correct | continue_rank8_unrelated | medical | 100/83 | -15.000 [-22.917, -7.368] | -13.417 [-22.145, -4.835] | +0.537 [+0.357, +0.726] | +0.68 |
| continue_rank8_correct | continue_rank8_unrelated | medical_KG | 267/103 | -7.116 [-10.425, -3.968] | -8.147 [-11.364, -4.924] | +0.277 [+0.161, +0.389] | +6.82 |
| continue_rank8_correct | continue_rank8_permuted | medical | 100/83 | -7.000 [-12.621, -2.105] | -3.963 [-9.849, +1.275] | +0.165 [+0.101, +0.230] | +0.52 |
| continue_rank8_correct | continue_rank8_permuted | medical_KG | 267/103 | -0.375 [-1.667, +0.787] | -2.705 [-4.521, -0.923] | +0.190 [+0.155, +0.222] | -0.17 |
| residual_rank1_correct | context_shared_none | medical | 100/83 | +1.000 [+0.000, +3.158] | -1.182 [-4.033, +2.034] | -0.740 [-0.913, -0.581] | -1.24 |
| residual_rank1_correct | context_shared_none | medical_KG | 267/103 | +0.000 [+0.000, +0.000] | -1.679 [-2.762, -0.734] | -0.693 [-0.761, -0.624] | +1.19 |
| residual_rank1_unrelated | context_shared_none | medical | 100/83 | +2.000 [+0.000, +5.051] | -1.786 [-5.373, +2.075] | -0.814 [-1.016, -0.626] | -0.81 |
| residual_rank1_unrelated | context_shared_none | medical_KG | 267/103 | +0.749 [+0.000, +1.901] | -0.266 [-1.846, +1.369] | -0.820 [-0.905, -0.732] | +2.11 |
| residual_rank1_permuted | context_shared_none | medical | 100/83 | +0.000 [+0.000, +0.000] | -3.455 [-5.947, -1.304] | -0.684 [-0.850, -0.529] | -0.42 |
| residual_rank1_permuted | context_shared_none | medical_KG | 267/103 | +0.000 [+0.000, +0.000] | -1.332 [-2.288, -0.481] | -0.696 [-0.767, -0.624] | +0.41 |
| residual_rank2_correct | context_shared_none | medical | 100/83 | +1.000 [-2.105, +4.465] | -4.188 [-8.769, +0.362] | -1.080 [-1.334, -0.845] | -0.89 |
| residual_rank2_correct | context_shared_none | medical_KG | 267/103 | +0.749 [+0.000, +1.901] | -1.732 [-3.443, +0.011] | -1.165 [-1.283, -1.044] | +2.39 |
| residual_rank2_unrelated | context_shared_none | medical | 100/83 | +2.000 [-1.961, +6.122] | -3.521 [-8.628, +1.809] | -1.170 [-1.454, -0.909] | -1.92 |
| residual_rank2_unrelated | context_shared_none | medical_KG | 267/103 | +0.749 [+0.000, +1.901] | -2.726 [-4.645, -0.833] | -1.340 [-1.474, -1.202] | +2.84 |
| residual_rank2_permuted | context_shared_none | medical | 100/83 | +1.000 [-2.105, +4.465] | -3.813 [-8.314, +0.710] | -1.045 [-1.305, -0.803] | -0.69 |
| residual_rank2_permuted | context_shared_none | medical_KG | 267/103 | +0.749 [+0.000, +1.901] | -2.298 [-4.020, -0.584] | -1.173 [-1.289, -1.055] | +2.30 |
| continue_rank8_correct | context_shared_none | medical | 100/83 | +8.000 [+1.923, +14.706] | +2.782 [-4.728, +10.575] | -1.903 [-2.309, -1.518] | -5.47 |
| continue_rank8_correct | context_shared_none | medical_KG | 267/103 | +1.124 [+0.000, +2.456] | -4.923 [-7.039, -2.814] | -2.162 [-2.430, -1.903] | -1.04 |
| continue_rank8_unrelated | context_shared_none | medical | 100/83 | +23.000 [+13.725, +32.654] | +16.198 [+6.929, +25.828] | -2.440 [-2.969, -1.934] | -6.15 |
| continue_rank8_unrelated | context_shared_none | medical_KG | 267/103 | +8.240 [+5.036, +11.554] | +3.224 [+0.043, +6.440] | -2.440 [-2.743, -2.139] | -7.86 |
| continue_rank8_permuted | context_shared_none | medical | 100/83 | +15.000 [+7.767, +22.917] | +6.745 [-1.100, +15.085] | -2.068 [-2.516, -1.652] | -5.99 |
| continue_rank8_permuted | context_shared_none | medical_KG | 267/103 | +1.498 [+0.361, +3.000] | -2.218 [-4.694, +0.202] | -2.352 [-2.637, -2.071] | -0.87 |

相关知识优于“无上下文”不能单独证明事实正确性的作用；还需同时检查低相关与错配控制。同一图像多题不是独立样本，医学100与KG267亦不能合并为367题。区间不能替代更多种子／独立资料验证。

限制：

- Exploratory single-seed diagnostic intervals, no multiple-comparison correction or population-wide claim.
- Image paths define clusters; distinct files may still depict related patients/scenes.
- Medical100 and KG267 overlap12 QA; groups never pooled.
- Only completed negative controls admitted; no test-guided changes to prompts, training or capacity.
- A positive generated-score difference is needed alongside content-specific controls; NLL alone is insufficient.
- Prediction length in Unicode characters, not tokens; heterogeneous language and answer types remain.
