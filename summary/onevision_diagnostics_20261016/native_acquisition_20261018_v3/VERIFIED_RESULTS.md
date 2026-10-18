# 原生任务习得：共同完成前缀

8个当前run中7个已完成预定Organ4锚，共同比较T01—T13；其余未到锚点的run待测。只读原输出、完整金标/F1及真实更新证明。

学习增量=刚学完当前任务F1−学习前F1，不是最终AP。一次／多次更新分组由固定batch16决定，不按分数选阈值。

| 骨干／方法 | 一次更新任务平均学习增量 | 多次更新任务平均学习增量 |
|---|---:|---:|
| onevision／SeqFT_LoRA | +3.597 | +17.569 |
| onevision／ER_LoRA | +3.425 | +16.903 |
| onevision／CL_MoE | +5.169 | +14.937 |
| onevision／PASs_MoE | +10.087 | +31.513 |
| onevision／PaRSP | +1.621 | +13.631 |
| onevision／OSFT | +10.743 | +21.852 |
| qwen／SeqFT_LoRA | +3.036 | +11.314 |

## 共同任务的逐项学习增量

| 任务／TRAIN／更新 | onevision／SeqFT_LoRA | onevision／ER_LoRA | onevision／CL_MoE | onevision／PASs_MoE | onevision／PaRSP | onevision／OSFT | qwen／SeqFT_LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|
| T01 None／347／22 | +36.482 | +36.560 | +36.253 | +34.703 | +35.215 | +36.646 | +8.070 |
| T02 Modality／11／1 | +3.460 | +3.548 | +21.242 | +10.766 | +8.014 | +17.972 | +6.092 |
| T03 Position／17／2 | +12.016 | +10.619 | +2.636 | +23.098 | +1.585 | -3.948 | +7.964 |
| T04 Organ／26／2 | +9.394 | +9.265 | +2.347 | +21.652 | +2.531 | +19.648 | +6.514 |
| T05 Size／6／1 | +0.782 | +3.422 | -6.621 | -11.559 | +1.344 | +10.081 | +11.632 |
| T06 Abnormality／14／1 | +0.805 | +2.262 | +5.923 | +9.969 | +0.315 | +9.184 | +2.488 |
| T07 Quantity／4／1 | +18.727 | +16.479 | +0.000 | +12.360 | +2.247 | +1.124 | -2.247 |
| T08 Plane／5／1 | +1.580 | -1.646 | +0.718 | +12.486 | +1.897 | +21.552 | +1.623 |
| T09 Shape／1／1 | +0.000 | -1.667 | +0.000 | +18.914 | +8.333 | +8.333 | +0.000 |
| T10 Color／3／1 | +0.923 | -0.154 | +10.769 | +37.538 | +6.154 | +33.385 | +7.692 |
| T11 KG／12／1 | +1.091 | +1.580 | +5.491 | +13.309 | +1.288 | +10.056 | +0.041 |
| T12 rural_urban／6／1 | +5.000 | +7.000 | +9.000 | -13.000 | -15.000 | -15.000 | +0.000 |
| T13 presence／170／11 | +12.386 | +11.168 | +18.511 | +46.599 | +15.195 | +35.059 | +22.707 |

完整初始/学习前/刚学完/最终、作者题型、实际loss/梯度/LR和SHA见[EVIDENCE](EVIDENCE.json)。主表不填本诊断分数。

- Delta acquisition is a diagnostic, not final AP/FWT/BWT or a new algorithm.
- One-update boundary is fixed batch16; tasks also differ in content/difficulty/domain. No data-size/step causal claim.
- Methods retain authored capacities/LR/auxiliaries; these are not matched-capacity mechanism interventions.
- Common prefix may omit finance/traffic; do not call it a completed five-domain comparison.
- Pending fresh Qwen ER excluded from comparison only because no training complete; no old-run splice.
- Formal common401408 FP16 is separate from nativeAnyRes research; no TEST tuning.
- Malformed resource records preserved; affected run excluded from resource comparison, never repaired or spliced across PIDs. Primary score/update gates unchanged.
