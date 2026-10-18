# 新封存V2基线：习得与保持分开

原只读核验器验证完整507题、原金标/元数据/Answer F1、每个前置阶段实际正LR/有限梯度/参数SHA改变及冻结源码；没有新GPU训练/推理。新Qwen ER从头运行，旧中断run不拼入。

本次接受17个完整格，其中8个为新增；当前任务习得配对5组、S4→S19保持配对2组。

| 骨干／方法 | S0英文／中文 | S3学习前英文／中文 | S4刚学完英文／中文 | S19最终英文／中文 |
|---|---:|---:|---:|---:|
| onevision／SeqFT_LoRA | 16.239/3.038 | 50.813/5.118 | 52.009/22.677 | 52.859/22.091 |
| onevision／ER_LoRA | 16.239/3.038 | 50.479/5.118 | 51.614/22.480 | 46.047/25.709 |
| onevision／CL_MoE | 16.239/3.038 | 49.506/23.045 | 49.177/28.058 | 待测 |
| onevision／PASs_MoE | 16.239/3.038 | 51.186/17.283 | 45.323/66.342 | 待测 |
| qwen／SeqFT_LoRA | 51.375/29.934 | 52.394/25.051 | 55.312/35.147 | 待测 |
| qwen／ER_LoRA | 待测 | 待测 | 待测 | 待测 |

## S3→S4：当前任务习得

| 骨干／方法 | 英文变化及95%图像区间 | 中文变化及95%图像区间 |
|---|---:|---:|
| onevision／SeqFT_LoRA | +1.196 [-1.938,+3.846] | +17.559 [+12.494,+23.391] |
| onevision／ER_LoRA | +1.135 [-2.105,+3.904] | +17.362 [+12.254,+23.198] |
| onevision／CL_MoE | -0.329 [-6.531,+5.208] | +5.013 [+2.151,+8.358] |
| onevision／PASs_MoE | -5.863 [-9.712,-2.273] | +49.059 [+41.770,+55.517] |
| qwen／SeqFT_LoRA | +2.918 [+0.128,+5.967] | +10.096 [+7.020,+13.319] |

## S4→S19：后续净变化

| 骨干／方法 | 英文变化及95%图像区间 | 中文变化及95%图像区间 |
|---|---:|---:|
| onevision／SeqFT_LoRA | +0.850 [-5.676,+7.260] | -0.586 [-4.117,+2.453] |
| onevision／ER_LoRA | -5.567 [-13.443,+1.681] | +3.228 [-0.172,+6.748] |

原TRAIN26题（中文15/英文11）；完整逐题增减数、全部语言/题型分层、实际loss/梯度/LR/阶段证明与SHA见[JSON](VERIFIED_EVIDENCE.json)。

- S3->S4 is an observed stage transition, not a randomized no-update intervention.
- S4->S19 combines later medical and cross-domain tasks, not pure domain-count causality.
- Language groups are author annotations, not verified translation pairs.
- Single seed; image intervals condition on fits, not seed/order uncertainty.
- Qwen vs OneVision changes architecture/pretraining as well as size; not knowledge/size isolation.
- New Qwen ER pending; never splice the interrupted run or assume its scores.
- Formal common401408 FP16 only; never fill nativeAnyRes research anchors with these cells.
- ER loss contains replay; raw loss levels across methods not target-only fit comparisons.
- Organ507 cell alone not full21454 or all-method ranking; full completion verified separately.
