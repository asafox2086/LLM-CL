# 正式V2基线：先前能力、当前习得与后续保持分开

只读取同一正式V2运行已封存完整Organ507题；同ID/原金标，复算唯一Answer F1。没有新训练/推理/翻译/重评分，也不替代nativeAnyRes研究的共同S1。

## 同一测试队列的三个关键时点

S0为未微调，S3是Organ学习前（已学自然/其他医学任务），S4刚学完Organ，S19为最后阶段。待测保持空值。

| 骨干/方法 | S0英文/中文F1 | S3学习前英文/中文F1 | S4刚学完英文/中文F1 | S19英文/中文F1 |
|---|---:|---:|---:|---:|
| onevision / SeqFT_LoRA | 16.239/3.038 | 50.813/5.118 | 52.009/22.677 | 待测 |
| onevision / ER_LoRA | 16.239/3.038 | 50.479/5.118 | 51.614/22.480 | 待测 |
| onevision / CL_MoE | 16.239/3.038 | 待测 | 待测 | 待测 |
| onevision / PASs_MoE | 16.239/3.038 | 待测 | 待测 | 待测 |
| qwen / SeqFT_LoRA | 51.375/29.934 | 待测 | 待测 | 待测 |
| qwen / ER_LoRA | 51.375/29.934 | 待测 | 待测 | 待测 |

## 当前Organ阶段实际学到了多少

原TRAIN26题（中文15/英文11），有效batch16、2次实际更新、LR为1e-4/5e-5、warmup0。两方法均记录有限非零梯度、前后参数SHA变化。所有前置阶段有效性与实际日志均核验，不只信passed字段。

| 方法 | 英文S3→S4变化/95%图像区间 | 中文S3→S4变化/95%图像区间 |
|---|---:|---:|
| SeqFT_LoRA | +1.196 [-1.938,+3.846] | +17.559 [+12.494,+23.391] |
| ER_LoRA | +1.135 [-2.105,+3.904] | +17.362 [+12.254,+23.198] |

## 六栏判断

- **探究的问题：** 中文低分发生在后续跨域之前，还是已经学会后才丢失？这一现象是否只有SeqFT？
- **做的实验：** 对两机6个实际运行收集S0/S3/S4/S19已封存同队列，逐ID/参考/元数据与原F1精确核验；原始源码快照、清单、训练日志完整前缀、正LR及参数变化留SHA。全部作者语言/题型分层保留，不只报告有利子组。
- **结果：** OneVision四方法未训练输出逐题精确一致，Qwen两方法亦一致。当前可核验Organ习得配对2组、最终点0/6；每条语言变化及区间见上表，缺失不填0。
- **证明了什么：** 两个已训练基线在后续跨域前就有明显语言分组差距，但中文并非完全不能学习。S0→S4不能全部算当前任务收益，必须看S3→S4；这个实测增量仍不能区分知识、语言/格式适配。英文/中文组不是翻译配对，不能用组间分数直接测语言能力。尚无最终遗忘量，不能把nativeAnyRes跨域终点移入本表。
- **改进的方法：** 先保存学习前/刚学完的完整参照，再判断最终下降；GPU1的固定权重四指令是另一设置的独立机制控制。本审计不改变任何基线算法、训练或评测。
- **改进的效果：** 获得更明确的习得/保持分界，尚无新算法提升；ER训练loss包含回放，不能把它与SeqFT的loss大小直接比较为目标任务拟合优劣。

全部9个语言/题型分层、逐题增减数、日志及SHA见[证据JSON](VERIFIED_EVIDENCE.json)。不报告未完成的主表/FWT/BWT或全部方法优劣。

- Same frozen formal V2 runs; not nativeAnyRes/BF16 research states, never fill its shared S1.
- Pending cells remain null; a sealed task cell is not full21454 completion or a method ranking.
- S3->S4 isolates observed current-stage learning along this curriculum, not a no-update randomized intervention.
- S4->S19 combines within-medical and later cross-domain learning; cannot uniquely attribute domain count.
- Language groups are not verified translation pairs; Han counts are literal flags, not correctness.
- Paired image intervals condition on fixed single-seed fits; no multi-seed or causal knowledge sufficiency claim.
- Only raw token_f1 is reported; legacy rougeL config field does not change this locked scoring or select models.
- Qwen2B vs OneVision0.5B changes more than parameter count; not a causal size/language/knowledge isolation.
- SeqFT and ER acquisition are observational two-method checks, not all-method final superiority or significance comparison.
