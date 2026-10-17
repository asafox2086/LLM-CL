# Answer F1唯一主指标：已有诊断的重新审视

用户决定：沿用原始token_f1，改称Answer F1；不做选项映射、不加前缀去除、不用语义裁判。原预测/训练/评分文件不改。以下均重新核验原文件SHA、金标及逐题分数完全相同。

## 1. 顺序学习：质量上升和下降是否相抵？

| 域 | 固定生成QA | 刚学完累计F1 | 最终F1 | 差值pp | 质量上升/下降/持平题数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| natural | 10 | 60.714 | 57.381 | -3.333 | 0/1/9 |
| medical | 73 | 35.327 | 46.058 | +10.731 | 12/3/58 |
| geography | 32 | 59.375 | 46.875 | -12.500 | 1/5/26 |
| finance | 10 | 30.000 | 30.000 | +0.000 | 0/0/10 |
| traffic | 30 | 31.254 | 30.465 | -0.789 | 9/9/12 |

全部155题：40.798→42.903；22题上升、18题下降、115题持平。刚学完值组合不同任务各自习得时点，不是一个模型快照。

19原生任务诊断AP=43.363，FWT=15.301，BWT=1.504。都基于同一个Answer F1，仅诊断155题，不进入完整21454题主表。

**同一批输出的聚合方向可以不同：** 19任务等权F1变化+1.425pp，五域等权F1变化-1.178pp；3/5域下降。医学有10/19个原生任务，任务宏平均权重大于五域等权的1/5。因此BWT为正不能被解释成每个领域都获益。这是聚合权重的诊断，不新增主指标或根据结果改AP定义。

## 2. 记住极小知识库是否足够？

| 条件 | 医学100题：0→64步F1 | 原生KG267题：0→64步F1 |
| --- | ---: | ---: |
| rank1_correct | 20.022→25.456 | 8.978→6.494 |
| rank1_permuted | 20.022→20.854 | 8.978→3.581 |
| rank8_correct | 20.022→22.133 | 8.978→6.523 |
| rank8_permuted | 20.022→21.945 | 8.978→4.779 |

正确/错配同rank同预算的配对差值与区间见JSON。不能因某一rank或某个分组点估计高就选择它。医学100与KG267重叠12，不合并计数。事实回忆已另有干预有效性证据，但“写入事实→目标问答普遍提高”的链条仍未成立。

## 3. 遥感坍缩与部分改进在F1下还成立吗？

| 固定损失条件 | 1步F1 | 4步F1 |
| --- | ---: | ---: |
| uniform | 76.000 | 44.000 |
| class_balanced | 75.000 | 50.000 |
| urban_high_negative | 74.000 | 44.000 |

这些单词标签任务的F1数值与旧二值评分相同，因此已证明的坍缩/类别平衡局部作用并未因换主指标消失。仍不证明视觉完全失效或类别比例是唯一成因。

## 4. 完整确认仅已结束的四域

| 域 | 完整QA | Answer F1 |
| --- | ---: | ---: |
| natural | 5000 | 75.917 |
| medical | 2094 | 40.943 |
| geography | 10004 | 49.010 |
| finance | 2233 | 12.991 |

交通在跑的尾部不读、不填0；上表不计算全21454 AP/FWT/BWT。研究nativeAnyRes/统一lr1e-4与正式双骨干V2设置分别报告，不跨协议排名。

## 下一步判断

已验证：有些退化在连续F1下仍存在，极小知识写入不能保证所有目标组获益。未验证：多域数量相对少任务/预算是主要原因。既定确定性双锚六机制控制仍是核心缺口；不以本次重评分替代该GPU实验。

- Answer F1 is lexical content overlap, not full semantic correctness. No option normalization or semantic judge.
- Old EM/ROUGE files preserved for traceability, not co-primary metrics or new tuning objectives.
- Native155 cohort is diagnostic; knowledge medical100 andKG267 overlap12, never pooled as367 independent QA.
- Fact-fit64 updates512 exposures are auxiliary interventions, not formal1epoch CL.
- Existing histories and unmatched domain order do not establish pure task-count/domain-count causality.
- QA bootstrap intervals are conditional on fixed fitted weights, single seed/order, no multiple-comparison correction.
