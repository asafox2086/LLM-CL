# 固定权重：语言/简答指令干预

完整运行证明：0/4。缺失为待测，不是零分。

| 范围 | 权重 | 指令视图 | 英文F1 | 中文F1 | 中文预测含汉字 |
|---|---|---|---:|---:|---:|

原始QA、图像、权重与Answer F1均不改；四种指令全部报告。完整范围507题，每个权重4视图；probe32仅诊断。没有GPU完成证明时不报告干预效果。

全部配对效应/区间、跨域与同域的指令效应差及源SHA见[JSON](VERIFIED_EVIDENCE.json)。

- Probe32 overlaps full507 and is not independent replication; never pooled.
- All8 full cells required regardless probe results; no TEST-based winner or setting selection.
- Literal Han presence is not language accuracy; existing raw Answer F1 unchanged.
- Paired image intervals condition on these fixed weights; single seed, uncorrected multiple comparisons.
- Recovery without new knowledge would isolate an elicitation component, not prove all knowledge intact.
- No improvement would reject only these fixed cues, not every alignment or knowledge explanation.
