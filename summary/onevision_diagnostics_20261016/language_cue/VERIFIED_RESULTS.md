# 固定权重：语言/简答指令干预

完整运行证明：4/4。缺失为待测，不是零分。

| 范围 | 权重 | 指令视图 | 英文F1 | 中文F1 | 中文预测含汉字 |
|---|---|---|---:|---:|---:|
| probe | within_medical | original | 30.769 | 16.667 | 19/19 |
| probe | within_medical | answer_only | 30.769 | 20.175 | 18/19 |
| probe | within_medical | question_language | 30.769 | 22.807 | 19/19 |
| probe | within_medical | both | 30.769 | 15.789 | 13/19 |
| probe | cross_short | original | 33.846 | 2.632 | 2/19 |
| probe | cross_short | answer_only | 49.231 | 2.105 | 3/19 |
| probe | cross_short | question_language | 28.205 | 6.842 | 3/19 |
| probe | cross_short | both | 25.641 | 19.850 | 12/19 |
| full | within_medical | original | 46.047 | 22.336 | 251/254 |
| full | within_medical | answer_only | 45.257 | 21.877 | 229/254 |
| full | within_medical | question_language | 44.335 | 22.638 | 247/254 |
| full | within_medical | both | 44.862 | 13.819 | 146/254 |
| full | cross_short | original | 36.087 | 1.234 | 25/254 |
| full | cross_short | answer_only | 37.741 | 2.948 | 43/254 |
| full | cross_short | question_language | 36.503 | 5.072 | 53/254 |
| full | cross_short | both | 30.632 | 13.225 | 145/254 |

原始QA、图像、权重与Answer F1均不改；四种指令全部报告。完整范围507题，每个权重4视图；probe32仅诊断。没有GPU完成证明时不报告干预效果。

全部配对效应/区间、跨域与同域的指令效应差及源SHA见[JSON](VERIFIED_EVIDENCE.json)。

- Probe32 overlaps full507 and is not independent replication; never pooled.
- All8 full cells required regardless probe results; no TEST-based winner or setting selection.
- Literal Han presence is not language accuracy; existing raw Answer F1 unchanged.
- Paired image intervals condition on these fixed weights; single seed, uncorrected multiple comparisons.
- Recovery without new knowledge would isolate an elicitation component, not prove all knowledge intact.
- No improvement would reject only these fixed cues, not every alignment or knowledge explanation.
