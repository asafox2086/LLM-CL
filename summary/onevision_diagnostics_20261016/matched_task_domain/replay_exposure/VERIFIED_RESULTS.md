# 固定8条缓冲区，实际保护了多少旧锚题目

TRAIN-only CPU审计：8条ER计划，2条完成GPU日志已精确核验。未完成行仅是冻结规则的计划覆盖，不是模型结果。

| 锚/序列 | 实际GPU完成 | 回放总曝光 | 旧锚回放曝光/独特QA | 各阶段旧锚曝光S2–5 |
|---|---|---:|---:|---|
| T04/repeat_anchor | True | 32 | 32/8 | 8/8/8/8 |
| T04/within_medical | True | 32 | 15/6 | 8/3/1/3 |
| T04/cross_short | False | 32 | 20/7 | 8/4/5/3 |
| T04/cross_five_domains | False | 32 | 17/7 | 8/4/2/3 |
| T02/repeat_anchor | False | 32 | 32/8 | 8/8/8/8 |
| T02/within_medical | False | 32 | 15/7 | 8/5/1/1 |
| T02/cross_short | False | 32 | 20/7 | 8/4/4/4 |
| T02/cross_five_domains | False | 32 | 20/7 | 8/4/4/4 |

每阶段缓冲区的QA/任务/领域、图像路径、实际抽中与独特旧锚题目见[JSON](EXPOSURE.json)。

数量相同不等于历史信息相同：跨序列对照必须同时报告覆盖。这里不改变采样，不把覆盖差异直接解释为准确率原因。

- Planned IDs are determined by frozen TRAIN policy, not completed GPU results.
- Only complete ER cases have actual replay-log validation; incomplete runs are not read.
- Eight-slot hash memory is this mechanism control, not every formal baseline's buffer policy.
- Anchor-QA coverage is not semantic knowledge coverage; other tasks/images can transfer.
- Image-path counts are not decoded-pixel/semantic deduplication.
- Coverage differences do not prove final F1 causality or justify TEST-selected sampling.
- No answer contents, gradient scores or future/test data choose replay in this audit.
