# 任务数量与领域数量：已完成对照快照

已完成24/24个完整序列／方法组合。仅纳入已完成标记及校验通过的阶段1/3/5，未完成组合不列成绩。

| 序列 | 机制 | S1 NLL | S3 NLL | S5 NLL | S1正确/23 | S3正确/23 | S5正确/23 |
|---|---|---:|---:|---:|---:|---:|---:|
| repeat_anchor | SeqFT_LoRA | 6.051 | 1.913 | 1.570 | 2/23 | 8/23 | 8/23 |
| repeat_anchor | ER_LoRA | 5.998 | 1.838 | 1.642 | 2/23 | 9/23 | 6/23 |
| repeat_anchor | EWC_lambda1 | 6.010 | 1.924 | 1.566 | 2/23 | 8/23 | 8/23 |
| repeat_anchor | EWC_lambda100 | 6.043 | 1.944 | 1.710 | 2/23 | 9/23 | 9/23 |
| repeat_anchor | anchor_gradient_noop | 5.959 | 1.890 | 1.542 | 2/23 | 9/23 | 8/23 |
| repeat_anchor | anchor_gradient_projection | 5.989 | 1.908 | 1.550 | 2/23 | 9/23 | 8/23 |
| within_medical | SeqFT_LoRA | 6.018 | 2.500 | 1.651 | 2/23 | 8/23 | 9/23 |
| within_medical | ER_LoRA | 5.984 | 3.270 | 1.730 | 2/23 | 3/23 | 9/23 |
| within_medical | EWC_lambda1 | 5.930 | 2.495 | 1.649 | 2/23 | 8/23 | 10/23 |
| within_medical | EWC_lambda100 | 5.946 | 2.572 | 1.746 | 2/23 | 8/23 | 10/23 |
| within_medical | anchor_gradient_noop | 6.034 | 2.496 | 1.651 | 2/23 | 8/23 | 9/23 |
| within_medical | anchor_gradient_projection | 5.955 | 2.479 | 1.636 | 2/23 | 8/23 | 9/23 |
| cross_short | SeqFT_LoRA | 5.985 | 6.100 | 5.512 | 2/23 | 4/23 | 4/23 |
| cross_short | ER_LoRA | 6.056 | 2.493 | 1.942 | 2/23 | 6/23 | 7/23 |
| cross_short | EWC_lambda1 | 6.003 | 5.695 | 5.314 | 2/23 | 3/23 | 6/23 |
| cross_short | EWC_lambda100 | 5.988 | 5.672 | 5.454 | 2/23 | 3/23 | 4/23 |
| cross_short | anchor_gradient_noop | 5.963 | 5.180 | 5.450 | 2/23 | 3/23 | 4/23 |
| cross_short | anchor_gradient_projection | 5.994 | 5.001 | 4.162 | 2/23 | 4/23 | 5/23 |
| cross_five_domains | SeqFT_LoRA | 5.982 | 5.933 | 5.452 | 2/23 | 3/23 | 4/23 |
| cross_five_domains | ER_LoRA | 5.971 | 2.502 | 1.905 | 2/23 | 7/23 | 7/23 |
| cross_five_domains | EWC_lambda1 | 6.019 | 5.982 | 5.484 | 2/23 | 4/23 | 4/23 |
| cross_five_domains | EWC_lambda100 | 5.973 | 5.547 | 5.582 | 2/23 | 3/23 | 3/23 |
| cross_five_domains | anchor_gradient_noop | 5.943 | 5.960 | 5.555 | 2/23 | 3/23 | 3/23 |
| cross_five_domains | anchor_gradient_projection | 6.019 | 4.941 | 4.679 | 2/23 | 6/23 | 3/23 |

S1对所有序列／方法均是同一个Organ锚任务、相同8条QA和4次更新。其目标函数／初始状态没有历史回放或Fisher／投影项；默认计算下已观察到阶段1权重差异。细小方法差距不能直接归因于机制。原生顺序／独立任务的固定后端配对重跑等待现有队列完成；本4×6机制对照自身的确定性复核仍待补齐。

重复锚序列增加优化次数但不增加独立训练QA；若同域新任务也改善锚性能，说明“任务一多必然遗忘”不成立于该前缀，但仍需跨域控制、更多锚／顺序和稳定数值设置。

限制：

- Single23-question medical Organ anchor; diagnostic, not full test.
- Correct counts are normalized exact-reference matches, not expert semantic/clinical judgments.
- Default numerical backend; same seed did not give identical stage1 weights across cases.
- Same first stage has no replay/Fisher/history/projection; small case gaps are not algorithmic proof.
- Repeat sequence reuses same8TRAIN QA for5 stages;40 target exposures, not40 distinct QA.
- ER/Fisher/gradient-reference use extra historical data/compute, recorded separately.
- No cross-domain attribution until corresponding cases complete; no ranking by interim results.
