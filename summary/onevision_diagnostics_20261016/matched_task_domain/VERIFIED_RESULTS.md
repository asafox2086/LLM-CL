# 双锚确定性机制复核：自动核验结果

已接受完整轨迹 **12/48**。仅接受完整SHA、共同stage1参数和输出精确一致、原ID/金标/评分一致的结果。未完成不填0。

| 锚 | 序列 | 机制 | 旧锚Answer F1 S1→S5 | 质量下降/上升题数 | 当前任务Answer F1 | 旧锚NLL变化 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| T04 | repeat_anchor | SeqFT_LoRA | 19.375→27.604 | 2/6 | 22.500 | -4.5018 |
| T04 | repeat_anchor | ER_LoRA | 19.375→25.000 | 3/5 | 23.958 | -4.5308 |
| T04 | repeat_anchor | EWC_lambda1 | 19.375→27.083 | 2/6 | 23.958 | -4.4959 |
| T04 | repeat_anchor | EWC_lambda100 | 19.375→23.958 | 3/5 | 22.917 | -4.4082 |
| T04 | repeat_anchor | anchor_gradient_noop | 19.375→27.604 | 2/6 | 22.500 | -4.5018 |
| T04 | repeat_anchor | anchor_gradient_projection | 19.375→27.604 | 2/6 | 22.500 | -4.5018 |
| T04 | within_medical | SeqFT_LoRA | 19.375→22.396 | 2/5 | 28.589 | -4.3787 |
| T04 | within_medical | ER_LoRA | 19.375→23.958 | 2/5 | 27.247 | -4.3852 |
| T04 | within_medical | EWC_lambda1 | 19.375→22.396 | 2/5 | 28.589 | -4.3209 |
| T04 | within_medical | EWC_lambda100 | 19.375→23.958 | 3/6 | 28.021 | -4.2887 |
| T04 | within_medical | anchor_gradient_noop | 19.375→22.396 | 2/5 | 28.589 | -4.3787 |
| T04 | cross_short | SeqFT_LoRA | 19.375→15.312 | 5/3 | 48.958 | -0.8043 |

同域/跨域差异、方法对SeqFT的配对差异、当前任务学习、实际训练loss/LR/梯度、额外信息预算及每阶段训练前后/峰值内存见[JSON](VERIFIED_RESULTS.json)。没有GPU结果时不得称该报告已证明领域原因。

- Zero accepted cases means pending, not zero effect. No incomplete-run inference.
- Two medical anchors, one fixed order/seed and32QA probes; no universal pure domain-count causal claim.
- ER, Fisher and anchor reference use extra history/compute; interpret old and new scores together.
- QA bootstrap intervals conditional on trained weights, uncorrected; cannot replace training-seed/order replication.
- Full native-anchor evaluation only final SeqFT507/228, not all methods/full21454/FWT/BWT.
- Original scores only; previous auxiliary format parsers do not replace primary values.
