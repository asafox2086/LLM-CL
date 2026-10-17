# 双锚确定性机制复核：自动核验结果

已接受完整轨迹 **0/48**。仅接受完整SHA、共同stage1参数和输出精确一致、原ID/金标/评分一致的结果。未完成不填0。

| 锚 | 序列 | 机制 | 旧锚Answer F1 S1→S5 | 质量下降/上升题数 | 当前任务Answer F1 | 旧锚NLL变化 |
| --- | --- | --- | ---: | ---: | ---: | ---: |

同域/跨域差异、方法对SeqFT的配对差异、当前任务学习、实际训练loss/LR/梯度、额外信息预算及每阶段训练前后/峰值内存见[JSON](VERIFIED_RESULTS.json)。没有GPU结果时不得称该报告已证明领域原因。

- Zero accepted cases means pending, not zero effect. No incomplete-run inference.
- Two medical anchors, one fixed order/seed and32QA probes; no universal pure domain-count causal claim.
- ER, Fisher and anchor reference use extra history/compute; interpret old and new scores together.
- QA bootstrap intervals conditional on trained weights, uncorrected; cannot replace training-seed/order replication.
- Full native-anchor evaluation only final SeqFT507/228, not all methods/full21454/FWT/BWT.
- Original scores only; previous auxiliary format parsers do not replace primary values.
