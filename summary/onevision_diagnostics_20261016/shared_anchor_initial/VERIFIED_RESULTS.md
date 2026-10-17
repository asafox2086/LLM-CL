# 完整旧锚：最终成绩减去真实共同取得点

已核验共同起点0/2，完整净变化配对0/8。未完成不填0。F1唯一主指标，NLL辅助。

| 锚/序列 | 完整题数 | 共同S1 F1→最终F1 | F1净变化及95%图像区间 |
|---|---:|---:|---:|

先区分跨域自身净变化与同域净变化，再解释两者差距；不能把相对同域的差距全部叫遗忘。

逐题质量变化、配对NLL、参数/输入/预测SHA见[JSON](VERIFIED_RESULTS.json)。

- Full native medical anchors507/228 only; not full21454 or19-task FWT/BWT.
- Common acquired stage1 differs from untrained base and original19-stage one-epoch protocol.
- Follow-up on already observed test sets, not untouched confirmation or TEST-selected model choices.
- Intervals conditional on fixed trained models; no seed/order replication, uncorrected image bootstrap.
- Final-minus-acquired measures net change; cannot uniquely attribute it to knowledge, alignment or interference.
- Missing starts/finals omitted, not zero; original runs and weights remain immutable.
