# 完整旧锚：最终成绩减去真实共同取得点

已核验共同起点1/2，完整净变化配对4/8。未完成不填0。F1唯一主指标，NLL辅助。

| 锚/序列 | 完整题数 | 共同S1 F1→最终F1 | F1净变化及95%图像区间 |
|---|---:|---:|---:|
| T04/repeat_anchor | 507 | 20.361→35.102 | +14.741 [+11.332,+18.558] |
| T04/within_medical | 507 | 20.361→34.168 | +13.808 [+10.564,+17.288] |
| T04/cross_short | 507 | 20.361→18.626 | -1.735 [-4.611,+1.062] |
| T04/cross_five_domains | 507 | 20.361→17.196 | -3.165 [-6.952,+0.527] |

先区分跨域自身净变化与同域净变化，再解释两者差距；不能把相对同域的差距全部叫遗忘。

逐题质量变化、配对NLL、参数/输入/预测SHA见[JSON](VERIFIED_RESULTS.json)。

- Full native medical anchors507/228 only; not full21454 or19-task FWT/BWT.
- Common acquired stage1 differs from untrained base and original19-stage one-epoch protocol.
- Follow-up on already observed test sets, not untouched confirmation or TEST-selected model choices.
- Intervals conditional on fixed trained models; no seed/order replication, uncorrected image bootstrap.
- Final-minus-acquired measures net change; cannot uniquely attribute it to knowledge, alignment or interference.
- Missing starts/finals omitted, not zero; original runs and weights remain immutable.
