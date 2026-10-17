# 记住知识后，字面相关题是否改善？

只读四条已完成知识轨迹；不新增事实、训练或GPU推理。唯一Answer F1，无额外答案处理。

四组训练前医学100题和KG267题的原ID、金标、预测、F1和NLL全部逐值一致，排除了这两组目标初始输出不同的混杂。两测试组重叠12题，不合并计数。

| rank | 原生KG字面分组 | QA/图像数 | 起点F1 | 正确知识64步F1 | 错配64步F1 | 正确−起点及95%区间pp | 正确−错配及95%区间pp |
|---|---|---:|---:|---:|---:|---|---|
| 1 | same_relation_fixed_tail | 31/29 | 15.054 | 11.885 | 5.376 | -3.169 [-17.497, +12.423] | +6.509 [-0.827, +16.765] |
| 1 | relation_only | 40/33 | 15.458 | 3.250 | 2.000 | -12.208 [-18.708, -5.385] | +1.250 [+0.000, +3.947] |
| 1 | relation_absent | 196/99 | 6.694 | 6.303 | 3.620 | -0.391 [-3.436, +2.663] | +2.683 [-0.082, +5.495] |
| 8 | same_relation_fixed_tail | 31/29 | 15.054 | 12.903 | 9.005 | -2.151 [-14.444, +10.753] | +3.898 [-5.376, +14.028] |
| 8 | relation_only | 40/33 | 15.458 | 5.000 | 0.000 | -10.458 [-17.350, -3.049] | +5.000 [+0.000, +11.111] |
| 8 | relation_absent | 196/99 | 6.694 | 5.824 | 5.085 | -0.870 [-4.491, +2.837] | +0.739 [-1.552, +3.057] |

**能支持：** 现有实验不能以“正确知识优于错误知识”代替“加知识优于不加”。31条同关系/固定tail字面匹配题的正确知识点估计也低于共同起点，至少不能仅凭总体覆盖不足就声称知识注入会成功。rank8的16事实文本回忆成功已另有证据，目标迁移仍需证明。

**不能支持：** 不能据此断言知识无用、视觉绑定就是唯一瓶颈或该31题真正被知识覆盖。该分组使用作者三元组，是事后审计，不用于选事实/路由/模型；所有三组、两个rank、8/64两个预算完整记录。区间含0的地方不能宣称显著损害。

**下一项可证伪问题：** 固定事实内容和预算，区分事实可记忆性、事实与图像实体的绑定、目标问答格式/监督；严格多域/少任务48组机制对照仍不可由本审计替代。

- Retrospective author-triple strata, not a predeclared holdout confirmation or semantic support annotation.
- Relation/tail overlap does not establish matching image/head entity, fact sufficiency, aliases or answerability.
- All strata/8 and64 steps/both ranks reported; no facts, method or budget selected from these scores.
- Conditional single-fit image-cluster intervals, no multiple-testing correction or seed replication.
- Medical100 and nativeKG267 overlap12, not367 independent QA. Only nativeKG267 is stratified here.
- Successful text fact recall plus weak native QA transfer does not identify binding as the unique cause.
- This is an auxiliary512-exposure intervention, not the formal1epoch1313-QA CL experiment or full21454 test.
