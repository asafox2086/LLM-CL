# OSFT短侧因子训练必要性：一组有限配方先导

问题与推导见[PROOF](PROOF.md)，预定干预/正负行动见[PROTOCOL](PROTOCOL.md)。固定完整短侧补基，在精确算术及同一阶段高基下不减允许集合；有限优化下的Answer F1是否保留仍待实测。LoRA-FA提供固定一因子的先例，不称新算法；端侧只科研背景。

五源及协议已独立评审接受。唯一小HF CPU准入3.513秒通过：初态state/输出/RNG、共享原始梯度精确，长/宽两分支、真实冻结侧投影后全digest、阶段refactor重冻、HostAdamW/scheduler一更新均通过。READY SHA61cbee3f91df15a997169e184754ef8bd8dbb364cb6823f76c826ee6ec6f667f。CPU门槛不是实际0.9B质量或成本收益。

2026-10-22 02:16:35UTC启动观察：独立nohup服务llmcl-onevision-osft-short-factor-control-20261022.service，controller64178/start3179335实际active/wait；无worker/STARTED/COMPLETE/FAILURE。02:17:09UTC GPU1他人63543/start3172525存活，资源4506MiB/40%为该次快照。正式6262/6264/6266/5774/5779身份仍匹配。当前只是等待资源，后续以真实queue/身份和sealed proof为准，不照抄本快照当live，不分钟巡检。

只NEW物理GPU1一研究流，预计89485056训练坐标，保留原全部USV存储。100已探索DEV/87更新完整封存后才看结果；训练梯度和moment字节另测，不称推理压缩或受控整模加速。若固定点估计及成本判据未过就结束；过也只保留后续必要性评审，不自动PASs/1313/21454，不写正式Ours。原完成作业不恢复，原源/READY后不热改。科研目标未实现。
