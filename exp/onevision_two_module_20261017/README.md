# OneVision两模块试验（2026-10-17）

用户授权：从已有问题设计两个模块，追求较大幅度提升。仅新GPU1，基线后台照常运行。

先做同参数rank8/540672的12个小任务原型；随后完整19阶段/1%1313TRAIN/21454TEST四组对照。完整训练、最终全测试、后续稀疏FWT/BWT、逐步梯度/内存/恢复和只读报告代码已实现并排队，尚无实际GPU结果，不能用CPU或小试验代替。唯一指标Answer F1；原始数据、测试与代码来源分别锁定。

详见[协议](../../summary/onevision_diagnostics_20261016/two_module_prototype/PROTOCOL.md)及[源码索引](../../code/OneVision_TwoModule/README.md)。大权重/原始数据/逐题运行文件只留服务器，不入Git。

[完整19阶段协议/成本](../../summary/onevision_diagnostics_20261016/two_module_fullstream/PROTOCOL.md) · [诊断/全量分开结果](../../summary/onevision_diagnostics_20261016/two_module_fullstream/VERIFIED_RESULTS.md)。完整轨迹四组都保留阶段快照支持后续评测，正式基线保存策略不改。
