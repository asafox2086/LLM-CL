# 新候选：连续习得＋有界功能缓存

研究原型，未发表、未验证新颖性。研究在新GPU0/1择空、我们最多占3卡；主训练仍1313QA/19阶段/seed42，原生AnyRes/BF16，唯一原始Answer F1，不混正式V2。CPU门禁已通过、已接唯一nohup调度；当前因其他用户GPU0/1进程存活等待，**不能称GPU已经开始或获得收益**。

| 模块 | 源码 | 实际约束 |
|---|---|---|
| A 连续习得＋有界状态保留 | [trial](../../analysis/onevision_state_cache_trial.py)、[core](../../analysis/onevision_state_cache_core.py) | 同一LoRA连续93次主更新，每阶段必须与原SeqFT所有48层A/B精确相同；当前状态必留、最多3个功能状态，TRAIN覆盖加复杂度代价决定保留 |
| B TRAIN功能质量调用 | 同core | 原20条TRAIN锚图像/问题特征，固定cos温度0.1估计缓存状态NLL；不训练新投影或头，不读TEST答案/任务/领域标签 |
| 调度和结果核验 | [queue](../../analysis/onevision_state_cache_queue.py)、[report](../../analysis/onevision_state_cache_report.py) | 全部四格诊断；几何/质量两个缓存候选无论诊断结果均完整21454，单状态两视图必须函数相同 |

[问题、预算、全部四格与GPU门槛](../../summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/PROTOCOL.md) · [实际激活与等待](../../summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/queue/README.md) · [结果主表](../../summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/VERIFIED_RESULTS.md)。

主优化器/loss/原始梯度与19阶段pool均保存；每阶段纯TRAIN峰值、整个方法选择前后/全局峰值分开。缓存3不含临时候选/父状态校验副本，0路由参数不等于零内存，原型最多76次特征调用、20TRAIN NLL查询及历史与checkpoint全计费。最终只有一个主训练流，缓存是历史参数状态，不称自动学出最优独立专家数量。

原[完整继承＋16参数质量头](../OneVision_ColdStartQuality/README.md)是不同冻结候选，四格155诊断已完成但完整确认仍待，不改名或替换其结果。新候选尚无F1收益、轻量优势或大幅提升证明，正式主表Ours保持空。

最新[设备授权补充](../../summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/DEVICE_POLICY.md)及[当前flex队列](../../summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/flex_queue/README.md)覆盖上述旧queue入口，算法原文件冻结不变。
