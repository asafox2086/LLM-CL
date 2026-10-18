# 数据因子完成后的包装器故障：恢复版本4

旧GPU427417已完整完成case00：四次正学习率/有限梯度更新、全部228题初始/1步/4步评测与原始参数/优化器/梯度/内存证据封存。随后包装器误向RepairedExperiment索要仅MetricExperiment才生成的辅助特征回执，触发断言；CPU405559正常按失败规则停止，不能将旧queue_state当仍在跑。

[原失败及SHA](../queue_v3/FAILURE_EVIDENCE.json)。不改、不重训、不拼接case00。新源码[queue_v4](../../../../analysis/onevision_identity_repair_queue_v4.py)只令数据因子调用原冻结入口；仍需要特征的其他任务继续使用原元数据修复包装器。仅在case00完整证明、源、全部artifact SHA、四步正LR/有限梯度和类调用路径通过后，承认这一条已解释的包装器失败；任何新failure仍停止。

版本3启动时缺失祖先授权字段，CPU即退出，未启动GPU；原[失败](../queue_v3/needs_attention.json)保留。版本4逐字段继承全部祖先授权，独立新SHA字段避免覆盖原规则；37策略+66祖先+36命令检查通过，[授权继承检查](AUTH_INHERITANCE_CHECK.json)。不通过放宽参数/输入/结果门槛恢复。

同nohup服务已实际激活CPU430421，GPU430521开始case01；以实际PID/startticks及服务器该目录的`queue_state.json`为准。全部受保护进程保全见[激活](ACTIVATION.json)。research仅GPU1，GPU0保留，基线继续。后续全部18数据因子、冷/温/组合完整21454、原机制/其他作业保持；v2/v3均不恢复。
