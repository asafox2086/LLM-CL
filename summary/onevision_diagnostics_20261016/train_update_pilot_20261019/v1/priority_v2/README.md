# 当前唯一CPU调度：TRAIN-only对照优先于后续排队完整测试

当前GPU自然完成后，执行原12格pilot，再回到原完整确认与全部作业。科学工作器/计划完全不变；不按TEST分数调序，不删负条件。[排队修订](SCHEDULE_AMENDMENT.md) · [实际激活](ACTIVATION.json)。实时queue_state.json与实际PID/startticks优先，旧CPU不恢复。

既有已就绪共同锚的优先级仍保留；新版只提前原TRAIN-only对照，不改任何科学条件或完整确认义务。当前无新增GPU效果。
