# 保留已有功能的轻量更新：TRAIN-only小对照

借鉴LoRA-FA的冻结A拓扑及LoRA+的不同学习率；四条件／三个目标任务／12格均执行。当前仅CPU准入并排队，尚无GPU结果，不发布F1或最优方法。

[协议与预算](PROTOCOL.md) · [冻结计划](PLAN.json) · [CPU检查](CPU_CHECK.json) · [当前队列](queue/README.md)。先完成两系数完整确认，再做本对照，保留原所有工作和基线。只依据TRAIN拟合与旧TRAIN损伤诊断，不用TEST挑配置。
