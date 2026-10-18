# 当前唯一研究队列：SVD机制对照v3

同nohup服务llmcl-onevision-causal-priority.service，CPU86675/startticks1647301，analysis/onevision_osft_budget_queue_v3.py。实际GPU1 worker86799/startticks1649052已通过48权重安装和图文前向门槛、保存stage00；仍在初始评测，无训练成绩。

[协议](../PROTOCOL.md) · [准入](READY.json) · [激活](ACTIVATION.json) · [v2故障及修复边界](../V2_REPAIR_EQUIVALENCE.json)。旧CPU75694/worker84389在训练前失败，原证据保留，不盲重试、不恢复旧所有者。修复仅数值分解／重建路径，未改原门槛／QA／秩／超参；CPU分解成本计费。源已冻结，不热改。

先两组19阶段训练→原18数据因子余项→两组完整21454→所有原完整与旧工作，全部保留。只用真正空闲GPU0/1单研究流、我们最多3新卡，基线不停、不抢他人。

最新用户取消预设模块数量；这里两组是机制条件，不是方法模块。单1%开关不能独自证明数据受限交互，后续数据控制见[问题优先方案](../../../DATA_LIMITED_PRIORITY.md)。
