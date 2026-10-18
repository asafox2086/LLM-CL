# 当前唯一研究调度：SVD参数预算对照v2

同nohup llmcl-onevision-causal-priority.service，初始实际CPU75694，源码analysis/onevision_osft_budget_queue_v2.py。以queue_state.json／ACTIVATION.json及实际PID/startticks为准；旧73625因模型前CPU故障停用，旧61499计划退役，均不恢复。

[协议](../PROTOCOL.md) · [准入](READY.json) · [实际激活](ACTIVATION.json) · [修复边界](../V1_REPAIR_EQUIVALENCE.json)。只用真正空闲0/1、单研究流、我们至多3新机卡，尊重所有其他作业。等待他人GPU不是worker故障。

顺序：保全当前自然完成→两组SVD19阶段→原18数据因子全部余项→两组SVD完整21454→全部coverage／其他方法完整和原工作。没有按TEST选赢家或删除负条件。新5项worker／0和1实际导入、10项新调度／3实际新旧命令及所有祖先CPU检查通过；不是GPU成绩。所有新源码和协议在READY之后冻结，修复必须新版本。
