> 当前唯一调度已接续到[CPU366824/cold_data_queue](../cold_data_queue/README.md)；本页保留前次准入过程，旧363113不恢复。原温启动实验仍保留，另新增均匀曝光的数据量因子控制，GPU效果均待。

# 当前唯一研究CPU调度入口

2026-10-18：`llmcl-onevision-causal-priority.service`改用新冻结`analysis/onevision_warm_start_queue.py`，CPU363113/startticks14840433。旧356616仅CPU计划退役，KillMode=process；完整评测GPU358091/startticks14710479原样保留并自然运行。原观察器、两机6个基线实际PID已核验，源码不变。

顺序：可用共同S1 → 原两组相似度训练 → 原四视图 → **两个既定功能调用完整21454** → **初始化单变量19阶段温启动训练** → **原可学习冷启动完整21454** → **温启动完整21454** → 原固定语言指令/严格48（SeqFT优先）/固定相似度完整确认/其余原方法和稀疏转移。所有原任务保留，不按TEST表现挑选或删候选。

调度28项、实际存活worker继承7项CPU检查通过。首项新增训练仍待前置运行，**准入不等于GPU训练已开始或已有效果**。外国GPU1进程、显存/利用率门槛继续阻止新作业准入；缺失worker完成证明视为故障，不能据旧状态或超时重启。

唯一实时状态：[queue_state.json](queue_state.json)。来源：[AUTHORIZATION](AUTHORIZATION.json)、[ACTIVATION](ACTIVATION.json)、[CPU策略](CPU_POLICY_CHECK.json)、[继承检查](INHERITANCE_CPU_CHECK.json)。旧`functional_confirmation_queue/queue_state.json`已redirect，不恢复旧所有者。

新增实验细节见[初始化控制协议](../warm_start_20261018/PROTOCOL.md)。GPU源、方法、训练ID和旧实验均没有热改。
