# 在原48组内部，先完成两个锚的同域/跨域SeqFT

用户已委托“这个gpu1的方案你自己定，我只要结论”。本次顺序由研究代理在该委托下选择，不是用户另行逐项指定。

目的：先取得目标所需的同任务/同域/跨域、两个医学锚的共同控制，避免等完一个序列的所有机制才看到跨域证据。排序只根据原计划中的方法标签与锚顺序，不使用TEST成绩选择实验；全部48组保留，训练/评测/数据/ID/模型/源码均不变。

- 优先原case索引0、6、12、18、24、30、36、42，即两锚各4条SeqFT；已完成组不重跑。T02的repeat首先建立原规范stage1，之后才轮到该锚其他序列。
- 当前GPU作业自然结束，随后采用新顺序；剩余40条机制控制按原索引补齐，不取消任何方法。
- 完整顺序仍为cap1已完成→12原型已完成→严格48→四组完整19阶段/最终21454→cap3→公共完整初始与稀疏FWT/BWT。
- 21项CPU检查通过，原GPU工作器、原调度库和两个CPU报告观察器未改。新CPU入口冻结，后续修复需新版本。

仍使用nohup服务`llmcl-onevision-causal-priority.service`，新入口为`analysis/onevision_causal_anchor_first_queue.py`。切换前确认`KillMode=process`，仅停止旧CPU220246。正在跑的GPU258179/startticks9229970、基线174247/175176和观察器184250/249343均保留；新CPU258763先等待该GPU自然结束。随后实际启动原case12的跨域SeqFT，正常接续，不是重跑或拼接结果。

当前入口：[queue_state.json](queue_state.json)，配合实际PID/日志/完成证明判断。wait阶段可能没有worker_pid；GPU忙或遗留的实际作业不是调度故障。旧队列状态已指向本路径，不能恢复旧CPU所有者。

[选择规则与CPU检查](CPU_PRIORITY_VALIDATION.json)、[委托依据与源SHA](AUTHORIZATION.json)、[进程保全证明](ACTIVATION.json)。调度成功不代表科学目标已完成，仍须完成配对对照与原因验证。
