# 当前唯一队列：函数补偿初始化诊断

> v1启动前置导入失败，原CPU／worker已退出；已保留故障并由独立[v2队列](../../v2/queue/README.md)修复接续。下文是原准入记录，不要恢复旧所有者。科学设置与门槛不变，v1没有模型或训练结果可拼接。

2026-10-18，同一nohup systemd服务的CPU所有者26045。原CPU12470已计划退役，KillMode=process；当前GPU25656、两条新机基线3256／7965及所有受保护进程PID/startticks保持不变。实时以[queue_state.json](queue_state.json)、[ACTIVATION.json](ACTIVATION.json)和实际PID／日志为准，不能恢复历史CPU所有者。

当前先等继承的原数据因子case2自然结束。原五格来源控制已全部完成，随后固定顺序：cold_compensated真实精确等价门槛及完整507→within_compensated完整507→cross_compensated完整507→全部原诊断／后续作业。两个公共完整确认仍后移保留。不根据TEST赢家决定运行哪些条件。

研究在实际空闲GPU0／1择一，仅一条研究流，我们的新机作业最多占三张卡，保留其他用户进程及两机基线。新CPU调度通过8策略／2新旧命令及全部祖先检查；CPU不是GPU性能证明。失败保存证据并停止，不放宽等价门槛、不盲重试、不拼旧预测。

[协议与解释边界](../PROTOCOL.md) · [全部结果](../VERIFIED_RESULTS.md) · [CPU实际源因子门槛](../CPU_PREFLIGHT.json) · [授权来源](AUTHORIZATION.json)。三条件为机制控制，不是已完成的五域新方法；最终表示与适配形式仍开放。
