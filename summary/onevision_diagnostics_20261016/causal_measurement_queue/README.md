# 共同取得点测量已入队，不改变原48训练

新CPU入口为`analysis/onevision_causal_measurement_queue.py`，同一nohup服务`llmcl-onevision-causal-priority.service`，启用时PID265251。旧CPU258763仅在确认KillMode=process后退役；基线、两个原CPU观察器及GPU1他人进程均保留。旧队列状态重定向到本目录，不恢复旧所有者。

新增测量用于区分完整旧锚的遗忘与同域正迁移缺失：规范共同S1已存在且其repeat-SeqFT完成时，先进行一次完整507/228推理，再接回原预声明SeqFT优先顺序。全部48控制及后续四组完整方法/确认仍保留；没有新增训练、路由、容量或TEST选模。735题是两个旧锚的测量，不另算训练轨迹。

- GPU1任意其他进程、显存或利用率未空闲时等待，不能因为wait无worker_pid而重启或抢占。
- T04父轨迹已完成，补测待空闲GPU；T02必须等其原repeat轨迹完成和规范S1产生。
- 每次先逐题复现原32题输出/NLL，再全新生成完整507/228题；原32题不拼进完整结果，无优化器/参数修改。
- 失败保留证据并停止，不盲重试；21项CPU调度检查不冒充实际GPU重载通过。

新CPU观察器`llmcl-onevision-acquired-anchor-report.service`启用时PID265271，只读完成证明，另外验证新完整队列中的32题与重载/原预测精确相同，按共同S1→各序列最终分数报告完整旧锚净变化。旧观察器184250/249343保留；没有成绩时不填0。

[实时队列](queue_state.json)、[CPU检查](CPU_VALIDATION.json)、[委托下的测量修订](AUTHORIZATION.json)、[进程保全](ACTIVATION.json)、[测量协议](../shared_anchor_initial/PROTOCOL.md)、[自动核验结果](../shared_anchor_initial/VERIFIED_RESULTS.md)。当前仅准入完成，科学目标与真实GPU门槛仍未完成。
