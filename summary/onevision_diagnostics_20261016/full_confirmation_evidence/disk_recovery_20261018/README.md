# 历史完整确认报告服务恢复

2026-10-18。本记录是CPU报告链恢复，不是新训练、推理、完整测试完成或方法提升。

## 发现的实际状态

原`llmcl-onevision-full-confirmation-report.service`于2026-10-18 03:56:02 CST以退出码1结束，旧PID224209已不存在；状态文件停在`verified_caps=[1]`。退出时间与此前记录的存储事故吻合，但该服务的异常日志为空，**直接原因未获证明**。最初尝试确认Errno28的检查因此拒绝了该推断，没有据此启动或重试服务。

这个服务只观察旧cap1/cap3完整确认，不拥有GPU训练。当前功能调用完整评测由另一个报告器处理，其运行不依赖这次恢复。cap3仍未完成，不能将恢复观察服务写成cap3结果。

## 恢复前证据

- 原始报告源码SHA与此前CPU门禁一致：`72cb0aae67483b6ba5b5189e4d5c6b11f52e4ba90e170043ea0319b159617f8d`，未改源码。
- cap1既有完整核验所绑定的41份文件逐一重新计算SHA，全部一致；没有重新生成或拼接预测。
- 本目录实际写入/读取探针通过，随后清理该临时探针。
- 用原源码单次执行CPU报告，返回0，确认`verified_caps=[1]`；其余原GPU/基线/调度/报告PID及startticks不变。

上述检查通过后才启动原CPU服务，不是对失败GPU实验的盲重试。原退出码/时刻、日志缺失、旧状态、源码SHA及保护进程见[BEFORE.json](BEFORE.json)；单次执行结果见[ONE_SHOT_VALIDATED.json](ONE_SHOT_VALIDATED.json)。

## 实际恢复

新观察器PID381186/startticks15195745，`CPU_only=true`，`verified_caps=[1]`。原service的`Restart=no`保持，未增加自动重试。恢复前后受保护进程的PID/startticks精确一致；GPU358091、新CPU调度376997、新机两基线及既有功能调用/温启动/数据因子报告服务均未停止。见[RESTORED.json](RESTORED.json)。

原cap1完整AP41.361保持不变；cap3与完整转移格仍待，恢复不产生新分数。仅恢复对未来完整证明的自动核验，所有实验条件和准入门槛保持。
