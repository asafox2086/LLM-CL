# 本轮运行进度

更新时间：2026-10-13T23:44:37.093642+08:00

当前阶段：**running_or_waiting_gpu**。

seed 42；每任务 train/dev/test = 2000/100/200；十任务一轮。

| 方法 | 状态 | GPU | 已完成训练阶段 |
|---|---|---|---|
| SeqLoRA | completed | 0 | 10 |
| SMoLoRA | running | 0 | 0 |
| DGG | adapter_pending | — | 0 |
| PaRSP | adapter_pending | — | 0 |
| PASs_MoE | adapter_pending | — | 0 |
| Hyper_LLaVA | adapter_pending | — | 0 |

待适配方法：DGG、PaRSP、PASs-MoE、Hyper-LLaVA；不以简化实现冒充作者方法。

总日志：`exp/ten_task_cl_20261012/runs/SMoLoRA/seed_42/20261012_200725/logs/retry_supervisor.log`。
运行日志与阶段检查点：`exp/ten_task_cl_20261012/runs/<method>/seed_42/<run_id>/`。

正式训练已开始：True。
仅在 GPU 空闲时分配不同方法，不停止其他用户进程。
