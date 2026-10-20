**2026-10-20更新：** FOREVER／OneVision正式完整核验通过，已填根README与内部主表；[完整证明、回放成本与聚合口径](../../forever_verified_20261020/README.md)。总目标16组已有6组完整，原七方法发布器14范围保持不变。下文早期启动快照不能用于重启已正常结束的1207220／1207239。

# FOREVER图文回放：正式运行与阶段恢复

两种骨干都通过真实模型、回放、保存与另进程重载门槛。正式数据和其他基线完全相同：1%训练1313条、19原生任务、完整测试21454条，唯一原始Answer F1。额外作者回放独立报告计算与内存，不把预检当成实验成绩。

| 骨干 | 正式运行状态 | 可发布成绩 |
|---|---|---|
| LLaVA-OneVision 0.5B | 正式运行自然完成，旧CPU／GPU均已退出；完整v3核验通过 | AP45.245／FWT17.748／BWT5.642 |
| Qwen2-VL 2B | 新CPU482168等待原队列后空闲GPU2 | 尚无 |

以上是启动快照。实时权威在各运行主机`formal_queue/<backbone>/state.json`和实际PID/startticks；本目录`old_runtime_snapshot`仅只读证据，不能用于重复启动。原五条基线继续，新增OneVision后六条GPU基线；完整核验仍4组。总目标8方法×2骨干=16组，原七方法发布器只覆盖其中14组，FOREVER由独立v3验证器核验。

[协议](PROTOCOL.md)／[正式准入与冻结源码](FORMAL_ADMISSION.json)／[双骨干GPU与重载证明](BOTH_GPU_MODEL_KERNEL_GATES.json)／[训练内核不变证明](IO_EQUIVALENCE.json)／[阶段恢复CPU控制](ROLLING_IO_CPU_CHECKS.json)。独立验证器为`analysis/forever_multimodal_formal_verify_v3_20261019.py`，完整19阶段、74必要评测格、21454最终QA、原评分、更新及最终权重证明齐全才发布。

每阶段完整结束后原子提交最新checkpoint，再删除早期权重；保留SHA元数据。恢复时不混入未提交分支，原分支整体归档，资源按进程分段。模型／回放内核与v2门槛相同，v3只改运行I/O。7项CPU事务检查不代表已进行真实服务器断电测试，也不代表F1改善。两骨干正式运行均从基座新鲜开始，预检权重及预测不复用。

[已保留的v1保存错误与v2修复](../v2/README.md)。故障保留证据，独立新版本修复；不热改活动源，不盲重试。
