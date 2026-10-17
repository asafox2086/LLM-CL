# 待选择的排队调整方案（尚未启用）

核心缺口是同任务重复/同域新任务/跨域新任务的严格配对，现有队列让其等待cap1和cap3两次完整确认。[当前双锚协议](../../../analysis/ONEVISION_MATCHED_TASK_DOMAIN_PROTOCOL.md)明确要求两者完成，因此保留原顺序，直到用户选择是否调整。

建议顺序：**当前cap1原运行完成 → 原48组双锚对照 → 原cap3完整确认**。实验数量/顺序/数据/种子/步数/方法/权重/评分均不变，只改变GPU1三个阶段的排队顺序；cap3不取消。新CPU调度器已准备，7项顺序/互斥/完成门槛CPU检查通过，未启动任何GPU任务。

- 当前cap1 PID137966保持，不发送中断、不改其源文件或图像/ID/参数。进程结束和完整证明均核验后再运行后续工作。
- 48组仍调用原onevision_matched_task_domain.py --case，索引0..47不变，逐案例独立进程，保留原首阶段精确一致门槛。完整case/source/原F1证明缺失或错误就停止，不盲重试。
- cap3仍调用原onevision_full_input_repair.py --cap3，同一原权重/冻结21454QA/renderer；兼容推理断点沿原代码恢复，不拼接不同运行结果。
- 只用GPU1，不抢GPU0/基线卡；存在任意GPU1进程、显存或利用率未空闲时等待。
- 当前两个CPU协调服务的KillMode实测为control-group。**不得直接stop/disable--now**，以免终止其GPU子进程。若用户选择新顺序，先用drop-in改并核验KillMode=process，再只停止/停用旧CPU协调器，核验GPU PID/startticks未变，才启新CPU服务。新服务在没有记录用户选择的AUTHORIZATION.json时拒绝运行。
- 现有CPU报告观察器继续，唯一主指标Answer F1，无新映射。基线nohup不变。新dispatcher仅CPU验证，实际调度切换尚未进行。

代码：[onevision_causal_priority_queue.py](../../../analysis/onevision_causal_priority_queue.py)。CPU准备证明：[CPU_PREPARATION.json](CPU_PREPARATION.json)。这份方案不是GPU结果、不是新增方法，也不改变既有结论。
