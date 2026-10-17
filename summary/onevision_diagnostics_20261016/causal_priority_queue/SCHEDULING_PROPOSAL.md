> 当前已由[两模块统一队列](../two_module_priority_queue/ACTIVATION.json)接管：cap1→48原控制→12新原型→cap3。以下保留前一版切换历史，不重启旧CPU协调器。

# 核心归因优先的排队调整（已启用，2026-10-17）

核心缺口是同任务重复/同域新任务/跨域新任务的严格配对，现有队列让其等待cap1和cap3两次完整确认。[当前双锚协议](../../../analysis/ONEVISION_MATCHED_TASK_DOMAIN_PROTOCOL.md)明确要求两者完成，原顺序作为历史记录保留。用户现明确委托“这个gpu1的方案你自己定，我只要结论”，本次由代理选择并启用核心归因优先顺序。

当前顺序：**当前cap1原运行完成 → 原48组双锚对照 → 原cap3完整确认**。实验数量/顺序/数据/种子/步数/方法/权重/评分均不变，只改变GPU1三个阶段的排队顺序；cap3不取消。新CPU调度器已通过7项顺序/互斥/完成门槛检查并启用；尚未新增GPU任务，继续等待原cap1进程自然完成。

- 当前cap1 PID137966保持，不发送中断、不改其源文件或图像/ID/参数。进程结束和完整证明均核验后再运行后续工作。
- 48组仍调用原onevision_matched_task_domain.py --case，索引0..47不变，逐案例独立进程，保留原首阶段精确一致门槛。完整case/source/原F1证明缺失或错误就停止，不盲重试。
- cap3仍调用原onevision_full_input_repair.py --cap3，同一原权重/冻结21454QA/renderer；兼容推理断点沿原代码恢复，不拼接不同运行结果。
- 只用GPU1，不抢GPU0/基线卡；存在任意GPU1进程、显存或利用率未空闲时等待。
- 切换前先把旧服务KillMode设为process并实测确认，然后只停用旧CPU协调器。GPU1原PID137966/startticks5598716和新机两个基线GPU PID/startticks均保持；原GPU1父PID退出后继续独立推理。旧服务禁止重启；重启机器后由新服务接续兼容推理或完整新案例。见[切换证据](ACTIVATION.json)。
- 现有CPU报告观察器继续，唯一主指标Answer F1，无新映射。基线nohup不变。新服务llmcl-onevision-causal-priority.service已active/enabled，CPU协调PID见[队列实时状态](queue_state.json)；GPU尚在原cap1推理，48组未开跑。

代码：[onevision_causal_priority_queue.py](../../../analysis/onevision_causal_priority_queue.py)。CPU准备证明：[CPU_PREPARATION.json](CPU_PREPARATION.json)。这份方案不是GPU结果、不是新增方法，也不改变既有结论。

授权来自[用户明确委托](AUTHORIZATION.json)，不是从goal续跑或超时推断。原训练函数、48组索引顺序、stage1精确门槛、ID/QA/参数、完整21454确认均不改；服务配置见[SERVICE_UNIT.service](SERVICE_UNIT.service)。
