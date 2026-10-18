# 历史研究调度：CPU366824已退役

当前入口已改为[causal_feedback_queue](../causal_feedback_queue/README.md)，新CPU376997。旧状态文件redirect，不恢复366824；下文是此前激活记录。

2026-10-18：同nohup服务`llmcl-onevision-causal-priority.service`现用冻结`onevision_cold_data_queue.py`，CPU366824/startticks14902957。它是原冻结warm队列循环的CPU准入适配，授权文件同时记录实际wrapper和不可变祖先的SHA，未修改旧文件。

旧CPU363113仅计划退役KillMode=process，原完整评测GPU358091/startticks14710479、所有新机基线/观察器与其他实际GPU进程的PID/startticks精确保全；不停止他人进程。[ACTIVATION](ACTIVATION.json)。

顺序：原可用共同S1/相似度训练/四视图 → **两个原功能调用完整21454** → **原温启动19阶段训练** → **新增18个独特TRAIN量×初始化轨迹，各保留1/4步终点** → 原可学习冷启动21454/温启动21454 → 原cue/严格48/固定相似度完整/其他所有原方法。全部保留，不按TEST得分改参数或取消条件。

唯一实时入口：[queue_state.json](queue_state.json)。旧warm_start_queue状态redirect，不恢复363113或更旧CPU所有者。142项CPU设计、25项新增策略、28项原策略及6项实际继承检查通过；**新增GPU数据量实验仍0/18待测**，不能称已有效果。GPU1外国存活进程/占用门槛始终阻止新准入；缺完整证明即故障而非重启许可。

[研究协议](../cold_data_factorial_20261018/PROTOCOL.md) · [授权与源SHA](AUTHORIZATION.json) · [CPU策略](CPU_POLICY_CHECK.json) · [继承检查](INHERITANCE_CPU_CHECK.json)。
