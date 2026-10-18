# 当前唯一队列：函数补偿v2修复

> v2冷组完整507已通过精确等价，同域／跨域继续保留；本CPU30721已计划退役，由[反馈试验唯一队列](../../../answer_compatibility_20261018/v1/queue/README.md)CPU34096保全当前GPU33289接续全部v2条件，再后续试验。下文为v2修复准入记录，不重启旧所有者。

同一nohup服务恢复为CPU30721，入口`analysis/onevision_function_neutral_init_queue_v2.py`。实际状态以[queue_state.json](queue_state.json)、[ACTIVATION.json](ACTIVATION.json)、PID与日志为准；激活后worker30823已在GPU1加载模型并进入原32初始推理，完整507结果尚待。

已完成原32初始门槛与两步真实训练；[独立TRAIN阶段核验](../RUNNING_TRAIN_EQUIVALENCE.json)确认两步原始梯度文件SHA、最终A/B、冻结参考、loss和实际LR均精确等于原cold。当前是TRAIN阶段完成，**不是完整507评测完成或方法F1提升**。

v1因导入CPU审计模块清空CUDA_VISIBLE_DEVICES，在模型／训练前失败，CPU26045和worker29349已经退出；[原始失败证据](../V1_FAILURE_EVIDENCE.json)保留，不重启旧版。v2改为无环境副作用的同一梯度辅助函数，AST及实际原cold两步梯度解析精确相同，科学函数、数据、学习率与门槛不变。[实际子进程设备正负控制](../DEVICE_IMPORT_GATE.json)复现旧错误并验证新0/1设备设置保持，未初始化CUDA；它不代替模型运行。

顺序保持冷组实际精确等价／完整507→同域补偿507→跨域补偿507→全部原作业；原五格已完整，原因子case2也已自然结束。两机基线保全、公共完整确认继续后移保留。新反馈匹配仅准备，尚未准入，不能自行另起竞争GPU作业。

只用实际空闲GPU0／1中的一张，新机我们的作业最多3卡，保留其他用户进程。新v2源码已冻结，故障保存、新版本修复，不热改、盲重试、降低精确门槛或拼预测。

[科学协议](../PROTOCOL.md) · [原v1记录](../../v1/queue/README.md) · [授权来源](AUTHORIZATION.json)。
