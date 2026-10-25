# 1%基线容量调度：实际接管与统计口径

结论：旧两卡限制和OneVision SMo全Qwen屏障已移除；原两个GPU基线身份完整保留。不是重新训练、改基线或新科学成果。

- NEW保留GPU0不由本队列分配，1/2/3仅在原ownerless／free≥16000MiB／util<10条件下派发。实际该时点所有卡有第三方或原基线进程，未对任何GPU进程发信号。
- CL-MoE completed16/active17，SMoLoRA initial_evaluation；PAS/PaRSP ready_reload，OneVision SMoLoRA/OSFT/SpaRTA waiting。新CPU1221743 active/running且原身份adoption已见证，空槽将接续原流程。
- OLD此时4卡空，七项OneVision已sealed；缺的SMo需要NEW原生BF16，不能强制2080Ti改方法。旧卡用于独立已评审研究，不重复已完成基线。
- 原FOREVER waiter不动，保其GPU2条件；只1%可派发，旧full自动开放CPU分支已停并disable，不触发100%队列。

## 参数记录

原8个已核验stage19的完整系数、必需状态、dtype字节、checkpoint文件MiB与来源已在[统计表](../../final_weight_reduction_20261024/checkpoint_weight_storage_v1/FINDINGS.md)。原有数值不改、不重提取。新组合须先完成正式核验，随后只对新增stage19运行已有元数据统计器，加入README；不得用中间阶段或设计估计冒充最终值。

完整模型参数／必要持久状态、原dtype权重bytes、checkpoint容器bytes、可训练坐标及梯度/optimizer状态分列；这些不能替代实际GPU峰。别名共享storage去重，普通buffers/保护历史等原排除口径保留。

## 证据与范围

[部署receipt](DEPLOYMENT_RECEIPT.json) · [前态与精确身份](DEPLOYMENT_STARTED.json) · [新机接管快照](NEW_AFTER.json) · [旧机快照](OLD_AFTER.json) · [源码自审及作者披露](SOURCE_REVIEW.json) · [部署结果复核](DEPLOYMENT_REVIEW.json)

调度器复用冻结原pipeline launch/idle/sources/write，原25科学py、method/dtype/manifest/训练参数不变。只修改两个NEW host的设备JSON；不是改变FOREVER已冻结策略。真实GPU启动仍由原预检／重载／训练流，首次失败保留失败不盲重试。主服务KillMode=process、Restart=no。

只有后续完整proof才更新正式表；本次正式8/16、FOREVER和全部存储列原字节保留。
