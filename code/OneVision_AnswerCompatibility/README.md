# 轻量能力匹配原型：已准入，GPU结果待测

研究载体为原nativeAnyRes 0.5B连续LoRA轨迹及最多3个功能状态。候选两个模块是连续习得／有界复用，和以原TRAIN输出质量反馈学习的匹配距离；不预先限定最终方法必须LoRA或线性空间。

- [预定协议](../../summary/onevision_diagnostics_20261016/answer_compatibility_20261018/v1/PROTOCOL.md)：原20TRAIN×3状态60条生成，1792参数CPU距离头，两个新条件均保留完整确认。
- [核心](../../analysis/onevision_answer_compatibility_core.py)、[独立GPU驱动](../../analysis/onevision_answer_compatibility_trial.py)、[设备入口](../../analysis/onevision_answer_compatibility_entry.py)。
- [原调用TRAIN偏差](../../summary/onevision_diagnostics_20261016/train_calling_fit_20261018/RESEARCH_FINDINGS.md)：8/20未选最低NLL，仅为训练目标拟合证据，非TEST性能原因。

当前核心CPU合成检查、实际20TRAIN图像遮罩、实际新旧分支及9项调度检查已通过，**已准入唯一队列，但尚无真实60条反馈或F1提升**。[当前队列](../../summary/onevision_diagnostics_20261016/answer_compatibility_20261018/v1/queue/README.md)先保留全部函数补偿作业和基线，再执行新试验；不把源93更新、3状态和额外生成成本隐去。正式主表保持空白，直到相同协议完整结果存在。
