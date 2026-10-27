# TaskGradReuse：正式候选的方法定义

本候选检验：在固定单轮多模态连续学习中，逐任务用当前任务梯度选择新的低秩初始化方向，能否以共享小完整图得到有用的新习得与最终质量。它借鉴梯度对齐初始化思想，不是原 LoRA-GA 的复现，不预设大专家空间无用。本地局部描述性结果仅提供候选动机；原全课程 FIT 超时和局部正交门失败均保留，不能充当本次正式成功证据。

每个语言层的 q/v 投影沿原 SeqLoRA 使用 rank 8、alpha 32、scale 4、dropout 0.1，A/B 为 FP32。主优化器共有 540672 个坐标，原答案 CE、QA 归约、原训练顺序、GPU AdamW、LR/scheduler/scaler、单轮数据次数与生成评分沿原正式 runner，不在本模块重写。正式使用全部原 1313 TRAIN，而非探索的 1213 FIT；19 阶段共 93 次主更新。

在任务 t 开始时，先以 GPU FP32 计算 `W + 4 BA`，再转换回原 FP16 W，B 清零。仅取当前任务原 TRAIN 文件顺序前 `min(16, n_t)` 条，按 QA 等权平均原答案 CE 梯度。此校准处于 eval 模式、原 FP16 autocast，只临时把 48 个 q/v 的 dense W 提升为 FP32 并启用梯度；A/B 与其余参数冻结，无 optimizer step。校准后恢复原参数值、buffers、dtype、requires_grad、逐模块模式、Python/NumPy/CPU/CUDA RNG，以及原视觉缓存与计数。每阶段记录校准输入 ID 摘要、费用及状态恢复证据，不保留 G。

对各 GPU FP32 梯度矩阵直接调用 `torch.linalg.svd(G, full_matrices=False)`，使用默认 driver，令 `A = κ Vh[:8]`、`B = 0`，其中 `κ = 1/√3`。不做 QR、额外归一化或参考函数补偿。形状和有限性是硬检查；行范数与 Gram 误差如实记录，不再声称有限精度输出严格正交。B0 使新支路初始为零；后续 A/B 均由原主优化器学习。此处方向与坐标共同改变，Adam 不具有子空间内旋转不变性，不能将收益单独解释为纯 span 的因果作用，也不能将构造的 SVD 能量最大当成发现或 F1 证据。

阶段 1–18 结束不额外合并，完整 checkpoint 保留当前 A/B；下一阶段开始再合并。阶段 19 在原评价与保存之前合并并用普通 dense Linear 替换全部 q/v 适配层，清空 `method.layers` 旧引用。最终没有 A/B、G、专家银行或新增推理头。阶段合并使累计有效更新秩可增长，因此“活动 rank 8”不等于整个课程的累计 delta 始终 rank 8。

训练成本必须分别报告：主更新 540672 坐标；校准临时解冻 22020096 个 dense 坐标，梯度本身 88080384 bytes（84 MiB），另有 FP32 promotion、SVD 和临时张量费用。正式 19 阶段的额外校准为 222 次 QA backward，另加原预检校准和原公共预检费用。实际训练峰与耗时不能用主坐标数或既有部署测量代替。最终完整神经权重预期为原 dense 图 893675552 个系数，必须由真实保存图和资源测量确认。

原完整 checkpoint 保存整个 method、processor 和 RNG；原 runner 在恢复时不再次调用 bind_context。因此中间 method 只保留 CPU TRAIN 行、manifest 与 processor 引用供后续阶段使用，不保留 TEST 行、GPU 梯度或历史因子副本；原 loader 重绑 state_root。最终阶段移除这些 TRAIN 上下文引用。`changed_state` 包含累计改变的 dense q/v W 及尚未部署时的 A/B，避免误用普通 LoRA 的 AB-only 导出；正式恢复仍以完整 method checkpoint 为准。

原预检只调用一次 begin_stage(1)，随后执行原 19 个公共 QA case 与原 T19 probe；这不是完整任务 1 或 19 阶段科学训练。本模块记录 preflight 标记，不伪造完成阶段，不扩展原预检。当前仅本地源码实现与 AST/compile 检查，未由本模块启动、运输或执行任何模型、生成或评分。作者为 research_review；runner 构建和独立审阅由其他 agent 负责。
