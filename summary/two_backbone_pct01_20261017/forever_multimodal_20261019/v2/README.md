> v2 OneVision全部图文／回放／保存／重载门槛已通过。当前只续Qwen，旧机实时入口为 `v2/qwen_cpu_resume/state.json`，CPU1189315／Qwen1189340是最近身份，先实查；原gate_attempt只保留OneVision完成与CPU空闲采样断言，不能重启它。训练源／原空闲守卫均未改。正式F1仍待。

# FOREVER 图文接入 v2：修复 AMP 封装保存

v1 的实际 LLaVA 图文反向与回放已成功，整项门槛在保存 checkpoint 时失败。HF 的 AMP forward 包装器明确禁止序列化，必须使用 `unwrap_model(..., keep_fp32_wrapper=False)`；默认参数仍保留该包装，不能修复这个错误。它是我们的接入错误，不是 FOREVER 性能失败。原源码、失败日志、36 次回放更新及原始梯度、模型残片均保留，不从残片恢复或拼接。

v2 独立放在 `exp/two_backbone_forever_20261019_v2`。所有训练与优化器语句 AST 与 v1 完全相同；仅在回放完成后、共同保存前移除 AMP 包装。检查全部参数的对象、存储、版本及训练参数摘要不变；同一条旧 TRAIN 的 NLL 前后必须精确相同，并恢复 RNG。每次额外两次 TRAIN 前向计费，无额外训练更新。

CPU 正控制复现相同序列化错误；移除封装后参数、RNG、显式 autocast 输出、cloudpickle 重载输出均精确相同。v2 实际 HF CPU 的 8 次更新、梯度和最终参数仍与 v1 及独立作者类相同。这些不是模型 F1 结果。

原 v1 预检额外成本：19 个共同 TRAIN 单题更新＋1 个 T02 更新＋36 次历史回放更新，回放实际 576 次 QA 曝光、35 次正 LR／1 次作者 warmup 零步。请求两轮并不意味着实际 606 次曝光；真实 HF 尾批记录保留。v2 必须从新模型重新跑预检，不能把 v1 的权重或预测当作新版本结果。

仅授权旧 GPU0 顺序执行 LLaVA／Qwen 接入门槛及另进程重载。实时入口在旧机本目录 `gate_attempt/state.json`；新机只存快照。原五条正式基线和新机研究不变。完整规范见 [GPU_GATE_PROTOCOL.md](GPU_GATE_PROTOCOL.md)，source/协议首次 GPU 启动后冻结。缺全部门槛和 `FORMAL_ADMISSION.json` 时，独立正式验证器必须拒绝发布；FOREVER 正式表分数继续留空。
