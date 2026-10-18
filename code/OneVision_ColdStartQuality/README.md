# 冷启动功能继承＋TRAIN质量调用

本研究两模块改造，尚无组合GPU结果，不是已发表方法或正式基线成绩。

- 问题一与已验证局部收益：[冷启动证明过程](../../summary/onevision_diagnostics_20261016/expert_cold_start_20261018/COLD_START_PROOF.md)。
- 问题二、两模块与四格消融：[方法协议](../../summary/onevision_diagnostics_20261016/warm_quality_20261018/PROBLEM_AND_METHOD.md)。
- A：沿用冻结[温启动入口](../../analysis/onevision_warm_start_trial.py)，同19阶段只改变新增专家完整A/B初始化。
- B：新[组合入口](../../analysis/onevision_warm_quality_probe.py)，复用A封存池，仅用原43条TRAIN拟合16参数质量头；不再训练LoRA，不增加专家。
- 新入口CPU导入/AST检查通过，实际温启动父池未完成，已准入自动链、仍待实际warm19父状态；原图文精确重载及完整21454效果仍待。

运行只由唯一研究调度在新GPU1空闲且依赖通过后准入，不从这里手动并发启动。原500NLL/155生成只是诊断，最终五域完整测试须独立确认；唯一原始Answer F1，禁止TEST挑路由或参数。内存、43条历史TRAIN及172次校准前向全部计费。
