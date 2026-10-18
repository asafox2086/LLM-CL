# 同预算反馈覆盖对照

从原反馈20锚仅覆盖11／19原生任务、学习距离未改变实际调用的证据出发，只改变TRAIN锚分配。主模型1313TRAIN／19阶段／93历史更新、3个功能状态、20锚／60生成／1792维／40步均相同。原生任务标签仅用于TRAIN抽样，不进入路由。

- [冻结协议及状态边界](../../summary/onevision_diagnostics_20261016/feedback_coverage_20261018/v1/PROTOCOL.md)
- [采样与真实19任务覆盖核验](../../analysis/onevision_feedback_coverage_plan.py)
- [独立worker](../../analysis/onevision_feedback_coverage_trial.py)、[设备入口](../../analysis/onevision_feedback_coverage_entry.py)、[只读报告](../../analysis/onevision_feedback_coverage_report.py)
- [CPU真实实现检查](../../analysis/onevision_feedback_coverage_cpu_check.py)、[唯一调度接续](../../analysis/onevision_feedback_coverage_queue.py)

复用原`FeedbackExperiment`与原距离拟合核心，不热改旧运行源。两新条件均完整诊断／21454确认，缺完整证明不填分数；不把同预算采样控制叫作已证明的新算法。模型／数据／checkpoint仅服务器exp，不进Git。
