# 八项已准入基线的只读结果发布器

入口`analysis/v2_eight_baseline_report.py`仅调整实验范围，复用冻结的原始V2逐题核验器（SHA256 `d7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a`）。原始训练源码和完成证明不改。旧发布器停止，新发布器独立目录与CPU服务，防止旧20组范围覆盖新主表。

当前要求8方法×2骨干＝16组完整结果；FOREVER为新增但未准入的比较，见[方法清单](../BASELINES.md)。主表中的FOREVER占位不算有效完成。旧证明按字节复制并继续绑定原核验器，不能因为改范围而重新选择最优TEST运行。只接受19阶段有效更新、全部74格原始Answer F1、最终21454 QA及最终checkpoint完整证明；Ours仍需独立完整确认。

运行时输出`VERIFIED_INDEX.json`与`observer_status.json`；历史20组目录保留，不是当前排队范围。范围调整的实际PID、源码SHA、队列归档与保护进程证据见[调整记录](../scope_revision_20261018/ACTIVATION.json)。
