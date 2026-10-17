# 可学习余弦路由与有界LoRA复用（研究试验）

用户提出的方案：模型依据输入判断是否已有可用LoRA，优先复用，必要时从基座新增。不是声称已发表的新算法，也未证明性能提升。

- 核心：[onevision_learned_reuse_core.py](../../analysis/onevision_learned_reuse_core.py)。
- 训练/完整确认：[onevision_learned_reuse_trial.py](../../analysis/onevision_learned_reuse_trial.py)。
- CPU单测：[check_onevision_learned_reuse.py](../../analysis/check_onevision_learned_reuse.py)。
- 协议/全部边界：[PROTOCOL](../../summary/onevision_diagnostics_20261016/learned_reuse/PROTOCOL.md)。
- 结果：[VERIFIED_RESULTS](../../summary/onevision_diagnostics_20261016/learned_reuse/VERIFIED_RESULTS.md)。

原冻结1%1313QA/19阶段；两组fixed/learned、Kmax3、同训练预算；图像+问题→1792维→32维→余弦。新增/复用与阈值只读TRAIN；单专家不声称已学会判别。所有有效组完整21454题确认。日志和19个pool/router快照在服务器`exp/onevision_learned_reuse_20261018/`，不入Git。
