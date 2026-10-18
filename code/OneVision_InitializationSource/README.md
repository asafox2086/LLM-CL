# 初始化来源控制：不把跨域复用当成正确答案

这是独立诊断，不是已经成功的新算法。每个条件只有一个rank8 LoRA，五个条件依次运行，并非同时增加五个专家。

同一目标Organ26TRAIN/2更新，比较冷启动、同域/跨域完整继承、同域/跨域仅继承A且B置零；最终每组均测原507题。两个历史源各6个唯一TRAIN QA、重复两轮，源成本另计。A-only在初始保持基座输出，真实32题NLL/答案须与冷启动精确一致。

[冻结协议](../../summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1/PROTOCOL.md) · [驱动](../../analysis/onevision_initialization_source_controls.py) · [唯一当前队列](../../summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1/queue/README.md) · [原始Answer F1表](../../summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1/VERIFIED_RESULTS.md)。

完整继承同时改变初始函数及首步A梯度可达性；同域/跨域仍含不同内容与题型，不推导普遍领域结论。初始32与最终507不同群体，不相减当习得。源/目标A/B、原始梯度、loss/LR、逐题结果、CPU/GPU前后/纯TRAIN峰值和checkpoint均保存；结果缺失留空，不捏造提升。

[只读TRAIN参数审计](../../summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1/TRAIN_PARAMETER_AUDIT_PLAN.md)由[独立CPU工具](../../analysis/onevision_initialization_source_geometry.py)完成，报告源A变化幅度、行空间主角、实际deltaW和两步A/B梯度；拒绝未封存状态，不修改训练或选择方法。合成实现检查不代表本模型收益。

[六栏与全部来源配对](../../summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1/CONTROL_OUTCOME.md)由[独立只读汇总器](../../analysis/onevision_source_controls_outcome.py)生成：仅在新完成证明后依次运行原评分核验、TRAIN几何核验、该汇总器。它汇总全部可用两两对比及真实内存，不筛选赢家，不修改任何训练／调度源码；没有新增GPU或轮询服务。
