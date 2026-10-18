# 当前方法：完整功能继承＋TRAIN质量调用

这是正在实验的两模块方案，不是已发表算法。它与`OneVision_TwoModule`目录所述的早期位移保护原型不同；原型及其未完成对照保留，不混合结果。

| 模块/部件 | 实际冻结源码 | 作用 |
| --- | --- | --- |
| 原专家池与向量分配 | [learned reuse](../../analysis/onevision_learned_reuse_trial.py)、[metric](../../analysis/onevision_learned_reuse_core.py) | 相同1313TRAIN/19阶段/rank8/Kmax3，TRAIN输入决定分配 |
| A：完整功能继承 | [warm start](../../analysis/onevision_warm_start_trial.py) | 必须新增时继承所选旧专家完整A/B，再独立训练；原生共同首任务精确门槛及19非处理变量门槛 |
| B：功能质量调用 | [quality core](../../analysis/onevision_functional_router_core.py)、[warm probe](../../analysis/onevision_warm_quality_probe.py) | 冻结A专家池，用43历史TRAIN/172次NLL前向拟合16参数头；推理仅输入图像/问题，不读金标，调用一个专家或基座 |
| 实际执行入口 | [metadata repair wrapper](../../analysis/onevision_identity_repair_worker.py)、[唯一队列](../../analysis/onevision_identity_repair_queue_v2.py) | 已验证的辅助行ID元数据修复；不改算法、QA或数学路径，不复用旧失败预测 |
| 只读结果发布 | [four-cell reporter](../../analysis/onevision_identity_repair_report.py) | 冷/温×几何/质量全部四格，不按诊断择优取消完整确认 |

**已经完成：** A的19阶段/93有效更新及500NLL/155生成；同155题AP31.025→35.025，医学F1 15.841→25.339，另四域答案不变、500条调用精确一致。仍低于同诊断SeqFT43.363。见[六栏结果、全部19任务及资源](../../summary/onevision_diagnostics_20261016/warm_start_20261018/first19_outcome/RESEARCH_FINDINGS.md)。

**当前B阶段：** GPU415408已通过原温池500NLL/155预测精确重载，已用43历史TRAIN/172次NLL前向完成16参数头的400次CPU更新（TRAIN校准CE 8.825→1.096），正在评测；这不是Answer F1，全部视图以[实时四格报告](../../summary/onevision_diagnostics_20261016/warm_quality_20261018/FOUR_CELL_RESULTS.md)及完整证明为准。尚无完整A+B收益，不能把原228题继承+9.388、A诊断+4和冷池B诊断+5.679相加。

[两个问题与协议](../../summary/onevision_diagnostics_20261016/warm_quality_20261018/PROBLEM_AND_METHOD.md) · [冷启动证明](../../summary/onevision_diagnostics_20261016/expert_cold_start_20261018/COLD_START_PROOF.md) · [当前执行与故障证据](../../summary/onevision_diagnostics_20261016/feature_identity_repair_20261018/README.md)。

只由唯一nohup调度在新GPU1依赖通过/空闲时启动，不从此索引手动并发启动。独立的18数据因子、原机制对照和全部完整21454确认保留。唯一原始Answer F1；TEST不选父专家、路由、容量、提示词或超参数；原生研究与正式V2分开。

多个LoRA允许但有界；3个池、投影、历史43QA/向量、CPU/GPU常驻、优化器、临时复制、额外前向均计费。16参数不是“总内存64字节”。未证完整五域宏平均超过SeqFT5pp或内存优于指定基线，不能填正式主表Ours成绩。模型、数据、梯度和checkpoint原文件只保留服务器，不入Git。
