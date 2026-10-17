# OneVision两模块候选：自己的实验原型

此目录是源码索引；正式基线作者源码不改。当前源文件排队后冻结，保持路径便于SHA追溯。

| 文件 | 用途 |
| --- | --- |
| [core](../../analysis/onevision_two_module_core.py) | A：分离旧基底与功能；B：有限位移TRAIN功能检查 |
| [CPU检查](../../analysis/check_onevision_two_module_core.py) | 8个实际因子/受控机制门槛 |
| [GPU试验](../../analysis/onevision_two_module_pilot.py) | 12组固定原型，原训练器精确门槛/原始梯度/参数/资源 |
| [统一队列](../../analysis/onevision_two_module_queue.py) | 仅GPU1，当前确认→48原控制→12原型→cap3 |
| [只读报告](../../analysis/onevision_two_module_report.py) | 唯一Answer F1；拒绝零更新的有效成绩声明 |

[原型协议](../../summary/onevision_diagnostics_20261016/two_module_prototype/PROTOCOL.md)、[相关工作](../../summary/onevision_diagnostics_20261016/two_module_prototype/PRIOR_ART.md)、[实际结果](../../summary/onevision_diagnostics_20261016/two_module_prototype/VERIFIED_PILOT_RESULTS.md)。CPU门槛通过，GPU效果尚未产生；不是已发表算法，不填入正式主表Ours成绩。
