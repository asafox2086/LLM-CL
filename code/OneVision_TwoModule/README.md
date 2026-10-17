# OneVision两模块候选：自己的实验原型

此目录是源码索引；正式基线作者源码不改。当前源文件排队后冻结，保持路径便于SHA追溯。

| 文件 | 用途 |
| --- | --- |
| [core](../../analysis/onevision_two_module_core.py) | A：分离旧基底与功能；B：有限位移TRAIN功能检查 |
| [CPU检查](../../analysis/check_onevision_two_module_core.py) | 8个实际因子/受控机制门槛 |
| [GPU试验](../../analysis/onevision_two_module_pilot.py) | 12组固定原型，原训练器精确门槛/原始梯度/参数/资源 |
| [原型队列旧版本](../../analysis/onevision_two_module_queue.py) | 保留源哈希；其CPU调度已由新版本接管 |
| [只读报告](../../analysis/onevision_two_module_report.py) | 唯一Answer F1；拒绝零更新的有效成绩声明 |
| [完整epoch引擎](../../analysis/onevision_two_module_engine.py) | 原AdamW轨迹、实际位移保护、优化器/RNG回滚、每步完整证据 |
| [完整19阶段](../../analysis/onevision_two_module_fullstream.py) | 四组共享1313TRAIN，逐阶段原轨迹精确门槛与可信恢复 |
| [全测试确认](../../analysis/onevision_two_module_confirmation.py) | 每组21454最终预测、独立进程复现；稀疏状态计算完整FWT/BWT |
| [当前GPU1队列](../../analysis/onevision_two_module_fullqueue.py) | 原型→完整训练/最终确认优先，保留原48控制/cap3 |
| [完整结果报告](../../analysis/onevision_two_module_fullreport.py) | 诊断155与完整21454严格分表，无结果显示待测 |

[原型协议](../../summary/onevision_diagnostics_20261016/two_module_prototype/PROTOCOL.md)、[相关工作](../../summary/onevision_diagnostics_20261016/two_module_prototype/PRIOR_ART.md)、[实际结果](../../summary/onevision_diagnostics_20261016/two_module_prototype/VERIFIED_PILOT_RESULTS.md)。CPU门槛通过，GPU效果尚未产生；不是已发表算法，不填入正式主表Ours成绩。

完整代码已实现并排队：[完整协议与成本](../../summary/onevision_diagnostics_20261016/two_module_fullstream/PROTOCOL.md)、[完整结果入口](../../summary/onevision_diagnostics_20261016/two_module_fullstream/VERIFIED_RESULTS.md)。四组均有完整最终确认，不按TEST挑候选；当前无新GPU效果。
