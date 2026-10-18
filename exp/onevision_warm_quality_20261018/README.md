# 温启动池的质量调用消融

研究仅新GPU1。新入口`analysis/onevision_warm_quality_probe.py`，报告与协议见[两问题两模块](../../summary/onevision_diagnostics_20261016/warm_quality_20261018/PROBLEM_AND_METHOD.md)。

本目录预留独立`pilot/时间戳`和`full_quality_hard/时间戳`，避免覆盖旧冷池四视图。无需重复训练LoRA：依赖原温启动19阶段完整证明及stage19池。新入口已准备、CPU导入与结构检查通过，实际父池未完成，**已准入自动链但尚未执行GPU，不存在新方法成绩**。

pilot先精确重载温启动父池全部500NLL/155原预测，再用原43条TRAIN进行16参数头校准；四辅助视图全部报告。完整候选预定为quality_hard，新进程逐题精确重载后测全部21454，不按pilot挑选。原冷池两个完整候选及其他全部既定任务保持。

实际已由新[cold_quality_queue](../../summary/onevision_diagnostics_20261016/cold_quality_queue/README.md)准入；仅CPU协调器交接，原GPU/基线保全，失败保存证据、不盲重跑。不要手动执行示例命令与当前GPU1作业竞争。
