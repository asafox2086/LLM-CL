# 文献轻量改造：唯一CPU接续入口

2026-10-18最新调度：同nohup服务新CPU356616接续GPU348381自然完成，两个预定完整21454QA候选提前到四视图之后；原48/旧完整测试/其余全部任务保留，基线不中断，不按TEST挑赢家。18策略/4继承门槛通过，尚不代表收益。实时入口见[functional_confirmation_queue](../functional_confirmation_queue/README.md)。

2026-10-18已实际接入原nohup服务`llmcl-onevision-causal-priority.service`，入口`analysis/onevision_functional_router_queue.py`，CPU340143；只读结果观察器340149。前CPU336769以核验过的KillMode=process退役，原case18 GPU338128源码/进程均保留，等待自然完成及原完整SHA证明；没有证明就停止，不重启或拼接。当前进程以[实时状态](queue_state.json)为准，以上PID为激活时快照。

原基线、GPU/报告观察器startticks保全，详见[ACTIVATION](ACTIVATION.json)；没有杀GPU进程，没有热改原训练器。新worker和队列已冻结，修复另建版本。15项CPU机制/包装层检查、10项策略、3项继承安全检查通过，不是GPU收益。

接续：已运行归因→可用共同S1→原两组训练（已完成）→新固定池四视图→固定指令（已完成）→原48 SeqFT优先及全部机制→原相似度两组完整21454→新quality_hard/quality_soft各完整21454→原完整方法/cap3/初始/稀疏任务。所有原任务保留，两个新完整候选不依据诊断结果取舍。

只用新GPU1；有外来进程就等待，不借其他卡、不停基线。活动worker先自然完成，重复检查等待不当研究进展。GPU实际精确重载、干预、效果、内存还待运行。

[完整机制、论文适用边界与资源收费](../functional_router/PROTOCOL.md)、[自动结果](../functional_router/VERIFIED_RESULTS.md)。
