# 双骨干正式1%对照（共享框架）

当前代码：exp/two_backbone_native_cl_20261017。表格入口：[五领域最终成绩＋AP/FWT/BWT，按骨干分块](../two_backbone_pct01_20261017/MAIN_TABLE.md)。两种骨干共用25份Python源码；[逐项公平协议](FAIRNESS_PROTOCOL.json)。所有主表单元格只从新版本完整有效结果产生。

38组真实AdamW/HF调度CPU检查均通过；旧1%有9个全零lr阶段，新版均有正更新和参数变化。实际小OneVision模型验证了注入与核心decoder共享、完整方法图pickle重载，双机25份源码SHA一致。这些CPU检查不是GPU准入完成；各方法仍须通过19任务、独立进程checkpoint重载及有限反向。

旧机先跑OneVision，新机Qwen；新GPU0保留、新GPU1研究。受影响的旧Qwen四个活动1%作业已SIGINT保存证据并隔离，旧已完成SeqFT/ER亦不作为有效正式结果。新机当前全量作业与GPU1研究startticks保持；旧机全量搬迁安排按最新用户指令取消。

OneVision早期FP16数值控制：同权重／RNG／loss下scale1024梯度非有限、scale1有限；scale1完成19任务真实更新。共享新框架将scale1统一用于两种骨干，不能只改OneVision。新框架进一步逐任务核验原生官方前向与4token生成完全等价；实际GPUproof完成后再收录。

研究onevision_diagnostics_20261016仍是单独可追溯的因果诊断，使用已验证正lr；500／155等子集与新正式完整测试不混合，不将不同学习率、预算或提示的旧诊断当成同设置方法主表。

硬件协调：analysis/two_backbone_hardware_queue.py，nohup服务llmcl-two-backbone-hardware-queue.service只用CPU等待Qwen1%优先任务退出，随后为原生BF16的OneVision SMoLoRA开放新GPU3；该方法完成且Qwen无失败待审时才开放后续Qwen全量。绝不占新GPU0／1、不抢运行卡、不盲目复跑失败方法。
