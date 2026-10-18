# 实际启动与重启接续

2026-10-18新服务器18:45:12启动；CPU调度随原systemd/nohup服务恢复为2468，实际研究worker4363在GPU1开始本候选第一条独立运行，见BOOT_RECOVERY_20261018.json。旧ACTIVATION中的PID是历史证据，不代表当前存活。

原先本候选没有GPU训练，故这次是首次开始，未从旧失败结果接训练。实际原生图像前向六门槛（含Fin5082）和辅助特征ID修复门槛通过；初始更新学习率为正、梯度有限。19阶段逐层权重精确等价、四格生成、完整21454和实际资源仍须各自完成证明，启动不当成收益。

正式Qwen SeqFT由原框架同run的--resume接续。CL-MoE旧运行未完成任何训练阶段，无有效阶段checkpoint；重启后原CPU框架将其标为failed_training。一次性recover_clmoe_boot_20261018.py保留中断run和全部证据，将CL恢复waiting，重新经过原预检、重载、独立新run；不拼旧初始预测。仅短暂停启KillMode=process的CPU调度，SeqFT、SMo预检和研究GPU进程的PID/startticks全部保全，源码/超参/划分不变。见summary/two_backbone_pct01_20261017/boot_recovery_20261018。原CL日志包含NUL，具体损坏成因未确定。

CPU_PIPELINE_SMOKE_OLD_ISOLATED.json是另一个CPU集成测试：旧机/tmp独立代码根、单线程、合成数据与模型替身，执行真实driver的19阶段保存、四格及两条各21454的评测/完成判定调用链。未加载真实模型或公开QA，无GPU操作；不提供真实梯度、F1、显存或GPU等价证据。新机同CPU检查在网络中断/重启前未得到完成证明，不将其算通过；不重复启动它。真实GPU运行使用原6线程，未因CPU测试改变。
