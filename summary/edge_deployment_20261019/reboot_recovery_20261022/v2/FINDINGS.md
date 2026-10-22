# 第二次核实重启后的正式队列接续

2026-10-22 03:34UTC，boot ID变为f801c9c5-7362-47c3-afc3-1454c1e9d29d，uptime约367秒；此前6262/6264/6266/5774/5779身份失效。不能把消失的旧PID当仍在跑，也不能因此重复已完整方法。

原正式native服务自动接续CL-MoE和SMoLoRA，当前训练4962/4874。只停止KillMode=process的CPU调度器2572，保全这两个已在执行的合法孩子；PASs-MoE与PaRSP被重启中断，原config的checkpoint_all_stages=false且checkpoint目录为空，不能做权重续训或拼旧预测。原run、部分结果和额外历史成本保留后，仅这两项按原条件新鲜重排。CPU原入口接续为5963；GPU训练未发中断信号，作者源码/数值/数据/评分不改。

原只读七方法发布器恢复5965，FOREVER Qwen CPU入口恢复5967。FOREVER原state仍worker=null且没有训练文件，旧OWNER已失效，文件在独立private归档后按原入口等待GPU2；没有训练盲重试。旧机只读核实六方法仍完整，SMo保持waiting_native_BF16_on_newGPU3，不恢复旧方法/PID。

这些是带时间的恢复证据，后续须核实真实进程和sealed proof；不把本PID写成永久live，不把重启恢复当科研收益。正式完整仍8/16，无新sealed不刷新主表。ShortFixed v1目录故障发生在重启前，两事件分开，研究修复另见独立v2；端侧只背景，目标未达成。
