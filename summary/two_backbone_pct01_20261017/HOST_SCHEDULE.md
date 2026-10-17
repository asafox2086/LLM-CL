# 用户最新调度

当前Qwen全量SeqFT继续新GPU3；完成19阶段、最终评测、最终参数SHA及进程退出后，由llmcl-baseline-host-handoff.service切换：新GPU2/3用于24GB的1%任务，后续Qwen全量交给旧GPU2（等待原PASs作业实际结束）。新GPU0继续保留、新GPU1继续痛点探究。

旧GPU1用于新增LLaVA-OneVision-0.5B的1%正式基线准入；旧GPU0上的PaRSP新预检、旧GPU2的PASs、旧GPU3的FV不中断。两种backbone的代码、日志、参数和报告分目录；不混用预测或给同一张卡启动冲突进程。

旧GPU无法容纳的全量方法由原队列回退24GB等待，不能声称仅训练数据量变化就一定能放进11GB。等待状态必须持续检查，不缩图、删数据、改作者方法或借用保留卡凑并发。
