# 五领域五图文任务持续学习

当前用户确认版本：five_domain_cl_20261016_full_1pct。来源与样例见根 README，文献使用依据见 data/MULTIDOMAIN_DATASET_SURVEY_20261016.md。

T1 TextVQA、T2 SLAKE、T3 RSVQA-LR、T4 MME-Finance、T5 AQUA。五个来源，全部图像＋原始文字问题；保留原题型。完整训练172407、1%训练1728、共同完整测试22246，两个条件的测试文件 SHA256 相同。

基座Qwen2-VL-2B-Instruct，seed42、一轮、无验证。GPU0保留；GPU3全量，GPU1/2为1%训练组，空闲才准入，不停止其他人的进程。九方法保持作者机制；Hyper-LLaVA保持作者10专家容量但仅训练五个阶段，不能暗改为五专家后冒称作者配置。

每方法初始及五阶段后评测全部五任务，保存6x5矩阵、逐题预测、EM正确/总数、Token F1、ROUGE-L、AP、ForgetRate、FWT、BWT。所有阶段保存loss、梯度和具体CPU/GPU资源占用。最终导出在训练后内存测量之后进行。

不存阶段/优化器checkpoint。每方法全部任务训练及评测结束后，保存一次final_parameters/：完整最终模型状态、方法/路由状态、处理器/模型配置、基座文件哈希、任务清单、源码及所有文件的大小和SHA256。最终权重仅保留服务器。加载需要重建自定义方法架构，后续评定前必须验证新进程重载的输出等价。

原始划分：TextVQA train/validation，SLAKE train/test（双语），RSVQA-LR作者active掩码train/test，AQUA train/test（知识与视觉题一起输入画作）。MME-Finance英文原始1171QA按图表组研究holdout，seed42、80/20；不是作者官方划分。1%按原题型比例分层，最大余数分配ceil(1%N)，再固定哈希选ID。

代码：prepare_data.py、run.py、pipeline.py、resource_monitor.py、final_parameters.py。执行策略execution_policy.json；进度summary/five_domain_cl_20261016_full_1pct/progress.md。
