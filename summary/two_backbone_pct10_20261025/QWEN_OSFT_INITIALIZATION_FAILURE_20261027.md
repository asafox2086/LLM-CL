# Qwen 10% OSFT：初始化预检显存溢出

原队列已记录failed_preflight，exit1。错误发生在OSFTMethod构造的create_svd_dict调用torch.linalg.svd时，尚未进入正式训练；没有训练events文件或最终S19 checkpoint。

日志报告：申请62 MiB失败，设备容量10.75 GiB，当时空闲34.75 MiB，本进程约10.71 GiB，其中PyTorch已分配10.16 GiB。原队列绑定物理GPU1；日志CUDA0是进程可见编号，不是另一张物理卡。

这表明当前固定骨干、方法实现和硬件条件下，初始化的分解及参数状态成本已经无法容纳。它不是推理峰值、全TEST内存指标、质量零分、数据不足导致学不会，亦不能外推为OSFT在所有硬件上不可用。按设置冻结原则保留失败，不改变rank、dtype或分解设置救跑。

[只读证据](QWEN_OSFT_INITIALIZATION_FAILURE_20261027.json)记录原run、exit及日志SHA。原config.json尚未生成，因此明确为空；不能编造完整运行配置收据。原runner快照、请求和失败日志仍留原路径。本次0模型、0评分。
