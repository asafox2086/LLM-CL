# 当前唯一研究队列：GPU0/1择空，最多占3张卡

2026-10-18最新用户允许GPU0空闲/GPU1占用时改用GPU0；当前同nohup服务实际CPU436033/startticks16424972，旧434457仅CPU计划退役，全部受保护PID/startticks保全见[激活](ACTIVATION.json)。旧queue、v4/v3及更早历史所有者不恢复。

[10项设备CPU检查](DEVICE_CPU_CHECK.json)覆盖0/1空闲优先、第三卡上限、两卡都忙不准入、已有一个研究流不复制，以及真实两类函数0/1断言AST回滚完全一致；原47worker、21调度、8命令、37+66+36祖先检查均保留。只证明设备入口/CPU集成，不是GPU等价或性能。

[实际观察](RUNTIME_OBSERVATION.json)：我们只占GPU2/3；GPU0和GPU1均有UID1024进程存活，当前健康等待，尚无新GPU候选结果。以服务器此目录`queue_state.json`和实际/proc/NVML为准，不因util暂为0或无worker_pid重启/抢占。

新[设备入口](../../../../../analysis/onevision_research_device_entry.py)在torch导入前设置CUDA_VISIBLE_DEVICES，实际再次核对目标无进程、我们的占卡数<=3；只改旧目标函数首个CUDA编号断言的字面量，完整算法源码保持原字节。每个实际run新增physical_device_execution回执并绑定SHA。缓存候选与原数据因子可用0/1；其他既有任务保留原GPU1入口/全部条件，不热改。祖先AUTH的GPU1_only是原历史门禁字段，原值保留供来源校验；当前调度操作字段明确为research_allowed_GPUs=[0,1]和max_our_GPU_cards=3。

顺序仍是新候选19阶段/四格→剩余18数据因子→新两完整21454→全部原任务，原已就绪依赖和失败停止门槛保留；不能据TEST挑赢家、容量或取消全量确认。

[规则与测量](../../../../../rules/022_research_gpu_flex_20261018.md) · [不变的科学协议](../PROTOCOL.md) · [原始Answer F1主表](../VERIFIED_RESULTS.md)。
