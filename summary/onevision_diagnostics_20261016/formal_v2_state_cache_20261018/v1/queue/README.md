# 当前唯一研究CPU队列：公共协议两模块优先

**此处已是历史队列。** 最新用户要求先做简单的初始化来源诊断，正式完整确认后移不删除；当前唯一入口为[初始化来源队列](../../../initialization_source_controls_20261018/v1/queue/README.md)，CPU12470。旧10233/11463不恢复，原生GPU4363仍自然运行。以下保留原准入过程。

2026-10-18，同llmcl-onevision-causal-priority.service已接入不可热改的onevision_formal_cache_queue.py；CPU10233为实际新所有者，旧2468仅CPU计划退役。当前继承GPU1 worker4363/startticks22088，自然完成原生连续LoRA/缓存的四格评测后才开始公共协议候选，不中断、不拼接旧预测。

实际入口为本目录queue_state.json、ACTIVATION.json与FORMAL_INHERITED_WORKER/COMPLETE。原flex_queue已redirect；旧激活PID只作历史，不恢复。CPU调度7项新策略检查、原10设备/21策略/8命令及祖先检查保留；合成CPU测试不算模型性能。

顺序：保全当前GPU作业及原已就绪共享锚依赖→正式FP16公共输入构建/全部四格→预先固定的cache_geometry与cache_quality两个完整21454→返回全部原18数据因子、原生完整及其他机制作业。所有原作业保留；按配置一致性缩短反馈，不按未知TEST成绩删选。唯一研究流，0/1择空，我们在新机最多3卡；两机基线不停止。

公共模块主训练精确复用正式SeqFT已完成19阶段及其93次更新，原主训练成本仍计入；绝不拿旧研究AnyRes/BF16权重冒充公共参数。CPU从原19个约1.93GB checkpoint核验提取各540672参数的小状态并精确重载，不加载真实GPU。正式公共源、配置、划分、原始Answer F1不变。见[冻结协议](../PROTOCOL.md)、[CPU来源核验](../SOURCE_EXPORT_COMPLETE.json)、[正式表](../VERIFIED_RESULTS.md)。

GPU首先做官方前向/生成门槛，原正式SeqFT同155预测/金标/评分精确重载，失败保存并停止；两完整候选各新进程重载后测21454。只有封存完成证明才报告收益，最终FWT/BWT仍待各自真实稀疏格，不能借其他方法填写。
