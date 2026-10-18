最新用户设备授权：当前[flex_queue](v1/flex_queue/README.md)在GPU0/1择空、我们最多占3张卡；当前两张均他人占用。原queue记录均历史，不恢复。

最新：GPU候选v1已实现并准入，[协议](v1/PROTOCOL.md)、[代码索引](../../../code/OneVision_StateCache/README.md)、[实际队列](v1/queue/README.md)和[四格表](v1/VERIFIED_RESULTS.md)。当前实际他人GPU1进程存活，方法GPU未开始，F1/内存优势未证。下文20TRAIN筛查为先前依据，不冒充v1效果。

# 有界功能状态候选：当前只有TRAIN筛查

当前GPU方法尚未实现或准入。现有两模块完整21454仍未完成；不以此替换现有结果。

先固定[协议](PROTOCOL.md)，随后执行[CPU源码](../../../analysis/onevision_functional_cache_train_probe.py)，完整[证据](TRAIN_SCREEN.json)。20条原TRAIN探针，遵守原native到达顺序；Kmax1/3，当前状态必须保留，固定0.02nats额外状态惩罚。最终K3保留T12/T18/T19；覆盖NLL1.700186→1.632893只是TRAIN逐题最低值，不是F1。虚拟在线查询K1/K3=186/212题次，实际只读既有结果。

依据与下一步边界见[六栏发现](../warm_quality_20261018/calibration_audit/RESEARCH_FINDINGS.md)。训练仍固定1313QA；未来若实现，须计入父训练93次更新、历史QA/向量、缓存、临时候选、额外前向和真实CPU/GPU内存，不能由参数个数直接宣称轻量。
