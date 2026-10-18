# 多领域方法的轻量机制改造：固定池质量路由／软聚合

借鉴D-MoLE输入识别、MVP相关性聚合，结合ProgLoRA关于调用旧技能的视角；**不是上述原方法复现或已证明的新算法**。

在已完成的OneVision固定3专家池上，原LoRA和57344参数向量投影均冻结。原43条TRAIN QA测base/3专家，共172次NLL前向，训练16参数质量校准器；再做原硬路由、校准硬路由、几何软融合、校准软融合四视图。测试答案不参与校准或候选选择。

[协议、文献对应与边界](../../summary/onevision_diagnostics_20261016/functional_router/PROTOCOL.md)；[结果](../../summary/onevision_diagnostics_20261016/functional_router/VERIFIED_RESULTS.md)。[核心实现](../../analysis/onevision_functional_router_core.py)、[GPU worker](../../analysis/onevision_functional_router_probe.py)、[CPU检查](../../analysis/check_onevision_functional_router.py)。

零新增LoRA，额外64字节参数不是总内存优势。两个预定候选均完整21454确认，不能根据诊断择优，不能把500/155诊断当正式全测试。参数/逐题文件只保留在服务器，源码和小型证据入Git。

当前两模块组合的源码/运行索引见[ColdStartQuality](../OneVision_ColdStartQuality/README.md)。原冷池同155题质量硬调用AP31.025→36.704，仍低于SeqFT43.363；原完整确认在FinMME:5082因辅助行ID日志异常停止，失败保留。新的实际执行包装器已修复并通过GPU门槛，完整候选需从头确认，不把旧部分预测当结果；这不是新增算法模块。
