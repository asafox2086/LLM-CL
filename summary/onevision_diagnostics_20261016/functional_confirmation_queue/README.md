# 固定池：两个预定完整确认提前，原归因全部保留

最新完整四视图诊断已核验：同155题AP，原调用31.025→质量硬36.704／几何软35.996／质量软36.387，仍低于顺序43.363。完整21454候选已启动、尚未完成；正式V2主表仍只填已完成SeqFT/ER。见[五域主表格式与边界](../functional_router/MAIN_TABLE_DIAGNOSTIC.md)。

2026-10-18按用户已委托的GPU1研究调度执行；用户强调领域总分与整体收益，不要求所有细分能力全面提升。本次由代理决定将**两个早已固定的21454QA候选**提前到四视图诊断后，而不是等待其余35个严格控制和旧相似度完整确认。两个候选均测，不读取诊断分数决定顺序、入选或配置。

当前唯一CPU调度服务仍为`llmcl-onevision-causal-priority.service`，新CPU356616/startticks14682713，冻结入口[onevision_functional_confirmation_queue.py](../../../analysis/onevision_functional_confirmation_queue.py)。旧CPU340143用已核实`KillMode=process`计划退役。GPU348381/startticks14466517原四视图worker与新机2基线、原观察器的PID/startticks均保持；旧机4基线亦核实存活，没有GPU/source热改。

当前继承GPU必须自然结束并给出完整SHA证明；无证明即停，禁止因进程消失盲重试。新CPU18策略/4继承门槛通过，不代表模型收益。所有模型worker、数据、权重、环境、评分和原48控制保持原版。

## 实际顺序

1. 可用共同S1推理、原两组相似度训练（已完成者按原SHA跳过）。
2. 正在运行的完整四视图固定池诊断，自然结束并校验。
3. `quality_hard`、`quality_soft`按既定顺序分别新进程精确重载500NLL/155生成，再各测全部21454QA；无论诊断孰优，两者都测。
4. 原语言指令、全部48控制（保留原SeqFT优先顺序）、原两组相似度完整测试及所有原完整方法/cap3/初始/稀疏任务。

完整候选只改固定池调用/残差，不增加知识、LoRA或模型优化器。最终单点评测不能算完整FWT/BWT；研究nativeAnyRes与正式V2分别报告。领域总Answer F1、既有19任务AP与五域宏均值按原定义并列，资源和额外计算均计费。未完成不能宣称收益或目标完成。

[实时状态](queue_state.json) · [18策略检查](CPU_POLICY_CHECK.json) · [4继承检查](INHERITANCE_CPU_CHECK.json) · [授权依据](AUTHORIZATION.json) · [实际激活及进程保全](ACTIVATION.json) · [原科学协议](../functional_router/PROTOCOL.md) · [模型结果](../functional_router/VERIFIED_RESULTS.md)。
