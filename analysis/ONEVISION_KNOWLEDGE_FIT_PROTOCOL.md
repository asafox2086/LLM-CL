# 知识写入的操作核验：固定8／64步（2026-10-17）

目标是区分“事实没有被成功写入”与“写入事实后仍不能改善目标任务”。这不是新持续学习方法，也不修改原1313条TRAIN、原事实池、提示或原评分。既有知识LoRA的原16事实严格回忆均0/16；固定前缀检查后仍0/16，因此先做干预生效检查，而非据此否定知识机制。

## 四条固定轨迹

1. 旧自然共享LoRA冻结，新增rank1知识残差：67584可训练参数，正确16事实。
2. 原rank8共享LoRA继续训练：540672可训练参数，正确16事实。
3. 同一原rank8共享LoRA继续训练：540672可训练参数，对应错配16事实。
4. 旧自然共享LoRA冻结，新增rank1知识残差：67584可训练参数，对应错配16事实。

共享起点由原知识实验的引用及SHA固定，不按测试分数选。保留原知识池SHA；错配沿用原池同subject/relation、不同value的合成错误目标，不宣称其为事实。无需另找测试相关知识或改事实池。

每轨迹独立GPU进程、seed42、确定性默认SDPA。AdamW lr1e-4，weight_decay0，clip1，每8条事实一次更新；固定训练32遍／64步／512条辅助曝光。按原`42:epoch:id`顺序，完整运行，不按结果早停。0／8／64保存状态，8步是64步同一轨迹前缀，对应64曝光；不是独立seed重复，也不是沿用旧默认后端检查点。

两个知识rank与错配对照是预先声明的机制控制，不按测试选rank、预算、路由或容量。64步用于检验更充分拟合，不预设一定能写入成功。辅助512曝光是额外训练成本，不能冒充原1epoch／1313QA正式CL。

预算元数据v2更正（在首个GPU病例启动之前）：原文把64步曝光误写为1024，正确算术为64×8=32×16=512。停止等待中的CPU协调器后才更正源码／文档并重新检查；训练循环、步数、顺序和数据均不变。旧CPU证明和修正记录分别保存在`knowledge_fit/CPU_PLAN_CHECK_v1_incorrect_exposure_label.json`与`knowledge_fit/BUDGET_METADATA_CORRECTION.json`。v2添加跨字段算术约束，不能将旧元数据标签当真实训练量。

对照覆盖v3补充（同样在任何新GPU病例／8步或64步结果之前）：增加rank1_permuted，让tiny-rank与rank8均有同架构、同预算的错配事实负对照。原三条训练设置不变；v2源码及CPU证明保留，记录见`knowledge_fit/MATCHED_CONTROL_AMENDMENT.json`。只停止CPU等待协调器后改动，不热改rural等GPU作业；所有四条均执行，不按临床分数选择。

## 测量与隔离

- 在0、8、64处，用同一权重评原16事实训练问法及固定改写；这是回忆／拟合检查，不是独立医学泛化。正确事实参照与自身训练目标参照分别保存；正确条件下它们重复，不扩充独立样本量。错配自身目标的分数只表示记住合成错误，不是事实准确率。
- 同时固定医学100QA与原生KG267QA，两者重叠12题，禁止合并为独立367题。保留原生成、原EM/F1/ROUGE-L及NLL；需要时另作统一固定前缀敏感性审计，不能改主评分。
- 端点评测前后保存并恢复Python／NumPy／Torch CPU／可见CUDA RNG与train/eval模式。评测不能消耗后续训练的dropout随机流。
- 每步保存pre-update loss、真实lr、梯度范数及原始完整可训练梯度；保存0／8／64全部适配参数与固定共享／源码／基座引用。rank1条件验证共享A/B逐元素未变。
- 单独记录0–8、8–64两个训练窗口的内存前后和GPU峰值；完整进程内存包含端点推理和导出，不能充当纯训练峰值。日志和大权重仅在服务器，Git只放代码／报告／汇总证据。

## 检查与执行

`analysis/onevision_knowledge_fit_probe.py --check`验证原事实／共享适配器SHA、16事实和错配ID／value约束、训练顺序、预定预算；用同一个CPU小型dropout模型实际调用旧训练器和新训练器，16步权重逐元素一致，并包含8步后的随机消费评测回调。该检查**不是OneVision GPU或临床效果验证**。首次CPU stub缺少target_tokens字段的失败已单独保留，修正后通过，原训练器不变。

nohup服务`llmcl-onevision-knowledge-fit.service`等待已有全量确认与rural诊断服务结束，且GPU1无其他进程、显存／利用率达到空闲门槛后串行运行。实际GPU检查包括共享初始函数一致、可训练参数计数和有限loss／梯度；当前GPU结果未产生，不能报告完成或有效。

任何硬失败写needs_attention并停止。中断病例在原目录标注后从共享起点新建run，禁止拼接训练／预测；已有完成证明需逐文件SHA复验。日志和状态在`summary/onevision_diagnostics_20261016/knowledge_fit/`。

## 解释边界

严格回忆失败可能还有别名／语言／长答案的影响；即使能回忆16事实，原池与目标问题的字面关系／固定tail覆盖有限，不能据此承诺医学大幅提升。只有写入核验成功、且正确相对错配控制在相同提示／预算下有稳定下游优势，才形成该知识机制的正证据。单seed、单一小事实池不足以支持所有领域或所有CL方法的结论。

## Read-only completion audit

`analysis/onevision_knowledge_fit_evidence.py` reads only durable completed cases, verifies source/artifact SHA and frozen IDs, checks all64 raw gradient norms against CPU FP64 and reports actual deltaW displacement. It preserves original scores, reports fixed-prefix sensitivity separately, and compares correct/permuted within each rank only after both complete. No rank or budget selection. CPU synthetic four-case fixture passed72 summary cells and12 paired cells; tampered SHA is rejected and missing completion proof emits no scores. See `knowledge_fit/COLLECTOR_CPU_CHECK.json`; these checks are not GPU knowledge results.
