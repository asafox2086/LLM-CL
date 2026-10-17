> 当前48内部先完成两锚的8条SeqFT控制，再补其余机制；全部病例和设置不变。新CPU入口及进程保全见[内部顺序修订](../summary/onevision_diagnostics_20261016/causal_anchor_first_queue/SCHEDULING.md)。实时队列改为`causal_anchor_first_queue/queue_state.json`，旧路径为重定向，不恢复旧所有者。

# 同域／跨域与小任务预算：确定性机制复核

目标对应主报告第7项的明确缺口：原默认后端相同首阶段已有参数差异，23题单锚不能支持严格归因。本研究复用原六控制的原始训练函数，不新增CL方法或路由，不修改运行中的基线／全量确认。

## 固定设计

- 两个原生医学锚：T4 Organ、T2 Modality。T4沿用原研究，T2补充不同问答类型；是观察旧结果后的追踪研究，不称未见数据确认。
- 每任务仍用原4×6研究按TRAIN ID哈希固定的8题；代码逐ID断言相同，全部来自原冻结1%池1313题。每阶段4次正LR更新、微批2，5阶段共40次新任务QA曝光，seed42。
- 四序列：重复同锚5次；同医学不同任务；跨4域短答案序列；跨5域序列。T4顺序与原研究完全相同，T2将其作为锚并在同域序列中换入T4；研究自定顺序，不称作者官方课程。
- 六控制：SeqFT-LoRA、ER-LoRA、EWC λ1、EWC λ100、anchor-gradient-noop、anchor-gradient-projection。后两者是原机制干预，不是已发表新算法。共2×4×6=48条固定轨迹，全部报告。
- 直接调用原`Mechanism.step_task`与`fisher`；原训练器文件保持SHA不变。fresh AdamW、lr1e-4、weight_decay0、clip1，每阶段重置优化器／seed；不按测试改超参数。
- 确定性默认后端，CUBLAS配置在导入Torch前设置；不把数学后端与默认后端结果拼接。

## 核心准入门槛

每个锚以首条repeat+SeqFT建立规范stage1参数。**每一条后续轨迹都真实训练stage1，并要求A/B逐元素完全一致，不设误差容忍。** 未通过就保存错误并停止，不放宽门槛、不纳入方法比较。CPU小模型仅证明原六训练器在此接口下共同首阶段相同，不能代替OneVision实际GPU检查。

该门槛消除“同一首阶段数值不相同”的旧混杂；它不消除任务内容、目标长度、额外历史计算或单seed限制。

输入结构CPU预检已覆盖本设计879不同QA（80训练、799测试）及全部提示占位符，固定队列不匹配0；队列外已知失败FinMME:5082成功复现2052/2106。使用真实return_tensors=pt processor与官方dummy特征打包；不是视觉编码器/GPU前向检查。[详细证据](../summary/onevision_diagnostics_20261016/matched_task_domain/INPUT_GEOMETRY_AUDIT.md)。不因此放松任何实际GPU门槛。

## 评测与信息／计算预算

1. 每锚从原生测试按固定ID哈希取32题，在stage1／3／5评原生成，以Answer F1为唯一性能指标、NLL为机制辅助，所有条件同ID。原文件附带EM/ROUGE仅保留追溯，不作为并列主成绩。
2. 每任务固定8条held-out测试题，在stage2／3／4／5评价当前任务学习，避免只保护旧任务却不学新任务。原问题／图片／答案不改。
3. SeqFT四种序列最终另外评完整原生锚：Organ507题、Modality228题。用于检查32题结论是否延续到同锚全测试；其他控制不能冒称已做完整735题，更不能当21454题主结果。
4. 不复用旧默认后端预测。所有方法保存stage1–5适配参数／Fisher及逐阶段训练loss、实际LR、原始首批梯度和投影参考。原训练器只保存**每阶段首批**完整梯度与所有更新范数，不声称保存全部更新的原始梯度。
5. 每阶段分别记录训练前后CPU/GPU字节，CUDA训练峰值在训练前重置；Fisher单列窗口、整条轨迹含推理的内存另记。采样CPU／NVML与CUDA分配峰值分开解释。

补充TRAIN覆盖核验：[固定8条缓冲区实际回放了什么](../summary/onevision_diagnostics_20261016/matched_task_domain/replay_exposure/VERIFIED_RESULTS.md)。统一源QA/采样规则不等于每个旧锚被回放相同次数；所有分析列实际覆盖，未完成组只列规则确定的计划。不改ER算法，也不将样本覆盖直接当F1原因。

名义新QA曝光总计1920；ER额外历史回放256，EWC Fisher额外320，noop／projection额外参考TRAIN梯度QA 2048。额外开销不能称与SeqFT完全等计算。正式1%一轮CL和这里8题／4步机制干预分别报告。

## 如何判断与停止

- 先比较每个锚共同stage1→stage3／5的保留、丢失、新增，与重复同任务／同域控制对照，不能只看最终总分。
- 当前任务8题的学习与旧锚保持并列，防止将“什么都不学”包装成抗遗忘。
- 六机制均使用同一锚、ID和顺序；ER／Fisher／projection额外信息单列。不要从小分差直接推出方法优劣。
- T4与T2、cross-short与cross-five分别报告，不选最好锚或序列。两锚均医学、单seed／顺序，领域与内容不能完全解耦；只主张该固定序列下的差异，不宣称已纯化任意领域数量的抽象因果。
- 任何首阶段严格门槛／源SHA／原始ID错误，保存证据并停止；中断轨迹从基座新目录重跑，不拼接旧训练或预测。

## 调度（2026-10-17用户授权后的优先级修订）

用户明确委托“这个gpu1的方案你自己定，我只要结论”。当前由nohup CPU服务`llmcl-onevision-causal-priority.service`统一管理：**原cap1自然完成 → 12组两模块准入原型 → 本协议48组原控制 → 四组完整方法训练/最终确认 → 原cap3 → 完整转移评测**。原因归属优先于昂贵方法主表，全部原控制和方法保留。旧CPU协调器已计划停用，不能重启，否则形成重复所有者。

切换前已先设置并核验旧服务KillMode=process，再仅终止CPU父进程；当前GPU1原PID/startticks及基线均保留，不中断推理、不抢其他卡。新CPU服务发现任意GPU1进程/未空闲就等待；每个原案例用独立GPU进程调用同一`--case`入口，训练器源码和全部门槛不变。失败保留并停止，不盲重试。新服务已enabled支持重启接续，但中断训练仍须原规定从基座重跑，不能拼接。

[委托下的优先级修订](../summary/onevision_diagnostics_20261016/causal_before_full_method_queue/AUTHORIZATION.json)、[实际切换证明](../summary/onevision_diagnostics_20261016/causal_before_full_method_queue/ACTIVATION.json)、[实时统一队列](../summary/onevision_diagnostics_20261016/causal_before_full_method_queue/queue_state.json)。只新增CPU入口，未改本GPU源码/训练器；调度变更不是实验结果，48组实际GPU门槛仍待执行。

报告在`summary/onevision_diagnostics_20261016/matched_task_domain/`，大权重在服务器`exp/onevision_matched_task_domain_20261017/`；模型／数据／checkpoint不入Git。初始CPU准备记录保留；实际GPU门槛与已完成轨迹由[只读完成证明观察器](../summary/onevision_diagnostics_20261016/matched_task_domain/VERIFIED_RESULTS.md)核验，当前尚未完成全部48条，不将初始等待描述当实时状态。
