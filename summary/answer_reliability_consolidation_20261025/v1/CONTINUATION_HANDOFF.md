# 固定r8三臂：一次模型前资源拒绝后的限定续接

研究命题／损失／数据／随机策略／参数／评价未变，尚无完整质量结果。原README仍为读者总览，不写本运行日志。

- 固定RUN `/tmp/llmcl-answer-readout-cl-1084-ft3em1z0`；READY `e654eb6f0c0599ef76022711866ff5e5b7061d129c1cf7f57565eb2ff36ff2fe`，8科学源原字节不变。
- 原unit `llmcl-answer-readout-cl-20261025.service` 保留failed/exit1，不拿它判当前新监督是否存活。
- 当前unit `llmcl-answer-readout-cl-prestart-continuation-20261025.service`。unix1792926762.8731668实际active/running；original worker4001502/start762499983/UID1084/PGID同PID/GPU2身份匹配，11梯度见证。这只是快照，今后重新实查。
- gate原完整19阶段／87更新／1408.252424500秒已保留；原worker3992422退出，不重训，不续它的CP训练。此前original尝试3999463在STARTED和Torch之前被组合资源谓词拒绝，0模型／QA；不能指认哪个条件或称NVML残留已证实。
- 原log／original FAILURE／original EXIT1／CONTROLLER_FAILURE四件已按SHA和原路径映射归档 `preSTARTED_guard_attempt_20261025/`。两子agent接受仅此纯资源准入调度补正；不是恢复已开始失败模型。
- 外部stdlib监督只执行从未STARTED的original/shuffle训练与三个eval；每阶段先等原同卡ownerless、memory<500、util<10连续两次间隔2s，不放宽条件，不向他人进程发信号。
- 所有等待／故障／维修仍在旧ACTIVATION+18300秒内：绝对截止 **2026-10-25T15:41:42.561657+00:00**，不能重置。各尚未开始phase的5400/600预算从其首次STARTED计；全部模型phase和仍≤18000秒。
- 后续真实模型故障按原点关闭；不读取partial生成／F1或挑配置。不要直接重启原controller，也不要运行旧ARC collector。

## 完整结束后

先用 `.deployment/arc_gpu_lifecycle_20261025.py observe` 核新unit/真实身份。监督在六phase全部成功后封CONTINUATION_MODEL_COMPLETE，再调用原verify一次300分，并封CONTINUATION_VERIFIED把原失败archive、原gate、新监督、ALL_EXIT、原VF连起来。

只有完整退出／原身份已无／新unit inactive0／两VERIFIED均存在后，使用 **`.deployment/arc_continuation_collect_20261025.py`**，这是只取回已完成proof的消费者，0新评分。**不要用 `.deployment/arc_gpu_collect_20261025.py`**：它假定旧unit成功且会再次调用verify，已不适合此有公开偏离的执行。

随后 `.deployment/arc_gpu_archive_20261025.py` 唯一私有归档整个RUN并核新CP与旧故障archive，已补CONTINUATION_VERIFIED／permit／supervisor校验。汇总必须读取原VF和续接receipt一起解释；不得呈现“从未失败”。gate/original/shuffle和naive各域／逐题全报告，接受数/Hamming/退化门、训练与推理成本分开，正负均保留。单seed已探索DEV不称正式胜最强；若选方法进入正式确认须另论证。

Qwen最强资源参照已完整成功：VF `dd97300219ecbfaeb5052c571c68a4fd61fc3dd8455ab24eff5aa8627ca05e7c`，原图五窗4312–4955MiB，LLaVA朴素r8逐窗低55.15–58.80%。主表Qwen4954.604†/PAS2959.415†及完整限定已push `2b5e84b0cd5a746a02b2d559d9fe18ca3b51d629`。原18行其它列、8组存储、FOREVER不变，不把FIT模型峰填formal Seq或Ours。新方法质量仍未证明。

NEW最新基线独立快照见BASELINES_DURING_CONTINUATION：原CL-MoE/SMoLoRA仍在GPU2/3，主1221743 active，GPU1他人占用，5项待原reload/空卡，GPU0不由我方分配；没有新增sealed结果，正式仍8/16。只有新完整proof才增正式结果及对应最终参数统计，勿重跑已封存组合。
