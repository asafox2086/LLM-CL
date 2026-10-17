# 双骨干公平持续学习：V2

当前正式框架 `two_backbone_native_cl_20261017_v2`：旧机运行OneVision 0.5B，新机运行Qwen2-VL 2B。新GPU0保留，GPU1研究，GPU2/3优先Qwen 1%。19个原生任务，seed42，一轮，无验证集。训练/测试ID、顺序、方法实现、优化预算、共同图像输入、评分一致；原生模型/tokenizer/processor因骨干而异，耗时与显存按硬件分开报告。

## 数据

1%训练1313条，完整测试21454条，冻结清单不变。原全量131019条中只有SLAKE:1622没有答案；用户于2026-10-17授权排除此条，有效全量131018条。原始文件不改；派生版本只生成删除该行的T04训练文件，其余37份QA文件复用原文件，所有剩余QA行字节不变。两机核验相同清单SHA，1%仍为全量子集。见[FULL_MISSING_LABEL_EXCLUSION.json](FULL_MISSING_LABEL_EXCLUSION.json)及[FULL_BOTH_HOSTS_VERIFICATION.json](FULL_BOTH_HOSTS_VERIFICATION.json)。未来全量所有方法都使用此版本，从新框架基座重新开始；旧stage4失败轨迹不拼接。

## 公平性与准入

两骨干都使用同一个RGB/BICUBIC等比例最大401408像素输入步骤，不裁剪，不覆盖原图，之后进入各自原生processor。同一128MiB CPU LRU视觉缓存、按原生FP16/BF16保留精度，禁止Qwen独占跨作业磁盘缓存。全部7033张使用图像文件SHA在两机一致；Pillow12.3.0一致，19个首样本共同渲染图像RGB和语义输入在两机精确相同。证据见[both_hosts_input_equivalence.json](both_hosts_input_equivalence.json)、[cache_CPU_validation.json](cache_CPU_validation.json)。CPU门禁不能冒充GPU训练或准确率结果。

同方法两骨干共享有效batch16、microbatch1、seed42、一轮、初始化/作者超参、梯度缩放初值1。修复公共ceil warmup问题，单更新阶段有正学习率；每阶段记录实际optimizer LR、正更新数、训练参数SHA变化。保留作者各方法差异，不把所有方法改成同一算法。OneVision原生官方前向NLL/4token生成、19任务有限反向及独立进程checkpoint重载必须通过。SMoLoRA需原生BF16，旧2080Ti不强制FP16；等待新GPU3专用lane。

## 版本与资源边界

V1在正式训练阶段0被取代，原因是两骨干缓存/精度/图像预算不一致及Qwen大图OOM；保留失败证据，结果不得进入主表。旧Qwen 1%公共warmup导致零LR轨迹同样隔离重跑。旧全量SeqFT于stage4因空答案反复失败，legacy队列已停用，不能称全量完成。旧全量自动恢复与V2 ER预检曾短暂共享GPU3；该预检窗口不用于内存/耗时结论，正式ER从基座单独开始。当前GPU进程源文件不热改。

共享源码逐文件SHA必须一致；本次只改派生数据与未来full策略，并仅重启CPU协调器以读策略，不中断1%GPU worker。排队顺序为Qwen1%优先，随后OneVision SMoLoRA，再由硬件协调器开放Qwen全量；失败需审计，不能反复盲重试。

## 结果

[主表](../two_backbone_pct01_20261017/MAIN_TABLE.md)按骨干分块，列为五域最终成绩与AP/FWT/BWT。初始全19、中间当前+下一任务、最终全19；未测格为空，不伪造完整矩阵/历史峰值ForgetRate。SeqFT保存所有阶段，其他方法最终checkpoint。逐题结果、loss、真实LR/梯度及CPU/GPU训练前后/峰值在runs中保存；数据/模型/权重/私钥不入Git。GPU1原研究保留独立输入预算与诊断协议，不能和本正式表混为严格同设置对照。
