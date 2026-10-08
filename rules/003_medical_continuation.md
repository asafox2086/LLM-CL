# 医学文本生成续训（2026-10-07）

用户授权：将已完成七任务的模型接着训练两三个医学新任务，观察学习效果与遗忘。本轮选择两个，四方法各自延续原 checkpoint。

## 任务与数据

顺序为原 SuperNI 七任务 → MTS-Dialog 医患对话生成临床记录段落 → IU X-ray Findings 生成 Impression。

每个新增任务使用 train/dev/test = 1000/100/200，英文纯文本。MTS 保留官方 train/validation/TestSet-1 成员关系，按固定 hash 截取训练预算。IU 直接读取 NIH OpenI 原始 XML 报告，去掉缺少 Findings/Impression 的记录，按归一化 Findings 去重后 hash 划分，以报告为单位；未提供患者标识，不能保证患者级独立。同一报告的多张影像不会产生重复训练行。本轮没有使用图像。

原始文件位于 data/medical_raw，冻结后的 JSONL 位于 data/prepared/medical_generation2_after_superni7_v1。rules/003_medical_manifest.json 固定所有划分的 SHA256；audit.json 记录排除样本。跨全部九任务检查 train/dev/test 输入重复。MTS 从官方 GitHub API 下载，并验证 Git blob SHA；IU 原始报告包含其 CC BY-NC-ND 4.0 许可信息。仅作为本地研究使用并保留原始来源。

这是自建生成式 CL 扩展，不是 MedCL-Bench 标签分类协议，也不是声称复现模型合并论文的任务序列。

## 模型起点与训练

SeqLoRA、MIGU-LoRA、O-LoRA、SAPT-LoRA 各自使用 superni_generation7_v2 对应已完成运行的 continual/checkpoints/stage_07/completed.pt。不更改父运行，不重新从基础模型训练。加载相同 google-t5/t5-large 基础权重和全部已训练 adapter；SAPT 还加载 router 和六任务伪记忆。每个新运行保存父 checkpoint SHA、父模型文件 SHA、配置、源代码快照。

第 7 阶段为医学训练前；第 8/9 阶段为两个医学任务训练后。继续沿用 rank=8、alpha=32、dropout=0.1、lr=1e-4、每任务1 epoch、有效 batch=16、microbatch=1、eval batch=8、FP32、prompt 768 / target 256、seed42。每任务63次更新，两个新任务126次。每个新任务重置优化器与调度器，与原训练各任务之间的策略相同，不把第七任务已结束的 optimizer 调度继续套用。

SAPT 原七任务实验最后阶段未执行 finish_task（当时没有下一任务）。续训在第七阶段医学基线评估之后、任务八训练之前补做 SciQ 的 generator + 128 伪样本记忆；这一步只读取原任务七训练数据，不更新原任务 adapter/router，并单独记录。任务八同样在任务九前生成记忆；任务九结束不再生成后续记忆。已有原任务1–6伪记忆正常保留。

长度审计：MTS 训练有13/1000输入超过768 tokens、38/1000目标超过256；dev 为1/100和3/100，test 为1/200和6/200。IU三个划分均无截断。保留原预算保证延续一致；测试评分始终使用未截断参考文本，完整输入文本也保留，tokenization.json 报告截断比例。此处是短预算探索实验。

## 评估与存档

医学前评估两个新任务。原七任务医学前分数与逐题回答从父 stage_07 校验后复制，避免重复耗时。医学每学完一阶段，完整测试九任务：原七任务各500条，新两任务各200条。每方法3×9矩阵，11,700条记录（其中3,500条旧基线复用，8,200条新生成）。

逐题保存原始 prompt、参考答案、生成文字/token、ROUGE-L、exact match、token F1。主要比较旧七任务平均分变化、医学两任务平均分增长、各任务变化，以及第一个医学任务学完第二任务后的变化。ROUGE-L不是临床事实准确率；自由生成任务不以exact match为主要质量指标。

结果目录 exp/result/medical_generation2_after_superni7_v1/<method>_t5large_after7_medical2_seed42_epoch1/<UTC timestamp>/。comparison.md / comparison.csv 随阶段更新；score_matrix.json、metrics.json、stages/、predictions/、resources.json、logs/ 全部保留。每个优化步骤保存模型/优化器/调度器/RNG/cursor；每阶段保存 completed.pt，预测按 batch fsync，可恢复。

启动：bash exp/start_medical_detached.sh METHOD GPU

恢复：bash exp/start_medical_detached.sh METHOD GPU --resume RUN_DIRECTORY

启动使用 nohup + setsid，stdin=/dev/null，日志重定向；父进程与worker忽略SIGHUP。run lock阻止并发恢复，恢复前验证源码、数据、父模型身份。
