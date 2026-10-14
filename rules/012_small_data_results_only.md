> 数据修订核验完成：T5 地理文本 GeoQuestions1089、T6 遥感 GeoChat、T8 金融原图 MME-Finance。当前训练尚未启动，待作者方法适配／GPU 预检。详见 data/DATASET_VERIFICATION_20261014.md。

# 小模型、小数据、轻量化：本轮固定协议

2026-10-14 用户明确要求：研究背景为“小模型、小数据、轻量化”；每任务训练 **100 条**；不要 checkpoint，要完整阶段结果。本文覆盖 008–011 的样本数、权重保存和旧结果比较设置。

| 项目 | 本轮设置 |
|---|---|
| 基座 | Qwen2-VL-2B-Instruct，同一固定版本 |
| 任务顺序 | 自然文本→自然图像→医学文本→医学图像→地理文本→遥感图像→金融文本→金融图像→艺术文化文本→艺术文化图像 |
| 每任务训练／验证／测试 | **100／0／50**，所有方法等量；一轮；seed 42 |
| 主比较（8 种） | SMoLoRA、PASs-MoE、Hyper-LLaVA、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER |
| 额外对照 | 初始未训练模型仅用于初始性能与 FWT；不运行 SeqLoRA |
| 历史 QA 回放 | 不使用；DGG 原作者流程含回放，本轮排除，论文二维表保留 |
| checkpoint | **不保存权重、优化器、last 或恢复快照**；方法必需的历史参考在运行内存中保留 |

数据来源保持原协议：SQuAD；VQAv2/GQA/TextVQA；MedQuAD；VQA-RAD/PathVQA/SLAKE；GeoQuestions1089 计数／人口文本；GeoChat 遥感开放 QA；FinQA；MME-Finance 原图表 QA；AQUA 知识文本及视觉问答。科学论文域没有重新加入。原问题、答案、图片和标注不改。七个既有任务保留原清单；新三任务按核验报告的论文问题类型／作者类别及研究 holdout 划分，各 100／50。不复制样本补齐。

当前清单为 `data/TenTaskCL/equal_budget_100_0_50_geo_fin_seed42_20261014/manifest.json`；执行策略为 `exp/ten_task_cl_20261012/execution_policy.json`。旧 2000/100/200 和 500/50/50 训练结果不与本轮混算。

训练前和每阶段完成后，评测 **全部十个任务，各 50 条**，共 11×10 矩阵、5500 条测试预测／方法；取消验证集，不做验证推理或模型选择。每条保留 ID、所属数据集、预测、原参考答案、EM、Token F1、ROUGE-L、生成 token 数和耗时；每个任务和混合任务内每个源数据集分别统计分数、正确数、样本数。

阶段完成只写 `stages/stage_XX.json`、结果表及矩阵，不写模型快照。训练日志保留逐 update 的 loss、样本数、时间、输入/输出 token 及峰值显存。内存中的模型顺序继续训练下一个任务；无 checkpoint 模式下，进程中断无法恢复训练参数，已有结果仍保留并可重新分析。

准确率是统一归一化答案匹配 EM，开放问答同时报告 Token F1 和 ROUGE-L。不能将这些分数冒称 VQAv2 官方软准确率、FinQA 程序执行准确率或 CUAD 官方 AUPR。AP、ForgetRate、FWT、BWT 分别可由三种分数矩阵计算；主分数沿用 ROUGE-L。只在完整 11×10 矩阵齐全时发布最终指标，不补值、不混旧预算、无多 seed 均值或标准差。

相同未训练基座的初始预测可复用：必须验证原始输入子集、答案引用、固定基座 asset manifest、缓存 SHA，以及当前 LoRA B 参数全部为零。仅复用 stage 00；任何训练后的阶段均重新评测。不是复用旧训练成绩。

完成后自动绘制 EM／F1／ROUGE-L 的三张阶段×任务热图及平均性能曲线，输出 PNG、PDF 和绘图数据 JSON。绘图数据来自逐条保存的真实结果，不平滑、不补不存在的阶段。绘图失败不丢失评测结果。

结果路径：`exp/ten_task_cl_20261012/runs/<method>/seed_42/small_100_0_50_<time>/`。汇总、逐任务与逐数据集 CSV、各分数的持续学习指标及图在 `summary/ten_task_cl_20261012/`。本轮八方法原始数据和基座共用一次；不保存 checkpoint 后，整轮日志、预测和图片先预留 **2 GiB**，不再使用旧 400–700 GiB 权重预算。

按实测旧运行参考：SMoLoRA 约 30–40 小时，主要耗时为十轮全任务生成评测。只是单卡估算，训练后回答长度及图像输入会改变耗时，排队等待另计。其他方法在实测首阶段后更新估算。

四卡队列按空闲状态分配不同方法；其他三卡在占用期间不抢停。新任务尚未开始训练；SMoLoRA、PASs-MoE 的新任务 GPU 预检待做，其余六种适配未完成。SeqLoRA 已移除，不能将准备状态计作已运行。当前状态以 comparison_status.json 为准。
