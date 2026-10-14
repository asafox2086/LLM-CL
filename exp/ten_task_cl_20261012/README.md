# 小模型、小数据、轻量化持续学习实验

当前协议：[012](../../rules/012_small_data_results_only.md)；基座 Qwen2-VL-2B-Instruct；五领域文本/图像十任务；每任务 **100/50/50**，一轮，seed 42。

**不保存 checkpoint**。权重、优化器、方法参考在运行内存中按作者机制更新，阶段之间顺序延续。进程中断不支持恢复训练；已完成的结果仍保留。

运行入口：`.vision-env/bin/python exp/ten_task_cl_20261012/comparison_queue.py`。已有队列不重复启动；独立方法等待空闲 GPU。当前可运行 SMoLoRA、SeqLoRA、PASs-MoE，其余六种适配状态如实记录。

每阶段结束评测全部十任务，每个 50 条；另测当前任务 dev 50 条。每条预测保留原答案引用、数据集、EM/F1/ROUGE-L、生成长度和耗时。各混合任务的每个源数据集单独统计。

`runs/<method>/seed_42/small_100_50_50_<time>/` 保存 config/environment、logs、evaluations、stages/stage_XX.json、task_metrics.csv、dataset_metrics.csv、score_matrices.json、metrics_by_score.json；没有 model/optimizer checkpoint。

汇总位于 `summary/ten_task_cl_20261012/`：comparison_report.md、task_metrics.csv、dataset_metrics.csv、metrics_by_score.csv、stage_scores.csv、compute.csv。仅在完整 11×10 矩阵齐全后发布最终 AP/ForgetRate/FWT/BWT，分别按三种分数计算。

完成后自动输出三张热图和平均性能曲线（PNG/PDF/绘图数据），图在 summary/ten_task_cl_20261012/figures/。绘图复用项目 .plot-env，不需要新环境或代理下载。

准备：`prepare_small.py` 从已验证源清单生成同一等量子集；只复用相同未训练基座的 stage 00 预测，绝不复用旧预算训练后成绩。旧 2000/100/200 和 500/50/50 运行只留历史结果，不参加本次统计。
