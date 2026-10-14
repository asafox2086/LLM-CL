# 新服务器实验准备状态

资源已迁移并完成内容校验；当前没有实验训练进程。

T5/T6 已核验并换为 GeoQuestions1089 地理文本／GeoChat 遥感图像，T8 为 MME-Finance 原始图表问答，艺术保留。预算 100 条训练／50 条测试，seed 42，无验证集，不保存 checkpoint。

SMoLoRA、PASs-MoE 作者适配已准备，仍需在当前已核验新任务上进行 GPU 预检；其余六个方法仍待适配。

迁移前报告保存在 summary/ago/old_server_before_migration_20261014，原逐题输出与日志仍在 exp/ten_task_cl_20261012/runs 对应旧运行目录。不得把这些旧任务结果标为修订后的新实验结果。
