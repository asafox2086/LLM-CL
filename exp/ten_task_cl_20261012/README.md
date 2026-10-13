# 本轮五领域十任务实验

配置与方法：[010 执行协议](../../rules/010_equal_budget_execution.md)。固定十任务 2000/100/200、seed 42。后台启动器已启动准备流程；是否进入模型训练以 run_status.json 为准。

```text
exp/ten_task_cl_20261012/
  README.md
  artifact_policy.json
  stage_checkpoint.py
  runs/<method>/seed_<seed>/<run_id>/
    config.json
    data_manifest.json
    environment.json
    logs/train.log
    logs/events.jsonl
    checkpoints/stage_01/ ... stage_10/
      model/                 # 所有本轮改变的参数、配置、专家与路由
      training_state.pt      # 优化器、调度器、混合精度、方法状态、随机状态
      metadata.json          # 阶段/步数/版本/数据哈希/基座位置
      evaluation.json        # 该阶段全部十任务的结果
      COMPLETE.json          # 原子保存完成标记
    evaluations/stage_00/ ... stage_10/
      scores.json
      predictions.jsonl
```

每阶段训练和评测后保存独立快照，不覆盖前阶段。stage_00 是初始模型的评测；基座固定且有版本/哈希时无需复制其完整权重十次。快照必须包含所有已改变的参数，不能只保存一个 LoRA 而遗漏专家、路由或其他可训练模块。方法历史样本索引、梯度统计、重要性、专家状态等一并保存；可重建的缓存须记录重建依据。

保存采用同文件系统临时目录写入后原子重命名，只有 COMPLETE 标记齐全的阶段可以用于恢复。训练器还须在每阶段内周期性保存独立的 last 检查点，避免中断后重跑整个阶段。中途恢复保留优化器、调度器、步数与随机状态；阶段边界是否重置优化器遵循固定协议，不能混淆两种恢复方式。

`stage_checkpoint.py` 已接入 SeqLoRA 与 SMoLoRA；真实小型 Qwen2-VL 文本/图像更新和下一步恢复一致性已通过，完整 2B GPU 预检在调度后进行。其余四个作者方法仍待适配，不能标作可运行。

日志与大检查点留服务器，不推送 GitHub。汇总、配置、样本 ID manifest、版本和报告可推送；预测若包含受限数据，遵循来源许可。本轮报告目录：`summary/ten_task_cl_20261012/`。

启动：`/tmp/llmcl-assets-env/bin/python exp/ten_task_cl_20261012/launch.py`。准备日志在 `pipeline_logs/<run_id>/`，每阶段内每 25 个 update 保存 last，全部 T1–T10 完整阶段保留。

2026-10-13 实际运行：SeqLoRA 已完成 seed 42 的全部十阶段，保存 stage_01–stage_10 和 11×10 评测矩阵。SMoLoRA 的完整 2B 模型预检发生 CUDA OOM，尚未正式训练；加入投影内部激活重计算后，CPU 输出和梯度与原作者 forward 完全一致，但 11 GiB GPU 重试仍 OOM。该优化保持八个专家、路由和 dropout RNG，不改变任务、采样或答案。`retry_smolora.py` 仅重试 SMoLoRA 并保留 SeqLoRA 已完成结果；需要先解决显存问题再重试。真实状态与指标见 `summary/ten_task_cl_20261012/`，不能将 CPU 验证等同于完整模型预检通过。
