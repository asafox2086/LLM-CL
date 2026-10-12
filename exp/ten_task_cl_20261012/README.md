# 本轮五领域十任务实验

配置与方法：[009 协议](../../rules/009_small_budget_no_replay_baselines.md)。目录已建立；没有开始正式训练。

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

`stage_checkpoint.py` 提供保存接口，**尚未接入五个作者方法的训练器**。正式运行前必须验证保存、加载和继续一步训练一致性。仅建立目录或保存接口不代表已实现断点续训。

日志与大检查点留服务器，不推送 GitHub。汇总、配置、样本 ID manifest、版本和报告可推送；预测若包含受限数据，遵循来源许可。本轮报告目录：`summary/ten_task_cl_20261012/`。
