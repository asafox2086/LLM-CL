# 阶段保存与研究优先

人类要求研究需要GPU时暂停一个基线，研究进入分析后恢复；所有新启动基线每阶段保存完整checkpoint，磁盘不足可人工清理旧中间权重，最终权重必须保留。

已部署独立版本，不修改已有活进程使用的旧源码。模型、数据、单轮更新预算、学习率、生成和评分规则保持；新增的是保存、暂停、恢复与归档。旧版没有保存的中间参数无法补齐。

| 范围 | 新运行目录 | 实际调度入口 |
| --- | --- | --- |
| 1% native未来任务 | `exp/two_backbone_native_checkpoint_20261027` | `analysis/baseline_pct01_pct10_checkpoint_capacity_20261027.py` |
| 10% native未来任务 | `exp/two_backbone_pct10_native_checkpoint_20261027` | OLD新版entry；NEW同上 |
| 10% FOREVER未来任务 | `exp/two_backbone_pct10_forever_checkpoint_20261027` | `exp/stage_checkpoint_dispatch_20261027/entry.py` |
| 1% Qwen FOREVER未来任务 | `exp/two_backbone_pct01_forever_checkpoint_20261027` | `analysis/forever_checkpoint_queue_20261027.py` |

每个完整阶段保存完整method/model、processor、RNG、配置与阶段进度。阶段之间原本重建optimizer/scheduler/scaler，因此恢复从下一阶段开始，不是任意batch续训。S0仅在收到暂停请求时另存完整初态；S19正常完成并保最终。FOREVER保持原子commit，取消自动删除历史权重。

在RUN下写入唯一`PAUSE_REQUEST.json`，包含`request_id`和`reason`，worker会在完整阶段评价及checkpoint提交之后写`paused_checkpoint`并正常退出。确认真实worker退出、checkpoint完整后才能让卡。恢复请求`RESUME_REQUEST.json`须包含当前`run_status.json`中的`pause_receipt_sha256`；原调度器同RUN、同源版本使用`--resume`。不把失败自动转为恢复，不复用旧版本checkpoint到新代码。

旧版本已通过但尚未训练的预检记录全部保留；在新保存版本首次训练前执行对应新版本的原预检及重载，不能冒称旧source proof覆盖新接口。已运行的旧baseline继续使用旧版本，其停止限制不因此消失。

OLD的新阶段权重由`analysis/stage_checkpoint_archive_20261027.py`逐stage复制到NEW私有归档，依据producer原SHA核验；不是仅复制latest，也不删除源或旧归档。NEW本地新运行直接保存在耐久磁盘。归档服务保20GiB传输前余量；无空间时明确记待归档，不把尚未复制称为已存。人工清理只考虑旧中间`.pt`，保留最终、latest/committed latest指向的可恢复权重，以及全部答案、分数、日志、配置、阶段元数据和核验记录。禁止批量删除RUN。

实际交接证据：`ACTIVATION_OLD.json`、`ACTIVATION_NEW.json`、`pct01_forever/ACTIVATION.json`、`ARCHIVE_ACTIVATION_NEW.json`。元数据、源码结构及Linux调度构造检查通过；尚无新版本真实GPU暂停—恢复结果，后续必须按实际记录报告。

本次已停止运行时间最短的LLaVA10% OSFT，为既定研究三臂让出OLD GPU1。其最终评价未完，完整285文件保留并在两机私有归档；没有中间checkpoint可续训。研究进入分析后，应保留旧RUN、以新版本fresh恢复这组基线的工作，不能冒称续接S19。主表不填该未完成结果。
