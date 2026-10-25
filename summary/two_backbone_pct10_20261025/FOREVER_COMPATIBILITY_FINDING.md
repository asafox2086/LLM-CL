# FOREVER：10%规模兼容问题（源码发现，未尝试模型）

本记录只说明原1%专用适配器不能直接用于当前10%预算，不是质量阴性，也不证明任何新方法。没有运行FOREVER模型、评分或额外QA。当前SeqFT／PASs-MoE等native组合不受以下两处FOREVER专用路径影响。

已冻结10%抽样账本实际给出S1为3461 QA／217次当前任务更新，S2为107 QA／7次更新，总13105 QA／828次更新。来源为本目录SAMPLING_AUDIT.json（SHA `218ce70e3f7e4600e25ae227bbac5118a6794a5a59a41b7f2a267dc87e93b9f4`），不是按1%数量乘十估计。

1. `analysis/forever_multimodal_core_20261019.py` 的 `ForeverStage.__init__` 以 `ceil(len(current_train)/16)` 得到主更新数，随后要求 `1 <= main_updates <= 24`。类注释明确原1%校准覆盖全部主更新，更大预算需要完整动态回放调度，不能静默省略可达分支。10% S1的217已超过24；仅删除断言会改变方法含义，而不只是修复规模检查。该核心还在阶段末统一设置校准／强度，并固定报告 `dynamic_replay_triggers=0`，不能把这套1%简化等同于完整10%实现。
2. 原FOREVER适配器 `preflight_replay()` 固定 `begin_stage(2)`，仅执行一次当前任务更新，随后调用 `after_update(1, 1)`。`after_update` 要求 `core.main_updates == updates`；当前S2主预算为7，因此该预检路径在完整正式S1之前就不满足断言。这是独立于上项的更早触发点；不能声称只修改两个1313训练量guard即可兼容。

核对来源：原 `analysis/forever_multimodal_core_20261019.py` SHA `2d9d5be87d81df80eab6ddf0095ce399c9a73d9c8c29909a5022d2364fdeeed8`，与已运输84文件receipt一致。适配器逻辑来自原 `exp/two_backbone_forever_20261019_v3/forever_adapter.py`；pct10构建仅把两处1313量检查改为manifest实际数量，未改变上述预检、`after_update`或核心调用。此判断基于源码和冻结计数，没有导入模型或执行这些路径。

2026-10-25本地 `LATEST_OLD_OBSERVATION.json` 的unix `1792938470.6747768` 快照记录两FOREVER组合仍为 `waiting`，只有SeqFT与PASs-MoE各两骨干已启动。此为当时快照，不是永久实时状态。资源占用不构成本次兼容判断的依据。

必要处理是先把尚未运行的两FOREVER条目留在明确的调度审阅等待状态，保留其余组合及已启动worker，不删断言强行过关、不声称FOREVER已获10%科学支持。旧1%源、proof和已发表结果保持原样。以后若落实10%完整调度，原FOREVER verifier中的1% profile SHA、历史TRAIN选择数量与零动态触发等断言也必须按实际新实现核验，不能复制旧1%证明。
