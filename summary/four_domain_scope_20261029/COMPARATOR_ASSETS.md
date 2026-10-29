# 四域S16对照：既有模型资产范围

本次仅核NEW已有proof、来源路径、小JSON和checkpoint文件stat；没有加载/重哈希大CP、读取QA、预测或分数，也没有使用OLD relay。OLD定点存在性事实由root另一次只读stat提供，下表明确区分。S19的T01–T16切片可作已跑结果展示，不能替代S16四域终点。

| 1% LLaVA方法 | 权威来源及当前资产事实 | 可用于S16终评？ |
| --- | --- | --- |
| q/v-r8 SeqFT | 原proof `completion_observer_seven_20261018/completed/b463d0e9fdfecc9eb456.json`，candidate.host=old2080，RUN `exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01/SeqFT_LoRA/training_20261017_161154`。root在OLD确认 `checkpoints/stage_16.pt` 存在，1926007972B；stage16元数据记录的**CP SHA**为 `773b304e66c194902c3d76e3d5923d1a58aa274bc744cc30c6a08a67487a4160`，latest仍S19。本轮未全SHA，未确认NEW有该S16耐久副本。NEW同相对原RUN不存在，不能据此误称OLD缺失。 | **唯一已确认实际S16载荷存在的本清单对象**；加载前仍须核完整SHA/元数据/来源及前缀边界。 |
| PASs-MoE | 原proof `completion_observer_seven_20261018/completed/88902345d5cf6fe545e3.json`，candidate.host=old2080，RUN末段 `PASs_MoE/training_20261017_161725`。NEW无同相对原RUN；root在OLD确认stage16.pt和checkpoint stage16.json均不存在、latest为S19。原proof绑定的 `stages/stage_16.json` / `validity/stage_16.json` 是执行/有效性证明，**不是完整模型CP**。 | 无已确认可重载S16资产；S19不能倒推出S16。 |
| OSFT | 原proof `completion_observer_seven_20261018/completed/5193ee4c6861655d85e8.json`，candidate.host=old2080，RUN末段 `OSFT/training_20261018_044103`。NEW无同相对原RUN；root在OLD确认stage16.pt和checkpoint stage16.json均不存在、latest为S19。原proof内stage16执行/validity小JSON不等于CP。 | 无已确认可重载S16资产。 |
| all-r8 / LoRA_all | NEW原目录 `exp/osft_increment_confirmation_20261022/v1/private/checkpoints` **有** `stage_16.json`（文件SHA `ff980e8ceb48ee3500cd4eb4f16258c3baed4535ea899b321512157dbe9747e0`），声明S16完整架构载荷1946027547B、CP SHA `e76b07765e844f4345f0f0a1f48da7ce288f67968f9e5bf7a4aa2766f76f513a`；但当前该目录只剩S19.pt，S16.pt不存在。不能写成从未保存S16，也不能把元数据当仍有模型。独立 `.deployment/all_r8_resource_snapshot_20261028/SNAPSHOT.json` 只绑定/硬链接S19，1941599549B / SHA `416ed177803b9bf986dacd73a6d1bf18c6c0c59a1eb65f42785d81a628d31519`，未提供S16副本。 | **仅元数据，现缺载荷**；本次所查原路径和已封独立快照没有可重载S16证明，不声称已穷尽其他未列归档。 |

前三条proof均位于 `summary/two_backbone_pct01_20261017/` 下。all-r8原COMPLETE SHA `ea9bd2283e2cc7d8948267289ff2e4cc7a20685cdc7b511fc176b628870d6a5c` 与VERIFIED SHA `abbd2da43753c6bbeb137d7bef535983efbcafdaf08686f61550886b2f18598d` 均只认证其最终S19，不能转授S16质量。完整S19正式结果与所有历史记录继续保留。

## 最小下一实验次序

1. 优先使用确有载荷的Seq S16，严格核来源/完整CP SHA后，独立重载原完整图，对同一16任务的19331 TEST做一次完整终评，保留新输出并明确是既有16阶段训练前缀，不重训或覆盖旧S19。若另折叠部署，必须评价实际部署图，不能自动转用未折叠质量。
2. PAS与OSFT若要成为匹配四域S16强对照，需要独立四域训练后完整终评；现无可用S16资产，不能以S19质量与Ours S16作匹配优劣结论。上述是必要实验缺口，不是本文件授予启动。
3. all-r8可继续保留S19四域展示和简单强参照身份；现有S16小JSON不足以免训练复用。只有另有可证实的真实S16载荷，或独立四域训练，才补S16行；不为此从S19重建/猜测历史模型。

## 初始化/前缀限制

沿用 `INITIALIZATION_SOURCE_REVIEW.md`（SHA `4f3a3665a708bc135c6d85aa1c041423e0968c4a035fddc8f0d0466e4f947c12`）：Seq忽略未来instructions、固定q/v-r8；PAS构造忽略instructions、OSFT按矩阵形状SVD，未发现未来任务总数初始化依赖，因此可讨论其模型前缀。但这不等于完整method/context/RNG/部署图已普遍认证；缺S16载荷尤其不能用构造器独立性补齐。all-r8并未被该文档单独认证，不把Seq q/v的源审自动推广到其动态目标注册实现。旧S16只评T16/T17，未已有完整R[16,1..16]；新四域最终质量仍需实际全量终评。
