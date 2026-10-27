# LLMCL：1%与10%数据下的多领域多模态持续学习

**研究目标：完整模型权重与 GPU 内存显著更小，同时保持有竞争力的总体 Answer F1。** 人类最新允许比最高分略低，按质量—内存取舍判断；允许域间取舍，五域分别报告，重点是论文方法与实测资源优势，不做端侧设备适配。

**方法方向：抽象强方法的有效功能，用简洁机制实现，或从已验证的共同问题设计新方法。** 不局限于改造某个现有算法，不把折叠、普通扩容或换基线本身当论文贡献。新假说必须给出功能理由及实验对照，最终按原F1和完整资源实测判断。

**研究重点（人类最新指示）：只考虑新习得，不研究遗忘。** 后续问题是现有方法为什么没有学好新任务、如何使1%与10%数据下的学习更有效；内存／参数量作为最终评估，不替代学习原因证据。

**设置确定后固定。** 沿用既定数据、训练、生成与评分条件，研究方法在这些条件下的新习得能力、失败和适用边界，不为改善结果反复调整设置。

**当前结论：尚未得到同时满足质量与资源目标的自研方法，Ours 暂空。** 已有1%完整正式结果8组；10%已有双骨干SeqFT-LoRA共2组完整核验，LLaVA／Qwen五域宏分别47.462／52.854，两组最终完整模型已耐久保存；其余基线继续运行／排队。单图LoRA相对PAS与Qwen的固定窗口显存差异已实测，新方法质量仍待验证。

本文只保留主表、探索命题和结论。[补充说明](SUPPLEMENTARY_NOTES.md#探索过程与证据索引)保存完整探索过程、正负证据、故障、成本及运行交接。

<!-- FORMAL_MAIN_TABLE_BEGIN -->
## 主实验：双骨干1%多领域持续学习

Acc采用唯一的**Answer F1（0–100）**。训练集为冻结的1%（1313条），测试集为完整21454条；五个数据集按19个原生任务顺序训练，seed=42。空白表示尚无完整核验结果，不能视作0分。

| 方法／Backbone | 类型 | 发表会议／年份 | TextVQA<br>自然／通用 | SLAKE<br>医学 | RSVQA-LR<br>遥感 | FinMME<br>金融 | CODA-LM<br>交通 | 五域宏 F1 ↑ | AP19 ↑ | FWT ↑ | BWT ↑ | 最终权重存储<br>保存态 MiB ↓ | 相对原基座增加<br>MiB ↓ | 固定五窗GPU峰值†<br>MiB ↓ | 首词延迟<br>s ↓ | 生成速度<br>tok/s ↑ | 常驻适配器<br>数量 ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen2-VL-2B-Instruct** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 朴素顺序微调 | 标准对照（无独立发表年份） | 86.312 | 47.654 | 51.110 | 26.576 | 33.210 | 48.972 | 47.812 | 13.258 | 4.415 | 4217.462 | 4.156 | 4954.604† | 未测 | 未测 | 未测 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| **LLaVA-OneVision-Qwen2-0.5B-OV** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 朴素顺序微调 | 标准对照（无独立发表年份） | 75.632 | 40.758 | 46.711 | 15.891 | 31.938 | 42.186 | 40.286 | 14.783 | 1.775 | 1706.613 | 2.062 | 未测 | 未测 | 未测 | 未测 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) | 64.486 | 46.103 | 50.570 | 9.623 | 33.370 | 40.830 | 45.245 | 17.748 | 5.642 | 1706.613 | 2.062 | 未测 | 未测 | 未测 | 未测 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) | 74.657 | 41.139 | 50.050 | 14.914 | 33.110 | 42.774 | 42.288 | 19.039 | 0.012 | 1846.463 | 141.912 | 未测 | 未测 | 未测 | 未测 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) | 70.697 | 44.202 | 55.738 | 25.223 | 37.241 | 46.620 | 46.406 | 14.726 | -1.732 | 2241.552 | 537.001 | 2959.415† | 未测 | 未测 | 未测 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) | 73.346 | 38.743 | 54.088 | 20.609 | 32.655 | 43.888 | 40.390 | 18.503 | -0.484 | 2273.345 | 568.794 | 未测 | 未测 | 未测 | 未测 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) | 73.907 | 43.894 | 51.709 | 23.982 | 36.196 | 45.937 | 46.357 | 18.932 | -1.616 | 2757.984 | 1053.434 | 未测 | 未测 | 未测 | 未测 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) | 73.596 | 40.921 | 45.942 | 21.328 | 34.520 | 43.261 | 41.829 | 17.145 | -2.992 | 1744.231 | 39.680 | 未测 | 未测 | 未测 | 未测 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

五个数据集列为训练结束后的完整域内QA平均F1；五域宏为这五个域均值等权，AP19为19个原生任务等权平均，FWT/BWT按已测的必要阶段计算。两种权重口径不同。[FOREVER（ACL 2026）正式运行](summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v3/README.md)已通过双骨干模型／保存重载门槛；OneVision已通过完整19阶段／74格／21454题及回放成本核验，Qwen尚无完整核验结果；原调度描述仅为历史，不能据此推断当前进度；Ours尚无可填入此正式表的完整结果。ER-LoRA已按用户要求移出比较，已有结果保留在历史证据中。

类型按主要抗遗忘机制归类，PASs-MoE兼有正则化；LoRA是参数化方式，不单独作为上述方法类型。会议链接指向正式发表页，年份按会议届次。

[方法分类、发表信息及状态](summary/two_backbone_pct01_20261017/BASELINES.md) · [完整主表与协议](summary/two_backbone_pct01_20261017/MAIN_TABLE.md) · [逐题核验索引](summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/VERIFIED_INDEX.json) · [FOREVER额外完整核验](summary/two_backbone_pct01_20261017/forever_verified_20261020/README.md)

<!-- CPU_EDGE_METRICS_NOTE -->
前两项已按8组完整结果的最终checkpoint提取：**原保存态、原dtype、未折叠的完整权重与必需持久状态**，包括视觉、语言、连接器、全部专家/路由、OSFT冻结权重基、PAS掩码及CL-MoE模型外任务锚/任务权重，底层共享存储只计一次。配置可重建的位置/旋转编码缓存、逐次输入激活、视觉缓存、训练保护历史、RNG和文件容器开销另列。增加量相对同骨干原始预训练权重；包含表示和dtype变化，**不等于新增参数量**。MiB=1048576字节。这些数值是保存态权重占用，不是完整checkpoint文件大小、实际推理峰值内存或公平折叠后的压缩比较；没有运行合并、量化或重新评价。只有完成原正式核验并保存最终checkpoint的8组填入数值，未完成与Ours不填估算。GPU列仅填已核验的五窗实测，其余运行指标仍未测。 [逐方法参数数量、dtype、排除项与来源](summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/FINDINGS.md)。
<!-- CPU_EDGE_METRICS_NOTE_END -->

† GPU列为原正式checkpoint、原dtype／表示下的**固定五窗最大 `max_memory_allocated`**：同OLD GPU1、同五条原始QA，每题生成32个新token，包含输入准备与生成；各骨干保留原生输入展开。不是全TEST最坏峰、CPU／NVML总内存或最优折叠部署。[PAS实测](summary/strong_pas_resource_20261025/v1/FINDINGS.md) · [Qwen实测](summary/strong_qwen_resource_20261025/v1/FINDINGS.md)。


<!-- FORMAL_MAIN_TABLE_END -->

<!-- PCT10_MAIN_TABLE_BEGIN -->
## 第二主表：双骨干10%训练／完整测试

训练10%已冻结为**13105 QA**，完整TEST仍为**21454 QA**；seed42、一轮、19任务，主任务共828次优化器调用，其余沿用1%正式设置。**2/16组完整核验：双骨干SeqFT-LoRA。** 当前OLD推进双骨干PASs-MoE及LLaVA CL-MoE；成对监督研究已完成，新版OSFT已保存S0并正常暂停，完整checkpoint已耐久归档；题型辅助研究在普通对照导出时失败，质量缺测；旧run记录保留。题型辅助v2已完成并未改善目标新习得；PaRSP已从S1恢复；条件混合三臂研究已完成。其余组合排队／待既定方法核验。空白不是0分，不复制1%结果或估计未测资源。

| 方法／Backbone | 类型 | 发表会议／年份 | TextVQA<br>自然／通用 | SLAKE<br>医学 | RSVQA-LR<br>遥感 | FinMME<br>金融 | CODA-LM<br>交通 | 五域宏 F1 ↑ | AP19 ↑ | FWT ↑ | BWT ↑ | 最终权重存储<br>保存态 MiB ↓ | 相对原基座增加<br>MiB ↓ | 固定五窗GPU峰值†<br>MiB ↓ | 首词延迟<br>s ↓ | 生成速度<br>tok/s ↑ | 常驻适配器<br>数量 ↓ | 完整参数量<br>系数 | 最终CP<br>MiB | 新习得五域宏F1 ↑ | 新习得AP19 ↑ | 状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Qwen2-VL-2B-Instruct** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 朴素顺序微调 | 标准对照（无独立发表年份） | 83.825 | 45.392 | 63.844 | 33.087 | 38.120 | 52.854 | 48.634 | 14.864 | -5.218 | 4217.462 | 4.156 | 4954.604† | 0.515 | 24.873 | 1 | 2210075136 | 4351.614 | 56.481 | 53.578 | 完整核验通过；最终模型已存 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队，待动态回放GPU预检 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 预检失败：损失／梯度有限性检查未通过 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 已接NEW原队列，等待原生BF16空卡 |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 正式运行 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 预检失败：SVD初始化显存溢出 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 空，不自动运行探索配方 |
| **LLaVA-OneVision-Qwen2-0.5B-OV** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 朴素顺序微调 | 标准对照（无独立发表年份） | 73.069 | 45.294 | 57.157 | 24.602 | 37.189 | 47.462 | 45.874 | 21.156 | -2.933 | 1706.613 | 2.062 | 2176.659† | 0.315 | 31.543 | 1 | 894216224 | 1836.784 | 49.996 | 48.652 | 完整核验通过；最终模型已存 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队，待动态回放GPU预检 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 正式运行 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 已接NEW原队列，等待原生BF16空卡 |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 正式运行 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 已从S1完整CP恢复；S2训练中 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | S0完整CP已保存并耐久归档；已暂停让卡研究 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 空，不自动运行探索配方 |

五域列及五域宏／AP19为最终成绩；新习得两列来自各task训练结束的对角格，按任务等权或域内QA加权再五域等权，均可由原74格得到，不是同一时点的域末模型。BWT仅为继承的描述量，不作优化目标。完整参数、权重／CP与GPU峰须逐组合实测后填写。

第二主表完整覆盖两个backbone各八类基线，共16组；主要报告与1%相同的五域最终F1、宏F1／AP19、FWT／BWT及全部资源指标，另保留完整参数量与CP大小。每组S19完整模型权重、必要方法状态、processor／config及来源记录均须耐久留存；不是只记录参数量或保存LoRA。[最终模型留存](summary/two_backbone_pct10_20261025/FINAL_MODEL_RETENTION.md)

刚习得及金融集中错答是补充分析。LLaVA SeqFT的最终五域宏从1%的42.186升至10%的47.462，四域提高、自然域下降；金融刚习得F1从16.405升至25.414，单选回答却更集中。**集中不能替代新习得判断；不同题型的学习效果也不同。** 数据量、更新预算及先前课程共同变化，不能单独归因样本量。[首组完整证据](summary/two_backbone_pct10_20261025/FIRST_COMPLETE_FINDINGS.md) · [分题型观测与结论](summary/train_answer_prior_20261025/PCT01_PCT10_RESPONSE_FINDINGS.md)

Qwen SeqFT最终五域宏由1%的48.972升至10%的52.854，遥感、金融、交通提高，自然和医学下降；完整参数仍为2,210,075,136，保存态权重4217.462 MiB，新增训练参数1,089,536。更多训练改善总体成绩，但没有均匀改善所有域；尚不能将效果归因于纯数据量。10% GPU资源已按原最终模型完成固定五窗实测。[第二组完整证据](summary/two_backbone_pct10_20261025/SECOND_COMPLETE_FINDINGS.md)

[10%正式协议与冻结证据](summary/two_backbone_pct10_20261025/PROTOCOL.md) · [10%独立主表](summary/two_backbone_pct10_20261025/MAIN_TABLE.md)

<!-- PCT10_MAIN_TABLE_END -->

## 主表怎么读

- **评价口径：** 主表为冻结1%训练集1313条、19阶段、完整21454 TEST、seed42；五域宏是五个域均值等权，AP是19任务等权，两者不混用。已完整核验8/16组，空白不代表0分。
- **最强参照：** 当前已核验的LLaVA-OneVision内五域宏最高为PASs-MoE **46.619976**；跨骨干最高为Qwen顺序LoRA **48.972351**。新方法需同时报告这两个比较范围，不能把PAS无条件称为全体最强。
- **基线接续：** 新服务器由队列使用GPU1/2/3，只保留GPU0不由我方分配；移除了不必要的队列屏障。原基线继续运行，新完整核验结果出来后补表。[运行与队列详情](summary/baseline_capacity_20261025/v1/FINDINGS.md)。

## 探索命题与结论

连续学习先导通常采用1213 FIT／100 DEV、19阶段；其它诊断使用各自固定范围。DEV已用于多次探索，结果用于筛选，不能与正式TEST分数直接排名，也不据此宣称显著提升。下表区分实际负结果、局部支持、质量缺测和待验证假说。

| 观测的问题与探索命题 | 检验结果 | 结论与设计影响 |
| --- | --- | --- |
| **强方法主要解决遗忘，还是先学得更好？** | 正式刚习得均分：OSFT47.887821、Seq38.604705；BWT分别−1.616191、+1.774861 | **支持优先研究有效习得。** OSFT在本轨迹上刚习得更强，保持更差；尚未识别其性能原因，不能把所有缺口归为遗忘。[证据](summary/final_weight_reduction_20261024/osft_initial_gap_check_v1/FINDINGS.md) |
| **低曝光新专家的冷启动能否靠继承改善？** | 局部11条医学QA／一次更新，从头19.88、完整继承29.26；独立功能中性控制没有复现完整继承收益 | **局部支持功能继承。** 不支持“所有方法每域清零”或“只开放首步A梯度就够”。[冷启动](summary/onevision_diagnostics_20261016/expert_cold_start_20261018/RESEARCH_FINDINGS.md) · [控制](summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v2/function_vs_factor/RESEARCH_FINDINGS.md) |
| **OSFT部分训练与保护机制是否可省？** | ShortFixed宏42.301329、原OSFT40.783801，梯度及Adam状态少21.34%；去逐步保护42.333991，但金融下降 | **局部支持简化训练状态，未证明最终模型更小。** 点估计不证明等效或保护无用；阶段刷新对照故障，质量缺测。[短侧](summary/edge_deployment_20261019/osft_short_factor_control_20261022/v2/FINDINGS.md) · [保护](summary/edge_deployment_20261019/osft_protection_control_20261022/v1/FINDINGS.md) |
| **增加更新位置就能稳定胜强基线？** | all-r8先导41.952549；完整确认五域宏44.894423，低于正式OSFT、PAS | **本配方被否证。** 先导入选未转成正式优势，不按TEST追配置。[完整负结果](summary/edge_deployment_20261019/osft_increment_confirmation_20261022/v1/FINDINGS.md) |
| **借原生FFN响应做轻量调制有净收益？** | 动态44.757603、固定44.681201；匹配宽q/v为45.085963，分别高0.328360／0.404762pp | **未支持新增调制的净质量收益。** 原生响应非退化并不等于有答题价值；宽q/v保留为容量对照，正式推进已关闭。[匹配结果](summary/native_response_selection_20261025/qv_control/FINDINGS.md) |
| **加宽带来的收益能否算新增模块创新？** | 匹配GPU朴素r8宏38.775370，宽q/v45.085963；宽模型训练坐标约8.14倍 | **只能作为容量对照。** wide高6.310593pp不证明新模块有效，不覆盖旧实现r8的历史分数，也不是正式TEST优势。[完整对照](summary/plain_r8_gpu_control_20261025/v1/FINDINGS.md) |
| **低损伤排名、低秩或局部拟合能直接承接回答功能？** | 联合删8个FFN：43.791987→19.620168；FFN低秩／功能拟合多组虽减重8.61%–25.83%，F1仍崩坏 | **这些固定替代方案保质失败。** 局部误差下降、单删低损伤都不足以证明完整图可删；不否定所有压缩。[删图](summary/ffn_direct_ablation_20261025/v1/FINDINGS.md) · [角色对照](summary/final_weight_reduction_20261024/ffn_role_retention_v1/README.md) · [读出](summary/final_weight_reduction_20261024/functional_readout_carrier_v1/README.md) |
| **换更小预训练载体，就能同时保质且省显存？** | 小视觉接口宏38.966256<父43.791987；完整SmolVLM学习后37.991536，权重少43.21%，五窗GPU峰却高14.09%–57.43% | **本组质量／资源联合目标失败。** 参数少不能代替GPU峰实测，预训练完整也不保证功能承接。[视觉接口](summary/vision_interface_transfer_20261025/v1/FINDINGS.md) · [原生小模型](summary/native_prior_learning_20261025/compatibility/INTERPRETATION.md) |
| **OSFT优势是图文配对利用整体更强？** | 61题四状态488次NLL，配对支持改善OSFT相对q/v差−0.013734 | **未支持预定解释。** 不能反推视觉无用或可以删视觉编码器。[证据](summary/final_weight_reduction_20261024/paired_image_support_diagnostic_v1/RESULT_FINDINGS.md) |
| **历史隐藏态约束应否先检查它对当前回答是否有利？** | DEV宏：朴素r8 38.775370、原法 43.703262、答案门 43.719891、同数打乱 43.671936 | **微小正向点估计，尚不足确立主创新。** 门仅高原法0.016628pp、高打乱0.047954pp；四个非交通域F1相同。主要收益已有于共同回放／EMA配方，不能归新门。[完整结果](summary/answer_reliability_consolidation_20261025/v1/FINDINGS.md) |
| **共同回放配方的终态收益来自更好的新习得？** | 两方法医学刚习得均36.857143、遥感均65；40条逐题F1全同，0高／0低／40平 | **本次两域不支持该解释。** 终态差在后续形成，不将约4.93pp宏收益归为更会学新任务。按人类要求，后续只查新习得原因，不开保持／遗忘分支。[证据与独立评判](summary/acquisition_readout_20261025/v1/FINDINGS.md) |
| **金融TRAIN标签扩大到10%后是否更均衡？** | 单选金标A/B/C/D：1%为22/18/7/6，10%为189/163/99/76；最大占比41.509%→35.863% | **监督偏斜减弱但仍存在。** 不能据计数认定模型塌缩原因或自动做先验校正；模型观测见下一行。[完整比较与独立审阅](summary/train_answer_prior_20261025/INTERPRETATION.md) |
| **10%会消除金融集中错答、均匀改善新习得吗？** | LLaVA／SeqFT刚习得F1：16.405→25.414；单选不同回答174→9，但原满分331→495、零分966→812。10%多选仍286题严格单字母且原token F1全零（非选项集合正确率），数值训练前8.640→刚学完6.974 | **集中与更高F1可以并存，题型习得并不均匀。** 不支持把集中直接当作失败或统一位置偏置原因；单组结果不推广全部基线，也不隔离纯数据因果。[完整同ID比较与独立评判](summary/train_answer_prior_20261025/PCT01_PCT10_RESPONSE_FINDINGS.md) |
| **10%是否普遍提高各任务刚习得，而不只是金融？** | 同LLaVA／SeqFT的19/19任务刚习得F1均更高；五域刚习得宏42.026→49.996。阶段内增量13高6低，10%医学Position／Plane自身阶段分别下降4.859／0.718pp | **同容量的更充分学习配置广泛提高刚习得水平，但当前任务更新并非都更有效。** 不把1%低分普遍认定为容量上限；前课程、数据与更新共同改变，非纯数据因果。[全19任务、参数空间与解释](summary/train_answer_prior_20261025/ALL_TASK_ACQUISITION_FINDINGS.md) |
| **两个医学任务的阶段降分都意味着内容退化吗？** | 全491题配对、208项回答变化经双agent盲读。Plane原F1−0.718pp而EXACT+0.862pp；Position的未知含义变化承载77.75%总失分量 | **不能统一归因。** Plane有明确表述效应，Position主因未识别；不据此选择统一格式／知识模块，EXACT也非语义正确率。[完整正负与未知](summary/train_answer_prior_20261025/MEDICAL_PAIRED_FINDINGS.md) |

**跨骨干的新结论：更多数据／更新提高总体新习得，但局部失败不固定在同一题型。** Qwen刚习得五域宏46.985→56.481，19任务中17提高、2下降（Quantity、Shape）；10%时Shape阶段仍下降1.111。LLaVA在10%中下降的Position／Plane，Qwen却分别提高3.425／7.447。因此不支持“这两类医学任务统一学不会”或一个共同根因；能保留的是固定配置下学习收益不均匀。两比例同时改变数据、更新与课程，Qwen历史训练硬件也不同，不能作纯数据因果。[Qwen完整对照](summary/train_answer_prior_20261025/QWEN_ACQUISITION_FINDINGS.md) · [LLaVA完整对照](summary/train_answer_prior_20261025/ALL_TASK_ACQUISITION_FINDINGS.md)

**PAS的优势包含跨域新习得，但不能归因于路由。** 正式1%全19任务补齐后，PAS刚习得五域宏48.349，普通LoRA为42.026；医学／遥感／金融／交通更高，自然更低，任务15高4低。但交通T18／T19刚习得更高、阶段增量反而更小：继承起点不同，不能把终点高分统一称为本阶段学得更多。[完整观测与局限](summary/conditional_shared_budget_20261026/PAS_ACQUISITION_FINDINGS.md)

**强方法改善的金融答题功能不同。** 同样1%数据，546道多选刚习得原F1：SeqFT／OSFT均为0，PAS为13.736（75题原F1满分）；严格单字母输出分别504／545／127题。OSFT的金融优势来自单选，PAS则同时获得单选和多选收益；三法数值题阶段均值都下降。不能统一解释为“数据太少学不会”，也不能把PAS差异直接归因于专家路由。[全部题型、前后读数与负贡献](summary/train_answer_prior_20261025/FINANCE_STRONG_ACQUISITION_FINDINGS_20261027.md)

**不能直接用隐藏题型或显式关键词分流。** 原输入不传入题型metadata；预定单／多选关键词在1%金融TRAIN的89题全部未命中，10%的887题仅9题命中多选提示（8多选、1单选）。这否定该字面规则作为可靠分流依据，不等于语义不可区分，也不证明输入歧义就是失败原因。[输入信号及方法约束](summary/train_answer_prior_20261025/FINANCE_TRAIN_TYPE_CUES_FINDINGS_20261027.md)

**已检验命题：同样的小预算，条件适应是否比简单权重混合值得？** 三臂完整结果不支持保留当前soft点：DEV最终宏shared／uniform／soft为38.623／39.663／39.456，soft低于uniform且显存略高。它的新习得收益集中在一条医学表述变化、一条金融选项变化，不能称普遍知识改善；uniform相对shared最终约96.1%的净提升来自同一条遥感二元题。当前点关闭，三臂完整参数及正负证据均保留，Ours仍空。[完整对照、逐题贡献与资源](summary/conditional_shared_budget_20261026/RESULT_FINDINGS_20261027.md)

**跨骨干金融题型：多选现象复现，数值降分不复现。** Qwen普通LoRA在1%／10%的多选刚习得F1为0／10.989，546题中严格单字母为403／259，与LLaVA的0／8.425及504／286有共同现象。但Qwen数值题两个比例的当前阶段均提高（+0.526／+0.789），不复现LLaVA的阶段下降。不能把跨比例终点变化当当前训练损伤，也不能据此认定统一题型混淆根因。该观察不单独证明题型监督有效。[完整题型、前后变化与边界](summary/train_answer_prior_20261025/QWEN_FINANCE_TYPES_FINDINGS_20261027.md)。

**PAS的多选收益不等于普遍学会完整组合。** 546题中，PAS刚习得输出长度只有1／2／4（127／198／221题），没有三字母输出，但合法参考含162道三字母题。75题原串匹配之外，还有176题仅多选、151题仅缺选、91题同时错选漏选；不能把它的高分直接归因精准选项组合或复杂专家必要。普通LoRA有353题输出是gold成员子集，但均匀字母本来也常是成员，不能据此断言内容已经学会或只需续写。研究FIT的17多选已覆盖2／3／4成员（9／4／4题），但不能据字符覆盖推出合法token负候选；成员辅助头和负损失均暂不推进，避免将PAS过选直接当小LoRA漏选的干预依据。[候选评判](summary/train_answer_prior_20261025/ANSWER_CONSTRUCTION_DECISION_20261027.md)。[完整观测与边界](summary/train_answer_prior_20261025/FINANCE_MULTICHOICE_LITERAL_FINDINGS_20261027.md)。

## 参数量与已测GPU优势

当前候选使用 **LLaVA-OneVision-Qwen2-0.5B-OV**：Qwen2-0.5B-Instruct语言骨干＋SigLIP视觉编码器，原完整模型 **893,675,552** 个系数。下表是已核验checkpoint的原保存表示，包含整个模型与必需持久状态，不是只计adapter。

| 骨干／方法 | 完整保存系数数量 | 权重存储 MiB |
| --- | ---: | ---: |
| LLaVA-OneVision／SeqFT-LoRA | 894,216,224 | 1706.613 |
| LLaVA-OneVision／FOREVER | 894,216,224 | 1706.613 |
| LLaVA-OneVision／CL-MoE | 930,877,024 | 1846.463 |
| LLaVA-OneVision／PASs-MoE | 1,034,447,216 | 2241.552 |
| LLaVA-OneVision／PaRSP | 893,675,552 | 2273.345 |
| LLaVA-OneVision／OSFT | 990,913,568 | 2757.984 |
| LLaVA-OneVision／SpaRTA | 904,077,344 | 1744.231 |
| Qwen2-VL／SeqFT-LoRA | 2,210,075,136 | 4217.462 |

保存dtype与表示不同，**存储增加不等于新增参数量**；普通LoRA与OSFT常规fold后都可回原稠密形状，不能拿OSFT未折叠保存态制造独占减重优势。[完整统计口径](summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/FINDINGS.md)。

**GPU实测成立的范围：** 已完成宽q/v对原完整PAS的同卡、同五题、各32新token测量。宽q/v最终893,675,552个系数，比PAS少**13.61%**；原dtype权重字节少**23.96%**；五窗口GPU峰 **1857–2166 MiB**，PAS为 **2490–2959 MiB**，逐窗低**24.84%–26.81%**。包含原FP32专家／FP16基座的dtype差异，仅是这五窗的原实现比较；未证明同dtype最优部署或全输入峰。[实测证据](summary/native_response_selection_20261025/qv_control/RESOURCE_COMPARISON.json) · [参数与dtype影响拆分](summary/edge_resource_comparison_20261025/v1/PRECISION_FAIRNESS_REVIEW.md)。

**全体最强基线的资源锚也已补齐：** 原Qwen顺序LoRA完整图在同卡／同五条原始QA／32-token窗实测为4312–4955 MiB；匹配LLaVA朴素r8为1857–2166 MiB，逐窗少55.15%–58.80%。完整系数较Qwen少59.56%，主要来自骨干与表示差异，不能称新增门的结构贡献。Qwen保原FP16基座＋FP32 LoRA、未fold；各自原生输入展开不同，非最优部署或全分布峰。[完整证据](summary/strong_qwen_resource_20261025/v1/FINDINGS.md) · [三组资源对照图](summary/edge_resource_comparison_20261025/v1/FINDINGS.md)。

**这仍不是Ours的联合成功：** 宽q/v训练参数4,399,104，比朴素r8的540,672多3,858,432，约8.14倍；合并后均无新增推理参数。其F1是DEV结果，尚未证明正式质量胜最强基线。

## 最新探索结论：当前条件混合点关闭

**问题与方法。** PAS有跨域新习得优势但持久专家成本较高；我们检验能否用540,672个训练坐标承接这项收益。普通shared r8、固定4/2/2混合uniform、A响应能量加权soft共用初始化与训练暴露。soft没有额外router参数，但保留条件分支；简单两臂实际fold。借鉴已有机制本身不是否决理由，取舍依据是以下实测结果。

| 内部探索臂 | 刚习得宏F1 | 最终宏F1 | 完整系数 | 权重MiB | 五窗GPU峰MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared | 43.252 | 38.623 | 893,675,552 | 1704.551 | 2166.101 |
| uniform | 42.367 | 39.663 | 893,675,552 | 1704.551 | 2166.101 |
| soft | 43.819 | 39.456 | 894,216,224 | 1706.613 | 2168.163 |

**为什么不保留soft？** 它相对uniform最终低0.208pp，权重与allocated峰各多2.0625 MiB；前四域逐题F1相同，最终差异全在交通。新习得虽高1.452pp，但医学增益仅来自一条中英文表述变化，金融增益仅来自一题且shared已同样答对。uniform最终高shared1.041pp，也主要由一条遥感题贡献，不能包装为新框架成功。

这是1213 FIT／100 DEV、单种子的完整探索，**不是正式主表结果**。阶段起点不同，不把增量差当同起点学习效率；固定五窗峰不是全输入最坏值。关闭本固定soft方案，保留全部参数和证据；不推广为所有条件机制无效，不调设置救结果。[完整结论及核验记录](summary/conditional_shared_budget_20261026/RESULT_FINDINGS_20261027.md)

**成对选项监督：仅见弱局部信号，尚不足以进入正式实验。** 两组完整训练与评估后，刚习得宏F1由43.274升至44.252，但正收益仅来自一道金融题；最终宏F1仅由38.701升至38.772。12题置换诊断中稳定且严格正确从1题增至2题，稳定且错误均为3题。干预前15阶段参数摘要并不相同，不能把全部差异单独归因于配对监督。两组完整权重均为893,675,552系数／1704.551 MiB，固定五窗GPU峰均2166.101 MiB：无新增部署成本，也没有相对普通小LoRA的内存优势。这是已探索DEV的单seed结果，Ours保持空，不追调当前配方。[设计](summary/train_answer_prior_20261025/PAIRED_OPTION_DESIGN_20261027.md) · [结果审阅](summary/train_answer_prior_20261025/PAIRED_RESULT_REVIEW_20261027.md)

**已结束的答案门探索：** 原法、答案门、同数打乱的DEV宏分别43.703262、43.719891、43.671936；新增门只高原法0.016628pp、高打乱0.047954pp。共同配方和普通r8的医学／遥感刚习得分别完全相同（36.857143／65），因此不能将终态改善归为这两域的新习得收益，也不足以确立新增门为主创新。这条探索已结束，不是当前正在验证的方法。[完整正负结果](summary/answer_reliability_consolidation_20261025/v1/FINDINGS.md) · [新习得对照](summary/acquisition_readout_20261025/v1/FINDINGS.md)

**位置诊断已完成：换选项位置后，所选内容常发生变化，10%也未消除这一现象。** 固定45条共同TRAIN、一次循环，在金融刚学完状态，1%有32/37、10%有31/40条有效配对改变所选内容；两比例共同有效的33题中为28→24。训练前有效配对各仅2题，无法据此判定金融训练如何改变位置敏感性。稳定选择也可能选错；这不是正确率或所有金融错误的统一原因，不自动推出PIF训练方案。[完整结果与边界](summary/train_answer_prior_20261025/POSITION_FINDINGS.md)

---

[探索过程与证据索引](SUPPLEMENTARY_NOTES.md#探索过程与证据索引) · [完整实验口径与数据来源](SUPPLEMENTARY_NOTES.md) · [历史研究记录](README_HISTORY_20261024.md)

**题型辅助监督未改善目标新习得。** 同一S15起点，Plain／真实题型／打乱题型的刚习得宏F1为43.255／43.127／43.121，最终宏为38.633／38.740／38.716；金融刚习得均15，多选4题仍全为单字母且原F1为零。真实标签的最终微增全来自交通，与打乱标签仅差0.023，不能证明目标功能有效。三组完整系数893,675,552、权重1704.551 MiB、五窗GPU峰2166.101 MiB一致；训练头新增2691参数、部署移除，仅证明无新增部署成本。关闭此固定方案，不推进正式表；单seed小DEV不支持题型信息普遍无用的结论。[完整正负结果](summary/train_answer_prior_20261025/TYPE_AUXILIARY_V2_FINDINGS_20261027.md) · [独立审阅](summary/train_answer_prior_20261025/TYPE_AUXILIARY_V2_RESULT_REVIEW_20261027.md)

10% Qwen OSFT在SVD初始化预检发生显存溢出，保留为固定条件下的失败；它不是推理显存或质量分数。[失败证据](summary/two_backbone_pct10_20261025/QWEN_OSFT_INITIALIZATION_FAILURE_20261027.md)

[题型辅助研究首次实际启动及基线暂停留存证据](summary/train_answer_prior_20261025/TYPE_AUXILIARY_FIRST_STARTED_20261027.json)。

[题型辅助导出故障独立审阅](summary/train_answer_prior_20261025/TYPE_AUXILIARY_EXPORT_FAILURE_REVIEW_20261027.md)；完整S15与失败现场已耐久归档，141文件逐SHA核对，原始记录保留。

10% SeqFT资源列：原最终未折叠模型、同卡五条固定QA、每条32新token；GPU为五窗最大allocated峰，TTFT与解码速度为五窗中位数。计时包含同步观测开销，不是全TEST最坏峰或生产吞吐。[完整实测与核验](summary/two_backbone_pct10_20261025/FINAL_RESOURCE_FINDINGS_20261027.md)。

三臂完整最终模型、S15、答案及失败历史已耐久归档，309文件逐SHA校验。PaRSP已从S1恢复并进入S2。[归档证据](summary/train_answer_prior_20261025/TYPE_AUXILIARY_V2_ARCHIVE_20261027.json)。
