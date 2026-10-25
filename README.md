# LLMCL：1%与10%数据下的多领域多模态持续学习

**研究目标：总体 Answer F1 略高于最强基线，同时完整模型权重与 GPU 内存显著更小。** 允许域间取舍，五域分别报告；重点是论文方法与实测资源优势，不做端侧设备适配。

**研究重点（人类最新指示）：只考虑新习得，不研究遗忘。** 后续问题是现有方法为什么没有学好新任务、如何使1%与10%数据下的学习更有效；内存／参数量作为最终评估，不替代学习原因证据。

**设置确定后固定。** 沿用既定数据、训练、生成与评分条件，研究方法在这些条件下的新习得能力、失败和适用边界，不为改善结果反复调整设置。

**当前结论：尚未得到同时满足质量与资源目标的自研方法，Ours 暂空。** 已有1%完整正式结果8组；10%四组已通过GPU预检与重载核验，正式运行处于初始评价，尚无完整正式结果；单图LoRA相对PAS与Qwen的固定窗口显存差异已实测，新方法质量仍待验证。

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

训练10%已冻结为**13105 QA**，完整TEST仍为**21454 QA**；seed42、一轮、19任务，主任务共828次更新，其余沿用1%正式设置。OLD四卡两骨干SeqFT-LoRA与PASs-MoE均已通过preflight及reload（return0），进入**正式运行：初始评价（S0）**；其余8个native组合OLD排队，两FOREVER待10%动态回放适配核验、两SMoLoRA待NEW。本表没有10%成绩或资源测量，空白不是0分，也不复制1%结果。

| 方法／Backbone | 类型 | TextVQA 自然／通用 | SLAKE 医学 | RSVQA-LR 遥感 | FinMME 金融 | CODA-LM 交通 | 最终五域宏F1 ↑ | 最终AP19 ↑ | 新习得五域宏F1 ↑ | 新习得AP19 ↑ | FWT | BWT（描述） | 完整参数量（系数） | 完整权重存储 MiB | 最终CP MiB | 固定五窗GPU峰 MiB | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Qwen2-VL-2B-Instruct** | | | | | | | | | | | | | | | | | |
| SeqFT-LoRA | 朴素顺序微调 | | | | | | | | | | | | | | | | 正式运行：初始评价 |
| FOREVER | 回放（多模态适配） | | | | | | | | | | | | | | | | 待10%动态回放适配核验 |
| CL-MoE | 架构 | | | | | | | | | | | | | | | | OLD排队 |
| SMoLoRA | 架构 | | | | | | | | | | | | | | | | 已freeze，待NEW原生BF16空卡 |
| PASs-MoE | 架构／正则 | | | | | | | | | | | | | | | | 正式运行：初始评价 |
| PaRSP | 正则 | | | | | | | | | | | | | | | | OLD排队 |
| OSFT | 正则 | | | | | | | | | | | | | | | | OLD排队 |
| SpaRTA | 正则 | | | | | | | | | | | | | | | | OLD排队 |
| Ours | 尚无正式自研方法 | | | | | | | | | | | | | | | | 空，不自动运行探索配方 |
| **LLaVA-OneVision-Qwen2-0.5B-OV** | | | | | | | | | | | | | | | | | |
| SeqFT-LoRA | 朴素顺序微调 | | | | | | | | | | | | | | | | 正式运行：初始评价 |
| FOREVER | 回放（多模态适配） | | | | | | | | | | | | | | | | 待10%动态回放适配核验 |
| CL-MoE | 架构 | | | | | | | | | | | | | | | | OLD排队 |
| SMoLoRA | 架构 | | | | | | | | | | | | | | | | 已freeze，待NEW原生BF16空卡 |
| PASs-MoE | 架构／正则 | | | | | | | | | | | | | | | | 正式运行：初始评价 |
| PaRSP | 正则 | | | | | | | | | | | | | | | | OLD排队 |
| OSFT | 正则 | | | | | | | | | | | | | | | | OLD排队 |
| SpaRTA | 正则 | | | | | | | | | | | | | | | | OLD排队 |
| Ours | 尚无正式自研方法 | | | | | | | | | | | | | | | | 空，不自动运行探索配方 |

五域列及五域宏／AP19为最终成绩；新习得两列来自各task训练结束的对角格，按任务等权或域内QA加权再五域等权，均可由原74格得到，不是同一时点的域末模型。BWT仅为继承的描述量，不作优化目标。完整参数、权重／CP与GPU峰须逐组合实测后填写。

核心检验是：扩大训练后，刚学完金融任务时的集中错答是否减少、正确率是否同步提高。10%沿用单轮，数据量和更新预算同时增加；改善只能说明本学习配置缓解了问题，不能单独证明样本量是唯一原因。当前尚无10%质量结论。[比较口径](summary/train_answer_prior_20261025/ACQUISITION_COMPARISON_PLAN.md)

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
| **金融TRAIN标签扩大到10%后是否更均衡？** | 单选金标A/B/C/D：1%为22/18/7/6，10%为189/163/99/76；最大占比41.509%→35.863% | **监督偏斜减弱但仍存在。** 不能据计数认定模型塌缩原因；10%刚习得表现仍待实测，不自动做先验校正。[完整比较与独立审阅](summary/train_answer_prior_20261025/INTERPRETATION.md) |

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

## 当前研究：只解释新习得不足

已完成的答案门探索借鉴[I-LoRA](https://arxiv.org/abs/2402.18865)的快慢适配器、回放与隐藏态一致性，针对1%多模态QA检验一项简单适配：将**实际一致性目标**送入同一冻结答案头；只有它的答案NLL低于当前状态时，才开启该QA的一致性约束。已有Seq正BWT及强方法的习得差距促成这个问题，但还没有证明I-LoRA原约束在本数据上有害。选择性蒸馏已有先例；当前贡献必须由实际max-hidden目标的读出适配及完整对照建立。[文献与边界](summary/answer_reliability_consolidation_20261025/v1/RELATED_WORK_SCOPE.md)。

固定q/v-r8 **540,672个训练参数**，门新增**0**参数；训练时另有540,672个冻结慢适配器系数及回放、teacher和head计算。三臂实际最终只保留working并fold，完整图均为893,675,552个系数／1,787,351,104权重bytes，普通buffers9,032bytes，新增推理参数0；同卡五窗GPU峰均为1857–2166 MiB，与匹配朴素r8逐字节相同。相对保留专家库的强方法可能省内存，相对同样fold的朴素LoRA／OSFT不声称新的结构减重。

三臂已完整结束。100个已探索DEV的五域宏为：朴素r8 **38.775370**，原法 **43.703262**，答案门 **43.719891**，同接受数打乱 **43.671936**。门高原法仅**0.016628pp**、高打乱**0.047954pp**，差异全部来自交通；接受425／886次回放，打乱Hamming392，事前点方向成立，但不称显著或稳定。相对朴素的约4.94pp改善主要在共同配方中就已存在；遥感+35pp、医学−10pp，不能归新门，也没有证明正式胜最强。末层NLL仍只是整包约束的代理。[完整结果与独立评判](summary/answer_reliability_consolidation_20261025/v1/FINDINGS.md)。

**新习得观测已经补齐：** 原共同配方与朴素r8的医学刚习得均36.857143、遥感均65，40条逐题F1全同；终态的遥感+35pp、医学−10pp不是这两域刚习得分差。不能把共同配方的终态总体收益解释为新习得改善，新增门也没有取得主创新证据。[完整观测](summary/acquisition_readout_20261025/v1/FINDINGS.md)。

**后续只研究新习得：** 依人类最新要求，不再设计遗忘／保持／历史保护实验。现有方法为什么学不好仍需训练前到刚学完的功能获得及匹配干预来解释；少样本／少更新的计数本身不能充当原因。当前没有已确证的新学习机制或正式确认，Ours仍空。

---

[探索过程与证据索引](SUPPLEMENTARY_NOTES.md#探索过程与证据索引) · [完整实验口径与数据来源](SUPPLEMENTARY_NOTES.md) · [历史研究记录](README_HISTORY_20261024.md)
