# LLMCL：少数据多模态持续学习与紧凑模型研究

> 更新：2026-10-25。本文集中记录已有成果、负结果、尚未解释的问题和新会话接续入口。细节见[补充说明](SUPPLEMENTARY_NOTES.md)；原README时间线保存在[历史归档](README_HISTORY_20261024.md)（数据样例省略，原件私有保留），其中“当前／下一步／正在运行”均为历史记录。

## 先看结论与当前状态

- **已经取得：** 双骨干正式对照中的8组完整核验结果、对应最终checkpoint的完整权重统计、多项习得／冷启动／回答行为诊断，以及一系列完整减重但质量失败的对照。失败结果和费用均保留。
- **仍未做到：** 尚无最终完整权重更小、又保留强参照回答质量的自研框架。也尚未因果解释OSFT习得优势及ShortFixed点估计表现的原因，不能把性能表或局部误差直接当机制。
- **最新完整结果：** 合法wide-q/v已学父的8个FFN真实删除对照已完成：完整权重少199.5MiB（11.704%），五个匹配工作负载GPU峰allocated少193.0–200.2MiB，但DEV宏F1从43.791987降至19.620168，五域全降；关闭该构造。原阶段刷新已因磁盘配额故障终止，没有完整质量结论。[完整结果](summary/ffn_direct_ablation_20261025/v1/FINDINGS.md)

研究目标是**更少的最终完整模型权重，同时尽量保留或提高原Answer F1**。可训练参数量和训练状态作为辅助指标；端侧是科研动机，不做设备迁移、部署交付或精度卡常。已有技术可以借用，贡献由具体功能问题、必要对照和证据决定；不能因有先例就否决，也不能因换背景就宣称原创。

## 实验口径：三类结果不能混排

| 类别 | 数据与评价 | 如何使用 |
| --- | --- | --- |
| 正式持续学习 | 原1313 TRAIN／19阶段／完整21454 TEST，seed42，原Answer F1 | 仅完整核验后填正式AP／FWT／BWT；TEST不能选方案 |
| FIT→DEV先导 | 原训练池分出的1213 FIT／100 DEV，19阶段87次更新 | DEV已反复探索，单seed、每域20题、89个已知组；只筛选，不当新留出或稳定收益 |
| 历史诊断 | 11／155／228／507题或S1合法FIT子集等各自固定范围 | 仅支持本诊断的局部结论，不与前两类直接排名 |

唯一指标为原始token Answer F1（0–100）。**AP为19任务等权，五域宏是五个域均值再等权；两者不同。** T7／8／9／12没有探索DEV。用户尚未给定可容许掉分范围，因此没有“非劣／等效”结论。正式1313 teacher见过探索DEV，即使只在FIT蒸馏，也不能拿来筛学生。

正式完成数最后一次证明检查仍为**8／16**：OneVision七组、Qwen顺序LoRA一组；这是已核验数量，不是其它作业当前存活状态。Ours没有正式完整结果。

<!-- FORMAL_MAIN_TABLE_BEGIN -->
## 主实验：双骨干1%多领域持续学习

Acc采用唯一的**Answer F1（0–100）**。训练集为冻结的1%（1313条），测试集为完整21454条；五个数据集按19个原生任务顺序训练，seed=42。空白表示尚无完整核验结果，不能视作0分。

| 方法／Backbone | 类型 | 发表会议／年份 | TextVQA<br>自然／通用 | SLAKE<br>医学 | RSVQA-LR<br>遥感 | FinMME<br>金融 | CODA-LM<br>交通 | AP ↑ | FWT ↑ | BWT ↑ | 最终权重存储<br>保存态 MiB ↓ | 相对原基座增加<br>MiB ↓ | 推理峰值内存<br>MiB ↓ | 首词延迟<br>s ↓ | 生成速度<br>tok/s ↑ | 常驻适配器<br>数量 ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen2-VL-2B-Instruct** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 朴素顺序微调 | 标准对照（无独立发表年份） | 86.312 | 47.654 | 51.110 | 26.576 | 33.210 | 47.812 | 13.258 | 4.415 | 4217.462 | 4.156 | 未测 | 未测 | 未测 | 未测 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| **LLaVA-OneVision-Qwen2-0.5B-OV** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 朴素顺序微调 | 标准对照（无独立发表年份） | 75.632 | 40.758 | 46.711 | 15.891 | 31.938 | 40.286 | 14.783 | 1.775 | 1706.613 | 2.062 | 未测 | 未测 | 未测 | 未测 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) | 64.486 | 46.103 | 50.570 | 9.623 | 33.370 | 45.245 | 17.748 | 5.642 | 1706.613 | 2.062 | 未测 | 未测 | 未测 | 未测 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) | 74.657 | 41.139 | 50.050 | 14.914 | 33.110 | 42.288 | 19.039 | 0.012 | 1846.463 | 141.912 | 未测 | 未测 | 未测 | 未测 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) | 70.697 | 44.202 | 55.738 | 25.223 | 37.241 | 46.406 | 14.726 | -1.732 | 2241.552 | 537.001 | 未测 | 未测 | 未测 | 未测 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) | 73.346 | 38.743 | 54.088 | 20.609 | 32.655 | 40.390 | 18.503 | -0.484 | 2273.345 | 568.794 | 未测 | 未测 | 未测 | 未测 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) | 73.907 | 43.894 | 51.709 | 23.982 | 36.196 | 46.357 | 18.932 | -1.616 | 2757.984 | 1053.434 | 未测 | 未测 | 未测 | 未测 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) | 73.596 | 40.921 | 45.942 | 21.328 | 34.520 | 41.829 | 17.145 | -2.992 | 1744.231 | 39.680 | 未测 | 未测 | 未测 | 未测 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

五个数据集列为训练结束后的完整域内QA平均F1；AP为19个原生任务等权平均，FWT/BWT按已测的必要阶段计算。两种权重口径不同。[FOREVER（ACL 2026）正式运行](summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v3/README.md)已通过双骨干模型／保存重载门槛；OneVision已通过完整19阶段／74格／21454题及回放成本核验，Qwen尚无完整核验结果；原调度描述仅为历史，不能据此推断当前进度；Ours尚无可填入此正式表的完整结果。ER-LoRA已按用户要求移出比较，已有结果保留在历史证据中。

类型按主要抗遗忘机制归类，PASs-MoE兼有正则化；LoRA是参数化方式，不单独作为上述方法类型。会议链接指向正式发表页，年份按会议届次。

[方法分类、发表信息及状态](summary/two_backbone_pct01_20261017/BASELINES.md) · [完整主表与协议](summary/two_backbone_pct01_20261017/MAIN_TABLE.md) · [逐题核验索引](summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/VERIFIED_INDEX.json) · [FOREVER额外完整核验](summary/two_backbone_pct01_20261017/forever_verified_20261020/README.md)

<!-- CPU_EDGE_METRICS_NOTE -->
前两项已按8组完整结果的最终checkpoint提取：**原保存态、原dtype、未折叠的完整权重与必需持久状态**，包括视觉、语言、连接器、全部专家/路由、OSFT冻结权重基、PAS掩码及CL-MoE模型外任务锚/任务权重，底层共享存储只计一次。配置可重建的位置/旋转编码缓存、逐次输入激活、视觉缓存、训练保护历史、RNG和文件容器开销另列。增加量相对同骨干原始预训练权重；包含表示和dtype变化，**不等于新增参数量**。MiB=1048576字节。这些数值是保存态权重占用，不是完整checkpoint文件大小、实际推理峰值内存或公平折叠后的压缩比较；没有运行合并、量化或重新评价。只有完成原正式核验并保存最终checkpoint的8组填入数值，未完成与Ours不填估算。其余运行指标仍未测。 [逐方法参数数量、dtype、排除项与来源](summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/FINDINGS.md)。
<!-- CPU_EDGE_METRICS_NOTE_END -->

<!-- FORMAL_MAIN_TABLE_END -->

## 为什么强参照学得好：已有证据与未解问题

| 问题 | 已有数据／干预 | 现在可以说什么 |
| --- | --- | --- |
| OSFT优势主要在习得还是保持？ | 正式19任务刚习得均分OSFT **47.887821**、Seq **38.604705**；S0为15.050963／15.098792，BWT为−1.616191／+1.774861。[证据](summary/final_weight_reduction_20261024/osft_initial_gap_check_v1/FINDINGS.md) | OSFT在这条轨迹上刚习得更高，但保持指标更差；简单“起点分数高了同样几分”不足以解释。位置、坐标量、LR、分解和保护混杂，尚未识别原因或具体知识 |
| 所有方法都每域从零、几乎没更新吗？ | 11条医学QA、一次更新的局部新LoRA：从头19.88，完整继承29.26；冷组48层A首步不更新，继承后均更新。[局部证据](summary/onevision_diagnostics_20261016/expert_cold_start_20261018/RESEARCH_FINDINGS.md) | 局部新专家存在弱习得，继承有帮助；**Seq和OSFT跨阶段连续，不是每域清零**。不能推广为所有基线共有故障 |
| 只是首步A没有梯度吗？ | 独立507题控制：冷14.681，同域／跨域完整继承18.113／18.468；A-only为14.340／14.464；函数中性补偿使首步梯度开放后15.428／15.533。[完整控制](summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v2/function_vs_factor/RESEARCH_FINDINGS.md) | 开放首步梯度没有复制完整继承收益；功能继承与参数化作用不能只用单一首步故事解释 |
| OSFT必须训练两侧因子吗？ | ShortFixed完成同1213 FIT／87更新，DEV宏42.301329，对原OSFT40.783801；训练梯度和两Adam状态少21.34%。[证据](summary/edge_deployment_20261019/osft_short_factor_control_20261022/v2/FINDINGS.md) | 本配方中冻结短侧仍有竞争力；**前向权重未删、长侧仍学方向**，不证明短侧普遍无用或新轻量方法成立 |
| 多位置、大坐标量是必要原因吗？ | 同4399104坐标宽q/v **43.791987**，七位置all-r8 **41.952549**，差1.839438pp、探索区间跨0。[证据](summary/edge_deployment_20261019/position_coordinate_control_20261022/v1/FINDINGS.md) | 原位置扩展收益混有容量影响；少数金融题贡献集中，尚未证明额外位置必要或无用 |
| 逐步保护包一定有净益吗？ | 仅关闭原OSFT梯度／参数两投影，DEV **42.333991**对40.783801；金融25→15，区间跨0。[证据](summary/edge_deployment_20261019/osft_protection_control_20261022/v1/FINDINGS.md) | 净作用未解，不能按平均分宣布保护无用；两组都保留阶段重分解，尚未检验刷新功能 |
| OSFT只是更会利用图像吗？ | 同环境61题、四状态488次原图／错图NLL，配对支持改善q/v .455118248、OSFT .441384426，相对差 **−.013733822**。[证据](summary/final_weight_reduction_20261024/paired_image_support_diagnostic_v1/RESULT_FINDINGS.md) | 未支持“OSFT整体图文配对支持增加更多”的预定解释；不能反推不需要视觉、识别能力无用或多学了某种知识 |

OSFT医学域刚习得相对Seq高15.201186pp，最终仅高3.136410pp；自然域最终反而低1.725122pp。不能把任务平均优势说成五域都由同一机制稳定受益。PASs与OSFT正式AP仅差0.049476pp，也没有分离路由、专家、保护与数值例外的各自因果。

### 非自然领域回答差：定位到了什么

[完整回答行为分析](summary/final_weight_reduction_20261024/response_failure_profile_v1/FINDINGS.md)消费三条正式运行的初态、刚习得与末态记录，没有新生成或重评分：

- **金融：** PASs最终2233题有1669题输出同一个单字母选项，其中1185题为零分；OSFT众数1079题，其中819题零分。大量零分在刚学完时已经存在，不能全部归因遗忘，也不能仅用“没按选项格式回答”解释。
- **遥感：** PASs计数任务2947题中1298题集中为同一数字，其中1213题零分。Seq比较任务4002题中3999题回答相同，任务F1仍约66.72；平均分会掩盖固定回答。
- **医学：** 最终2094题，Seq／OSFT／PASs分别1022／986／1001题零分，预定字面异常仅覆盖91／58／41题，主要失分仍未解释。
- **交通：** 2123题全部部分分，是词面重合分布，不能写成全部正确；原记录不足以由触及长度上限断言被截断。

已查金融89 TRAIN＋2233 TEST的所查字段、选项与当前原始投影一致，未发现本地复制错位或检索丢选项；4条非空选项不能严格解析，仍未知，且不认证全部历史逐token输入或作者金标语义。[金融输入合同](summary/final_weight_reduction_20261024/finance_input_contract_v1/FINDINGS.md)

**没有找到统一的致命输出bug。** 输入利用、答案先验、知识调用、有限步优化和参数表示的作用仍未隔离；这些行为现象不是参数可删的证据。

## FIT→DEV探索结果

以下均为同一100已探索DEV的五域宏，不能与正式21454分数混排。各实验完整域分、逐题正负和费用在链接中。

| 条件 | DEV五域宏 | 实际意义与终态 |
| --- | ---: | --- |
| q/v LoRA-r8 | 39.870439 | 原先导参照；固定条件完成 |
| 原OSFT | 40.783801 | 原FIT-only参照；机制尚未解释 |
| LoRA_all-r8 | 41.952549 | 小测曾入选；完整确认后来失败，见下 |
| LoRA_extra | 40.365016 | 四组筛选完成，未入选 |
| 同坐标宽q/v | 43.791987 | 同4.399M坐标对all高1.839438pp，区间跨0；固定对照关闭 |
| OSFT去逐步保护 | 42.333991 | 平均点高、金融下降，净作用未解；固定对照关闭 |
| ShortFixed | 42.301329 | 对OSFT高1.517529pp，9高7低84平，区间[−.126262,3.962820]；主要由自然和医学各1题贡献；仅必要性消融 |
| NormOnly | 23.242734 | 权重确更新但低于两锚；无同DEV零样本base，不能证明已有参数普遍不足；关闭 |
| fresh PASs FIT参照 | **缺测** | v1重启中断；v2冻结4小时内仅77／87更新，无DEV，不是质量阴性；禁止续跑 |

四组原筛选见[结果](summary/edge_deployment_20261019/osft_increment_screen/v1/RESULTS_20261022.md)。ShortFixed对q/v高2.430890pp，但医学低5.5pp，探索区间也跨0；它没有通过fresh PASs完整比较，所以未放行完整确认。不能把已有最强正式PASs当合法FIT-only teacher替代。

**LoRA_all完整确认：** 原1313训练／19阶段／93次更新→21454题，五域宏**44.894423**、AP19 **44.794146**，低于正式OSFT和PASs，五域均低OSFT。已关闭这套配方，不据TEST救rank／LR／层／路由。[完整负结果](summary/edge_deployment_20261019/osft_increment_confirmation_20261022/v1/FINDINGS.md)

## 真正减少完整权重的探索：哪些失败了

这些是实际完成的负结果，不能因为权重变小或局部误差下降就写成轻量高性能方法。表中0分是实际测得，缺测另标；F1均为对应100已探索DEV宏。

| 固定探索 | 完整最终权重系数 | 起点／末态F1 | 结论与证据 |
| --- | ---: | --- | --- |
| 强wide-q/v父，37FIT真实单删定位后联合删8个FFN | 789,080,096（−11.704%） | 已学父43.791987 → 删图19.620168 | 0训练／反向，真实保存重载；五域F1均降，GPU峰少193.0–200.2MiB，保质失败。[证据](summary/ffn_direct_ablation_20261025/v1/FINDINGS.md) |
| 语言三矩阵r200，Spectral／Answer两完整流 | 662,833,184（−25.8307%） | 末态 .910394／.880147 | 两流各1213 FIT／87更新；19次回答候选均未采用。真实减重但质量崩坏，关闭。[证据](summary/final_weight_reduction_20261023/answer_guided_pilot_v1/FINDINGS.md) |
| KeepGate：保gate/down、只压up | 816,728,096（−8.6102%） | 初 .539926 → 末5.196071 | 同q/v连续适应未恢复；损伤在更新前已出现。[证据](summary/final_weight_reduction_20261024/ffn_role_retention_v1/README.md) |
| KeepUp：保up/down、只压gate | 816,728,096 | 初0 → 末8.905754 | 同数据／87更新，仍不保质；不能在低质量地板挑通用赢家。[同组证据](summary/final_weight_reduction_20261024/ffn_role_retention_v1/OUTCOME.json) |
| up局部功能拟合：Static／gate-response Reuse | 816,728,096／816,706,592 | .255299／.071795 | 各1536个block更新，held局部误差均约降至.25，却没保住完整回答。[证据](summary/final_weight_reduction_20261024/gate_response_reuse_v1/OUTCOME.json) |
| 保gate/up、只压down：WeightD／ResponseD | 两者816,728,096 | .355919／0 | 48个合法S1教师请求、0适应更新，完整保存重载后各100题；两者均失败。[证据](summary/final_weight_reduction_20261024/functional_readout_carrier_v1/README.md) |

最后三组使用的同管线**未适应父模型**宏为**16.116268**，不是强训练基线。父分只按完整digest和逐ID输入绑定的既有证明复制。真实弱父仍明显高于这些替代图；不能把它们与原正式基线的差值当纯压缩因果。

较早半宽方案在5道teacher已见TRAIN探针上，完整／视觉单支／语言单支／联合均分47.089／33.333／22.051／21.754，全部未过原逐题门，**不是独立泛化测试**。[证据](summary/edge_deployment_20261019/width_component_split_20261021/v1/FINDINGS.md)

共享稀疏载体因捕获超过8GiB预算而失败，固定稀疏连接初始化出现非有限值，Smol原生起点在首题processor接口失败；这些都没有可用质量结论。故障与未执行提议详见补充说明。**目前没有一组基座前向权重已被证明可物理删除、同时保留目标质量。** 这些结果只关闭对应固定点，不否定所有低秩、共享或压缩。

## 参数量、最终权重与训练开销

### 最终checkpoint的原保存态

下面来自已保存完整checkpoint的元数据提取，包含视觉、语言、连接器、专家、路由、冻结基和必需持久状态，按底层共享storage去重。不是只算adapter，也不是文件容器总大小。

| 骨干／方法 | 完整保存系数数量 | 最终权重占用 MiB | 相对原基座增加 MiB |
| --- | ---: | ---: | ---: |
| OneVision／SeqFT-LoRA | 894,216,224 | 1706.613 | 2.062 |
| OneVision／FOREVER | 894,216,224 | 1706.613 | 2.062 |
| OneVision／CL-MoE | 930,877,024 | 1846.463 | 141.912 |
| OneVision／PASs-MoE | 1,034,447,216 | 2241.552 | 537.001 |
| OneVision／PaRSP | 893,675,552 | 2273.345 | 568.794 |
| OneVision／OSFT | 990,913,568 | 2757.984 | 1053.434 |
| OneVision／SpaRTA | 904,077,344 | 1744.231 | 39.680 |
| Qwen／SeqFT-LoRA | 2,210,075,136 | 4217.462 | 4.156 |

原OneVision基座**893,675,552**系数：语言494,090,112、视觉397,747,744、连接器1,836,800、image_newline896；原Qwen基座2,208,985,600。保存dtype和表示不同，因此**字节增加不等于新增参数**，例如PaRSP系数不变但部分FP32保存。普通LoRA／OSFT／ShortFixed公平常规fold后仍是原稠密形状893,675,552，消除表示冗余不算结构压缩；这里没有声称已运行数值折叠。[全部dtype、排除项和来源](summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/FINDINGS.md)

### 训练参数是辅助项

| 表示 | 未折叠完整权重系数 | 可训练坐标 |
| --- | ---: | ---: |
| 原基座 | 893,675,552 | 本项目起点冻结，不作为训练方法 |
| q/v LoRA-r8 | 894,216,224 | 540,672 |
| 七位置LoRA_all-r8 | 898,074,656 | 4,399,104 |
| 原OSFT | 990,913,568 | 113,766,144 |
| ShortFixed | 990,913,568 | 89,485,056 |

ShortFixed实测梯度357,940,224字节、两Adam moment715,880,448字节；较原OSFT源码计数均少21.342982%，但全部USV仍在，不能称最终模型轻量。其19阶段末释放训练局部引用后allocated约2.958–2.972GB，首末增加14.61MB；RSS后期仍上升。这些不是训练／推理峰或增长因果证明。[已有一次内存增长分析](summary/edge_deployment_20261019/shortfixed_memory_growth_20261022/v1/FINDINGS.md)

阶段图未来需分别画完整权重P_total(t)和可训练参数P_train(t)。固定结构画水平线，不编阶段增长曲线；模型权重存储、训练状态、运行峰值内存分别报告，没测的仍不填。

## 其它已完成的局部发现

这些结果保留科研价值及适用边界，不转写成已完成主方法。历史诊断使用不同题集和输入协议；其中已见TEST的分析仅作解释，不用来筛新配置。

| 已有发现 | 数据与限制 | 证据 |
| --- | --- | --- |
| 适应与遗忘不能只看末态 | 同医学任务学完20.36，后续同域34.17、跨域18.63；增加更新医学29.26→51.96、遥感76→44，不能统一说“多训就好” | [研究报告](summary/onevision_diagnostics_20261016/RESEARCH_REPORT.md) |
| 完整初始化继承在连续诊断有局部收益 | 温启动19阶段／155题AP31.025→35.025、医学15.841→25.339；另外四域未变，仍低于同协议Seq43.363 | [结果](summary/onevision_diagnostics_20261016/warm_start_20261018/first19_outcome/RESEARCH_FINDINGS.md) |
| 输入相似不保证适配器答得好 | 同155题固定相似度22.39、学习相似度31.03、TRAIN回答质量选择36.70，均低同协议Seq43.36 | [路由诊断](summary/onevision_diagnostics_20261016/functional_router/PILOT_FINDINGS_20261018.md) |
| 缓存调用点收益未稳定确证 | 同155题顺序43.363、缓存几何45.700、缓存质量42.860，域宏差区间含0；部分题NLL下降而F1下降 | [四格结果](summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/first_four_view_outcome/RESEARCH_FINDINGS.md) |
| 新TRAIN质量反馈没取得收益 | 新匹配／可学习距离均42.860；20锚只覆盖11／19任务，逐题最优相对最好固定状态空间仅.144pp | [五格结果](summary/onevision_diagnostics_20261016/answer_compatibility_20261018/v1/all_five_outcome/RESEARCH_FINDINGS.md) |
| 相同前向功能不保证相同更新几何 | 保BA不变、A缩放4倍，三任务真实梯度下AdamW权重更新约4倍；作者RITE更新相同，没有对应最终F1提升结论 | [几何干预](summary/onevision_diagnostics_20261016/optimizer_geometry_20261019/v1/RESEARCH_FINDINGS.md) |
| 知识记住不等于答题会用 | 16条专业事实可记住，但相关医学问答没有稳定改善 | [原报告](summary/onevision_diagnostics_20261016/RESEARCH_REPORT.md) |
| 少量更新时部分保护项可能来不及起效 | PASs代码诊断中9个单步阶段一保护系数一直0；以当前参数为中心的EWC首步保护梯度0。未证明最终分数因果 | [保护预算诊断](summary/onevision_diagnostics_20261016/regularizer_budget_20261018/RESEARCH_FINDINGS.md) |
| PAS保护机会不均 | 正式93步条件重建40步结构零、53步仅有正系数资格上界；不是实际有效保护见证，旧上游源码SHA有缺口，不是成绩因果 | [覆盖证据](summary/final_weight_reduction_20261023/pas_rs_coverage_v1/FINDINGS.md) |
| 选择变化不等于选择器学了 | SMoLoRA代码检查两选择器无梯度、专家正常更新；固定选择器仍能随输入切换，没有质量因果结论 | [梯度检查](summary/onevision_diagnostics_20261016/smolora_gate_20261018/RESEARCH_FINDINGS.md) |
| 名义更新范围不等于真实权重变化 | PaRSP部分o/down在名义mask并集外变化，占所测ΔW能量2.9344%；SpaRTA93步未到200步恢复周期 | [原证据](summary/edge_deployment_20261019/sealed_update_constraints_20261021/FINDINGS.md) |
| 自然重排Kron不值得当前点继续投入 | 首层三个FFN矩阵、近同系数预算，K69自然布局权重残差0／3胜普通r200；仅离线优先级阴性，非F1结论 | [原证据](summary/final_weight_reduction_20261023/kron_layout_priority_v1/FINDINGS.md) |

文献／方法分类整理了29篇2024–2026正式会议工作，以及五数据集、19任务、来源与复用依据。[分类、数据表和历史基座核查](SUPPLEMENTARY_NOTES.md#文献分类与数据组成)保留完整索引。LoRA-FA、SVFT／SVFit、RandLoRA、HOFT、共享／低秩／动态压缩等已有技术可作工具；它们本身不是我们的实测贡献，也不因此禁止有意义探索。

## 最新完整对照与缺测终态

真实单删定位借鉴功能消融，只在37条合法FIT上计算925次NLL，再固定8层并联合物理删除；最终父／删图各100DEV。父宏与已有wide-q/v点一致；各域NLL均升高，五域F1如下，不把减重本身当成功。

| 领域 | 完整父F1 | 删除F1 | 删除−父pp |
| --- | ---: | ---: | ---: |
| 自然 | 61.500 | 27.500 | −34.000 |
| 医学 | 44.500 | 25.000 | −19.500 |
| 遥感 | 45.000 | 25.000 | −20.000 |
| 金融 | 30.000 | 0.000 | −30.000 |
| 交通 | 37.960 | 20.601 | −17.359 |

完整权重893675552→789080096系数；五个固定输入、各恰32token的GPU peak allocated下降193.007–200.162MiB。显存与F1必须联合判断，本构造未获得保质证据。单层相对排名不是绝对可删性；单删损伤已经非零，不能把全部损失归于层间交互。所有原始产物和完整包已转存NEW private；细节及一次过程隔离修复见[完整结果](summary/ffn_direct_ablation_20261025/v1/FINDINGS.md)。

原OSFT阶段刷新对照：prefix与Fixed完整退出，Refresh因Errno122账户磁盘配额超限失败；原controller／worker已无，三个故障收据.tmp为空，旧queue仍running已失真。没有VERIFIED，不消费部分质量，也不判刷新有效或无效。[终态证明](summary/ffn_static_delete_20261025/v1/STAGE_REFRESH_TERMINAL_OBSERVED.json)。另一个门梯度定位点在联合完整性断言失败，没有删除质量；它与本次已完整测得的功能阴性分开。[实施历史](summary/ffn_static_delete_20261025/v1/README.md)

## 新会话接续

1. 先读本README、[补充说明的交接段](SUPPLEMENTARY_NOTES.md#新会话运行交接)和[AGENTS顶部](AGENTS.md)，以最新人类目标为准。
2. stage-refresh及当前直接8-FFN构造均已终止，不继续等旧PID。后续必须检验新的功能承接问题，不能把相对低损伤排名说成参数无用，也不按已见DEV缩小删层幅度救分。
3. 人类目标仍为各域F1对最强基线略升、完整权重与GPU/内存显著有利。已有技术可借用；必须由真实功能与适应对照支持，不能仅靠工程准入或权重排名。
4. 正式基线保持原运行；只有新sealed proof才更新正式表。禁止重复旧评分、配对、参数／谱／梯度审计或泛读填时间。

项目权威仓库在NEW `/data1/liyapeng/program/LLMCL` 的Git main；本机为非Git镜像。模型、数据正文、图像、特征、checkpoint、密钥和作者源码均不入Git。详细运行身份、SHA、历史费用、故障和关闭分支见补充说明与原证据，不能从本文的历史快照推断永久存活。
