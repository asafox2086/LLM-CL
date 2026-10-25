# 第二主表：10%训练／完整测试

人类于2026-10-25明确确认训练10%＋完整测试。数据与独立runner已freeze：TRAIN13105 QA、完整TEST21454 QA、19任务、单轮828次主任务更新，seed42；manifest SHA为`02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7`。OLD四卡两骨干SeqFT-LoRA与PASs-MoE均已通过preflight及reload（return0），正式运行处于初始评价（S0）；其余8个native组合OLD排队，两FOREVER待10%动态回放适配核验、两SMoLoRA待NEW原生BF16空卡。0个10%组合完整核验，不填1%数值或资源估计。见[正式协议](PROTOCOL.md)。

Answer F1取原token_f1，0–100。五域列是S19最终域内QA平均；五域宏给五域等权，AP19给19个原生任务等权。本表覆盖两个backbone各八类基线，共16组合，与1%原表的18列逐列对齐，另保留完整参数量、CP大小、新习得补充和运行状态。新习得／金融错答不替代完整主表，BWT仅继承描述，不作为优化目标。空白表示没有完整核验结果，不是0分。

| 方法／Backbone | 类型 | 发表会议／年份 | TextVQA<br>自然／通用 | SLAKE<br>医学 | RSVQA-LR<br>遥感 | FinMME<br>金融 | CODA-LM<br>交通 | 五域宏 F1 ↑ | AP19 ↑ | FWT ↑ | BWT ↑ | 最终权重存储<br>保存态 MiB ↓ | 相对原基座增加<br>MiB ↓ | 固定五窗GPU峰值†<br>MiB ↓ | 首词延迟<br>s ↓ | 生成速度<br>tok/s ↑ | 常驻适配器<br>数量 ↓ | 完整参数量<br>系数 | 最终CP<br>MiB | 新习得五域宏F1 ↑ | 新习得AP19 ↑ | 状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Qwen2-VL-2B-Instruct** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 朴素顺序微调 | 标准对照（无独立发表年份） |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 正式运行：初始评价 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 待10%动态回放适配核验 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 已freeze，待NEW原生BF16空卡 |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 正式运行：初始评价 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 空，不自动运行探索配方 |
| **LLaVA-OneVision-Qwen2-0.5B-OV** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 朴素顺序微调 | 标准对照（无独立发表年份） |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 正式运行：初始评价 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 待10%动态回放适配核验 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 已freeze，待NEW原生BF16空卡 |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 正式运行：初始评价 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 空，不自动运行探索配方 |

新习得AP19取各task刚训练结束的`R[t,t]`等权均值；新习得五域宏先按各task TEST QA数在域内加权，再五域等权。它们来自不同task时点的模型，不是S19或单一域末模型。74格覆盖仍为stage0全19、stage1–18当前＋下一、stage19全19，无需为这两列额外请求。

完整参数量须实际统计全部冻结／可学参数、视觉／语言／连接器／专家／router，另列dtype；“完整权重存储”沿1%原保存态／原dtype，包含必需持久状态并按底层storage去重，掩码等非参数状态另附明细。相对基座增量是同骨干权重bytes差，不等于新增参数量。最终CP是文件容器大小，与去重权重不同；训练坐标及训练峰／CPU RSS在资源明细报告。常驻适配器统计完整驻留数量，不能以单次激活数代替。

固定五窗GPU峰只有独立真实重载／同原始QA／每题32新token／实际卡型UUID及顺序绑定后才填；不同骨干保留原生processor展开。没有测量就标未测，不能从1%峰、checkpoint文件或训练峰估算。首词延迟、生成速度目前没有本10%测量，不作数值声明；列保留并待逐组合实测。

每个组合均保留S19完整checkpoint及必要方法状态、processor／config和来源绑定，按真实文件／SHA归档；参数量标量、适配器文件或latest指针不能替代最终模型。[最终模型留存与16组索引](FINAL_MODEL_RETENTION.md)。

所有结果须绑定本10% manifest、源码／环境、完整19阶段／74格、原评分与完整S19 checkpoint，才进入本表。原1%表、FOREVER历史和既有结果保持原样，不因本表新增而覆写。全量100%不属于此次授权。
