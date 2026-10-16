# 多领域公开数据集：实际使用论文与选择依据

核验：2026-10-16。用户最新确认 **五领域、五个图文问答任务**，每领域只用一个来源；不再单独寻找纯文本数据集。TextVQA、SLAKE、RSVQA-LR、MME-Finance、AQUA 共 172,407 条全量训练、1,728 条1%训练、22,246条共同完整测试；矩阵6×5。此前十任务方案只作历史比较。本文记录论文**实际实验使用**，不把参考文献中提到某数据集当作使用证据。公开可获取不等于可无限制再发布，下载时仍保留原许可。

## 先区分论文在做什么

| 论文与发表 | 实际使用的数据／范围 | 正文依据 | 与本实验的关系 |
|---|---|---|---|
| [CL-Cross VQA: A Continual Learning Benchmark for Cross-Domain Visual Question Answering](https://openaccess.thecvf.com/content/WACV2025/html/Zhang_CL-Cross_VQA_A_Continual_Learning_Benchmark_for_Cross-Domain_Visual_Question_WACV_2025_paper.html)，Yao Zhang 等，**WACV 2025** | VQA Abstract、Toronto COCO-QA、AQUA 的视觉子集、PathVQA、VQA-Med-2019，分别对应抽象场景、通用图像、艺术、病理、放射 | 正式会议页；作者[公开正文 §3.1–3.2](https://arxiv.org/html/2211.10567) | 真正讨论领域迁移；但它使用固定答案池分类，且病理／放射均属医学。可引用跨域动机与数据来源，不能冒称我们复现其五领域或生成式协议。公开 arXiv 最初发布于 2022，正式会议年份为 2025；旧稿的实现细节不能自动代表最终版。 |
| [CoIN: A Benchmark of Continual Instruction Tuning for Multimodal Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a45500d9eda640deed90d8a62742be5-Paper-Datasets_and_Benchmarks_Track.pdf)，Cheng Chen 等，**NeurIPS 2024 Datasets and Benchmarks** | ScienceQA、TextVQA、GQA、VizWiz、VQAv2、OCR-VQA，以及 ImageNet、RefCOCO／RefCOCO+／RefCOCOg | §3、Table 1、Appendix A.2；[作者代码](https://github.com/zackschen/CoIN) | 以多任务／指令类型为主，包含分类和定位，不是五个行业领域。TextVQA 明确报告 34,602 train／5,000 test，可直接支持保留这一小规模来源。论文主表与附录部分计数口径存在差异，其他来源须核对发布文件。 |
| [SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)，Ziqi Wang 等，**ICCV 2025** | 连续训练：ScienceQA、TextVQA、Flickr30k、ImageNet、GQA、VQAv2；训练结束后的迁移评测：VizWiz、TextCaps、OCRVQA；Places365 做 5／10-shot | [作者正文 §4.1、Appendix D、Table 8](https://arxiv.org/html/2411.13949) | 生成式 MLLM 持续指令学习依据充分。必须把训练数据与下游零样本数据分开。TextVQA 34,602／5,000；ScienceQA 12,726／4,241。GQA 表报 72,140 训练样本属于作者 benchmark 口径，不等于官方原始 943,000 QA。 |
| [CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html)，**CVPR 2025** | 同一个 VQAv2，按 recognition、location、judge 等划为十个题型任务 | [补充材料 §1、Table 1](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Huai_CL-MoE_Enhancing_Multimodal_CVPR_2025_supplemental.pdf) | 是生成式持续 VQA 方法参考；十个任务不能解释为十个独立领域，也不能据此声称它使用了医学、金融、艺术。 |
| [MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI](https://openaccess.thecvf.com/content/CVPR2024/html/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.html)，Xiang Yue 等，**CVPR 2024** | 11.5K 题，覆盖艺术设计、商业、科学、医学、人文社科、技术工程，30 学科；同时有选择和开放题 | 原论文及[作者项目](https://mmmu-benchmark.github.io/)、[作者代码](https://github.com/MMMU-Benchmark/MMMU) | 是真正多学科公开评测集，适合作为跨域外部评测候选。它不是天然提供几万条监督训练题的五域十任务 CL 数据流。作者于 **2026-02-12 公布 test 答案**，不能沿用“测试答案始终隐藏”的旧说法；仍不应把 test 擅自改作训练池。 |
| [Adapting Large Language Models via Reading Comprehension](https://proceedings.iclr.cc/paper_files/paper/2024/file/d51cd79a85833b022841f7a2383b32d3-Paper-Conference.pdf)，Daixuan Cheng、Shaohan Huang、Furu Wei，**ICLR 2024** | 医学：PubMedQA、ChemProt、MQP、RCT、USMLE；金融：ConvFinQA、FPB、FiQA SA、Headline、NER；法律：SCOTUS、CaseHOLD、UNFAIR-ToS | §4、Table 4、Appendix D Tables 10–11；[作者资源](https://huggingface.co/AdaptLLM) | 三个明确行业领域，但主要研究文本领域预训练与下游评测，不是多模态顺序 CL。多数任务是分类／实体抽取；其中 ConvFinQA 才是金融会话数值 QA，不能把整套都称开放生成问答。 |

## 规模比较必须使用同一单位

| 数据集／发布配置 | 已核对的规模或论文所报口径 | 谁实际使用 | 当前判断 |
|---|---|---|---|
| **TextVQA 原始 train／validation** | **34,602／5,000 QA**；本地 Parquet 元数据一致 | CoIN（NeurIPS 2024）、SMoLoRA（ICCV 2025） | 自然／通用图像任务优先保留；去掉 VQAv2、GQA 后不再叠加百万 QA。 |
| **ScienceQA 作者 CVIT 配置** | **12,726／4,241**；这是上述 benchmark 报告数，不代表全原始 ScienceQA 全部模态 | CoIN、SMoLoRA | 几万规模候选，含选择题；属于科学／教育，不作为自然域之外另一个独立新域。本轮不新增任务。 |
| **VizWiz 在 CoIN 中的配置** | 附录报告 **20,523 train／8,000 test**；还需对作者发布清单核对 | CoIN；SMoLoRA 仅作为迁移评测 | 小规模真实场景候选；无障碍场景包含无法回答的照片，须保留作者答案语义。不能把 SMoLoRA 的零样本使用写成监督训练。 |
| **GQA 在 SMoLoRA／CoIN 中的配置** | 论文报告 **72,140 train**；本地官方 balanced 原始池则是 **943,000 QA** | CoIN、SMoLoRA | 可研究作者正式发布的小配置，但要先确认计数单位、QA 对和筛选机制。本轮先去掉原始大池，不把“72k”套到旧 943k 文件上。 |
| **AQUA 视觉子集** | 作者 train **29,568**／val **1,507**／test **1,270**，本地 JSON 与原标记一致 | CL-CrossVQA；[ArtQuest: Countering Hidden Language Biases in ArtVQA（WACV 2024）](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf) | 艺术图像来源有直接跨域 CL 依据。AQUA 知识子集另用于 T9，属于我们的文本视图，不能冒称 CL-CrossVQA 采用了这一纯文本任务。 |
| **SLAKE／VQA-RAD 当前发布配置** | SLAKE **9,835 train／2,094 test**（中英双语）；VQA-RAD 镜像 **1,793／451** | [Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.pdf)，Table 1 | 可保留小规模医学图像池。此文是医学 VQA／可靠性实验，并非五域 CL；原始发表为 ISBI 2021／Scientific Data 2018，不能因复用论文是 ICCV 就称数据集首发 ICCV。该文通常采用英语 SLAKE 配置，与本轮保留双语的数量不同。 |
| **RSVQA-LR 原始发布** | **77,232 QA／772 图像**；作者 split 中 active 掩码明确指定 **57,223 train／10,005 val／10,004 test**，图像为 **572／100／100**；已从 Zenodo 原文件计数 | [GeoChat（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/papers/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.pdf)、[LHRS-Bot（ECCV 2024）](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09511.pdf)、[RSAdapter（TGRS 2024）](https://github.com/AI4RS/RSAdapter) | 遥感的小规模优先候选；首次发表 TGRS 2020，[作者页](https://rsvqa.sylvainlobry.com/)、[原始发布](https://zenodo.org/records/6344334)。部分复用工作量化计数、只测若干题型；本轮保留原数字和全部题型，不冒称完全复现其评测。 |
| **FinQA／ConvFinQA** | FinQA 本地原始 train **6,251**／test **1,147**；ConvFinQA 属独立的多轮扩展，数量尚未做文件级核对 | FinQA [原始论文（EMNLP 2021）](https://aclanthology.org/2021.emnlp-main.300/)；ConvFinQA 被 AdaptLLM 用于金融 QA 评测 | 当前保留已经核对且较小的 FinQA。不能把 ConvFinQA 的复用记录记在 FinQA 名下，也不能把表格文字当金融视觉图像。 |

### 合并为五任务前的收缩记录（历史方案）

优先将 T2 改为 TextVQA，T4 保留 VQA-RAD＋SLAKE，T6 改为 RSVQA-LR；其他小规模来源保留。当前单个训练任务最大为 SQuAD 的 87,599 条，均不超过 100,000；十个任务训练合计 **281,958**、1% 组 **2,825**、共同完整测试 **37,998**。这是**每个来源／任务几万或更少**，十任务相加仍约 28 万，不能说整个实验总共只有几万。原大池合计 2,322,005 条，因此训练量下降约 87.9%。

该方案同时满足完整原始配置、保留原题型、缩小单任务规模；它不是照搬某篇论文现成的五域文本／图像配对方案。已核验的论文中，没有证据能把我们的完整五域十任务构造归给任一篇。医学原始来源并非 CVPR／ICCV；此前要求的“新增顶会医学来源”不能仅凭后续复用论文满足，应在文章中如实报告原始发表与复用场所。

## 金融源的复用边界

MME-Finance 有一项已核实的正式复用：[OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f97eefb5e158cc6f64c7883cf4afc6d3-Abstract-Conference.html)。其 §3.1 从既有来源取图并另标自身问答，属于**图像复用**，不能写成原 MME-Finance 全套 QA 评测。[BICR 作者库](https://github.com/Ledengary/BICR) 则明确包含 mme_finance 重建和评测代码，但目前仅核实 2026 预印本。两条分别记载，不用“正式取图复用＋预印本 QA 评测”冒充多篇正式原 QA 复用。

## 补充线索：不混作已发表多域 CL 依据

- [TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](https://arxiv.org/html/2310.06762)，作者正文 §4.1：ScienceQA、FOMC、MeetingBank、C-STANCE、20Minuten、Py150、NumGLUE-cm／ds，统一采样每任务 5,000 train／2,000 test。它说明文献会控制任务预算；但本文未核实其正式会议版本，记为作者预印本。也不是五域多模态开放 QA，不据此改变已授权的全量／1% 设置。
- [Continual-NExT 作者工具库](https://github.com/ECNU-SII/Continual-NExT) 列出 ArXivQA、GeoChat、IconQA、ClevrMath、CodeQA、ImageNet、Flickr30k、DocVQA、TextVQA、MathQA、ChartQA、PathVQA、Grounding、ScienceQA、WikiQA。仓库已核实；其对应论文与正式发表版本仍需进一步对应，不计作已核验顶会实验。仓库表中“GeoVQA 31K”和顺序中的“GeoChat”存在命名口径问题，不把它直接当我们的 GeoChat 全原始 QA 数。

## 记录规范

论文写作分别引用：**提出数据集的论文**与**复用数据集做实验的论文**。表格保留实验位置、模态、训练或仅评测角色、版本、原始数量与作者子集数量；原始 QA、图像、会话数不能混用。候选集合变更、原始文件校验、固定 ID 和测试哈希另存当前实验目录，不回写旧实验。
