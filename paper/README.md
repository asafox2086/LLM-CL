# 2024–2026 顶会持续学习论文索引

检索及下载核验日期：2026-10-11。收录 28 篇已在正式会议主会发表的论文：持续预训练 6 篇、持续微调 20 篇、持续对齐 2 篇；其中 CVPR 21 篇、ICCV 1 篇、ICML 1 篇、ICLR 4 篇、NeurIPS 1 篇。仅收录经 CVF / 正式会议 proceedings 页面核实的主会论文，不将 workshop、Findings 或未录用预印本当成主会论文。年份按会议发表年份。

## 目录与分类口径

```text
paper/
├── README.md
├── continual_learning_manifest.json
├── collect_cl_papers.py
├── pre_training/                  # 持续预训练：6
├── pre_finetuning/                # 持续微调：20
│   ├── method_replay/             # 回放：6
│   ├── method_regularization/     # 正则化：8
│   └── method_architecture/       # 架构：6
└── pre_alignment/                # 持续偏好／奖励对齐：2
```

这里的 `/paper` 指仓库内的 `paper/`；服务器实际路径为 `/data2/liyapeng_grp/program/LLMCL/paper`。原有医学论文文件保留，未计入本次 28 篇。

- **持续预训练**：用自监督／语言建模／图文对比目标继续训练。区分严格序贯多域预训练与领域、上下文、模型范式扩展型 continued pretraining；后者不能直接视为抗遗忘算法。
- **持续微调**：广义任务监督的序贯适配，包括 LLM、VLM、ViT、扩散模型、压缩、分割和在线 CL。并非全部为 LLM 指令微调；表格逐条标出适用对象。通用在线训练策略作为方法参考单独注明。
- **持续对齐**：偏好／奖励／RLHF 的持续或迭代更新。CPPO 是明确的动态偏好持续学习；SeRA 是论文明确描述的 continual alignment 迭代自举扩展阅读，不等价于标准多任务遗忘评测。普通视觉特征／跨模态表征对齐不据此归入本类。本轮未收录已核验的 2026 主会偏好持续对齐论文，避免以同名表征对齐或审稿稿件补数。
- **微调小类**按主要抗遗忘机制单选归档：回放涵盖真实样本／实例／生成样本；正则化涵盖权重约束、蒸馏、梯度投影和子空间约束；架构涵盖参数隔离、专家／Adapter／Prompt 和动态容量分配。混合方法在机制栏列次要机制。该归档是整理者基于摘要及方法的判断，不是作者统一分类。

## 方法 × 方向二维表

✓ 表示所属大类或归档小类；横轴同时列三大方向及持续微调的三小类。每行提供正式论文链接和本地 PDF。

| 方法 / 论文超链接 | 会议·年份 | 持续预训练 | 持续微调 | 微调·回放 | 微调·正则化 | 微调·架构 | 持续对齐 | 分类依据 / 适用对象 | PDF |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| **Knowledge replay** — [Towards Backward-Compatible Continual Learning of Image Compression](https://openaccess.thecvf.com/content/CVPR2024/html/Duan_Towards_Backward-Compatible_Continual_Learning_of_Image_Compression_CVPR_2024_paper.html) | CVPR 2024 | — | ✓ | ✓ | — | — | — | 知识回放；兼有模型架构设计；预训练图像压缩模型的序贯微调 | [PDF](pre_finetuning/method_replay/2024_CVPR_Knowledge_replay.pdf) |
| **EIR** — [Beyond Background Shift: Rethinking Instance Replay in Continual Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_Beyond_Background_Shift_Rethinking_Instance_Replay_in_Continual_Semantic_Segmentation_CVPR_2025_paper.html) | CVPR 2025 | — | ✓ | ✓ | — | — | — | 实例存储与粘贴回放，缓解背景漂移；持续语义分割 | [PDF](pre_finetuning/method_replay/2025_CVPR_EIR.pdf) |
| **View-batch model** — [Do Your Best and Get Enough Rest for Continual Learning](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_Do_Your_Best_and_Get_Enough_Rest_for_Continual_Learning_CVPR_2025_paper.html) | CVPR 2025 | — | ✓ | ✓ | — | — | — | 调节样本再训练间隔；兼有自监督学习；通用持续学习训练策略（非专门 LLM 微调） | [PDF](pre_finetuning/method_replay/2025_CVPR_View_batch_model.pdf) |
| **GIFT** — [Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Synthetic_Data_is_an_Elegant_GIFT_for_Continual_Vision-Language_Models_CVPR_2025_paper.html) | CVPR 2025 | — | ✓ | ✓ | — | — | — | 扩散生成合成样本；兼有蒸馏和权重正则化；VLM 持续微调 | [PDF](pre_finetuning/method_replay/2025_CVPR_GIFT.pdf) |
| **Smart Replay** — [Smart Replay: Adaptive Scheduling of Memory Rehearsal for Computational Resource-Aware Incremental Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Smart_Replay_Adaptive_Scheduling_of_Memory_Rehearsal_for_Computational_Resource-Aware_CVPR_2026_paper.html) | CVPR 2026 | — | ✓ | ✓ | — | — | — | 在计算预算下自适应调度回放比例；资源受限增量学习（通用训练策略） | [PDF](pre_finetuning/method_replay/2026_CVPR_Smart_Replay.pdf) |
| **UCo-VQA baseline** — [Re-evaluating Continual VQA: Toward Fair and Robust Evaluation for Multimodal Continual Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Re-evaluating_Continual_VQA_Toward_Fair_and_Robust_Evaluation_for_Multimodal_CVPR_2026_paper.html) | CVPR 2026 | — | ✓ | ✓ | — | — | — | 仅问题回放；兼有双层蒸馏；持续 VQA；论文兼含评测基准 | [PDF](pre_finetuning/method_replay/2026_CVPR_UCo_VQA_baseline.pdf) |
| **InfLoRA** — [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html) | CVPR 2024 | — | ✓ | — | ✓ | — | — | 限制低秩更新子空间，抑制任务干扰；兼有 LoRA 架构；ViT 参数高效持续微调 | [PDF](pre_finetuning/method_regularization/2024_CVPR_InfLoRA.pdf) |
| **One-for-More** — [One-for-More: Continual Diffusion Model for Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Li_One-for-More_Continual_Diffusion_Model_for_Anomaly_Detection_CVPR_2025_paper.html) | CVPR 2025 | — | ✓ | — | ✓ | — | — | 梯度投影约束保护历史知识；扩散模型持续异常检测 | [PDF](pre_finetuning/method_regularization/2025_CVPR_One_for_More.pdf) |
| **ConceptGuard** — [ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_ConceptGuard_Continual_Personalized_Text-to-Image_Generation_with_Forgetting_and_Confusion_Mitigation_CVPR_2025_paper.html) | CVPR 2025 | — | ✓ | — | ✓ | — | — | 记忆保持正则；兼有 shift embedding 和概念绑定提示；扩散模型持续个性化 | [PDF](pre_finetuning/method_regularization/2025_CVPR_ConceptGuard.pdf) |
| **CoMasTRe** — [Continual Segmentation with Disentangled Objectness Learning and Class Recognition](https://openaccess.thecvf.com/content/CVPR2024/html/Gong_Continual_Segmentation_with_Disentangled_Objectness_Learning_and_Class_Recognition_CVPR_2024_paper.html) | CVPR 2024 | — | ✓ | — | ✓ | — | — | objectness 蒸馏、多标签类别蒸馏；兼有两阶段结构；持续语义分割 | [PDF](pre_finetuning/method_regularization/2024_CVPR_CoMasTRe.pdf) |
| **LfI** — [Learning from Itself: Mining Internal Knowledge from Vision Language Models for Continual Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Gong_Learning_from_Itself_Mining_Internal_Knowledge_from_Vision_Language_Models_CVPR_2026_paper.html) | CVPR 2026 | — | ✓ | — | ✓ | — | — | 自适应双向蒸馏；兼有伪字幕生成；CLIP 持续微调 | [PDF](pre_finetuning/method_regularization/2026_CVPR_LfI.pdf) |
| **EWC Done Right** — [Elastic Weight Consolidation Done Right for Continual Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Elastic_Weight_Consolidation_Done_Right_for_Continual_Learning_CVPR_2026_paper.html) | CVPR 2026 | — | ✓ | — | ✓ | — | — | 修正权重重要性估计的权重正则化；通用持续学习正则方法 | [PDF](pre_finetuning/method_regularization/2026_CVPR_EWC_Done_Right.pdf) |
| **CCL-DC** — [Improving Plasticity in Online Continual Learning via Collaborative Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Improving_Plasticity_in_Online_Continual_Learning_via_Collaborative_Learning_CVPR_2024_paper.html) | CVPR 2024 | — | ✓ | — | ✓ | — | — | 蒸馏链和协同学习；可与回放方法结合；在线持续学习（非专门 LLM 微调） | [PDF](pre_finetuning/method_regularization/2024_CVPR_CCL_DC.pdf) |
| **MoE-Adapters / DDAS** — [Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.html) | CVPR 2024 | — | ✓ | — | — | ✓ | — | 动态专家 Adapter 扩展和输入路由；CLIP 持续微调 | [PDF](pre_finetuning/method_architecture/2024_CVPR_MoE_Adapters_DDAS.pdf) |
| **CPrompt** — [Consistent Prompting for Rehearsal-Free Continual Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Consistent_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2024_paper.html) | CVPR 2024 | — | ✓ | — | — | ✓ | — | 任务提示与分类器；兼有一致性学习；冻结预训练模型的持续提示微调 | [PDF](pre_finetuning/method_architecture/2024_CVPR_CPrompt.pdf) |
| **S6MOD** — [Enhancing Online Continual Learning with Plug-and-Play State Space Model and Class-Conditional Mixture of Discretization](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Enhancing_Online_Continual_Learning_with_Plug-and-Play_State_Space_Model_and_CVPR_2025_paper.html) | CVPR 2025 | — | ✓ | — | — | ✓ | — | 附加状态空间分支与类别条件路由；兼有对比损失；在线持续学习结构插件 | [PDF](pre_finetuning/method_architecture/2025_CVPR_S6MOD.pdf) |
| **CHEEM** — [CHEEM: Continual Learning by Reuse, New, Adapt and Skip - A Hierarchical Exploration-Exploitation Approach](https://openaccess.thecvf.com/content/CVPR2026/html/Savadikar_CHEEM_Continual_Learning_by_Reuse_New_Adapt_and_Skip_-_CVPR_2026_paper.html) | CVPR 2026 | — | ✓ | — | — | ✓ | — | 架构搜索选择复用、新建、适配和跳过；预训练 ViT 持续适配 | [PDF](pre_finetuning/method_architecture/2026_CVPR_CHEEM.pdf) |
| **GRAPA** — [Parameter-efficient Continual Learning for Enhancing Plasticity without Forgetting under Limited Model Capacity](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Parameter-efficient_Continual_Learning_for_Enhancing_Plasticity_without_Forgetting_under_Limited_CVPR_2026_paper.html) | CVPR 2026 | — | ✓ | — | — | ✓ | — | 梯度引导参数复用和强化学习参数分配；有限容量下参数高效持续学习 | [PDF](pre_finetuning/method_architecture/2026_CVPR_GRAPA.pdf) |
| **MedQwen / Sparse Spectral LoRA** — [Sparse Spectral LoRA: Routed Experts for Medical VLMs](https://openaccess.thecvf.com/content/CVPR2026/html/Nejatimanzari_Sparse_Spectral_LoRA_Routed_Experts_for_Medical_VLMs_CVPR_2026_paper.html) | CVPR 2026 | — | ✓ | — | — | ✓ | — | 谱分段 LoRA 专家和路由；医学 VLM 序贯微调 | [PDF](pre_finetuning/method_architecture/2026_CVPR_MedQwen_Sparse_Spectral_LoRA.pdf) |
| **Learn more, but bother less** — [Learn more, but bother less: parameter efficient continual learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b0bc711f48724237b38823c4d9cee10b-Abstract-Conference.html) | NeurIPS 2024 | — | ✓ | — | ✓ | — | — | 梯度投影保持低秩子空间正交；兼有敏感度初始化；LLM 持续任务微调 | [PDF](pre_finetuning/method_regularization/2024_NeurIPS_Learn_more_but_bother_less.pdf) |
| **MedCoSS** — [Continual Self-supervised Learning: Towards Universal Multi-modal Medical Data Representation Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Ye_Continual_Self-supervised_Learning_Towards_Universal_Multi-modal_Medical_Data_Representation_Learning_CVPR_2024_paper.html) | CVPR 2024 | ✓ | — | — | — | — | — | 跨模态自监督预训练；回放、特征蒸馏和 mixup；医学多模态序贯预训练 | [PDF](pre_training/2024_CVPR_MedCoSS.pdf) |
| **CoSMIC** — [CoSMIC: Continual Self-supervised Learning for Multi-Domain Medical Imaging via Conditional Mutual Information Maximization](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_CoSMIC_Continual_Self-supervised_Learning_for_Multi-Domain_Medical_Imaging_via_Conditional_ICCV_2025_paper.html) | ICCV 2025 | ✓ | — | — | — | — | — | 条件互信息最大化和 Anatomy-Guided Calibration；医学多域序贯自监督预训练 | [PDF](pre_training/2025_ICCV_CoSMIC.pdf) |
| **CHIPS** — [CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](https://openaccess.thecvf.com/content/CVPR2026/html/Zhuang_CHIPS_Efficient_CLIP_Adaptation_via_Curvature-aware_Hybrid_Influence-based_Data_Selection_CVPR_2026_paper.html) | CVPR 2026 | ✓ | — | — | — | — | — | 曲率和影响函数驱动的数据选择；CLIP 域持续预训练；单域扩展型 CPT | [PDF](pre_training/2026_CVPR_CHIPS.pdf) |
| **128K CPT** — [Data Engineering for Scaling Language Models to 128K Context](https://proceedings.mlr.press/v235/fu24d.html) | ICML 2024 | ✓ | — | — | — | — | — | 领域平衡与长文本上采样；长上下文能力扩展型 CPT，非多任务遗忘专用方法 | [PDF](pre_training/2024_ICML_128K_CPT.pdf) |
| **DiffuGPT / DiffuLLaMA** — [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0fa81c3f0d57f95b8776de3a248ef0ed-Abstract-Conference.html) | ICLR 2025 | ✓ | — | — | — | — | — | 通过持续预训练从自回归模型适配扩散语言模型；模型范式转换型 CPT，非多任务遗忘专用方法 | [PDF](pre_training/2025_ICLR_DiffuGPT_DiffuLLaMA.pdf) |
| **Mix-CPT** — [Mix-CPT: A Domain Adaptation Framework via Decoupling Knowledge Learning and Format Alignment](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc721384c26c0bdff3ec31a7de31d8d5-Abstract-Conference.html) | ICLR 2025 | ✓ | — | — | — | — | — | 混合领域文本、通用指令和对齐数据，解耦知识与格式；领域适配型 CPT | [PDF](pre_training/2025_ICLR_Mix_CPT.pdf) |
| **CPPO** — [CPPO: Continual Learning for Reinforcement Learning with Human Feedback](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6246f93ebf4f2d76ad2bcb3158220d34-Abstract-Conference.html) | ICLR 2024 | — | — | — | — | — | ✓ | 按奖励和生成概率加权学习及知识保持，持续奖励模型；动态人类偏好下持续 RLHF（核心持续对齐） | [PDF](pre_alignment/2024_ICLR_CPPO.pdf) |
| **SeRA** — [SeRA: Self-Reviewing and Alignment of LLMs using Implicit Reward Margins](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fdcf2565ea86530d65b3622c06d90841-Abstract-Conference.html) | ICLR 2025 | — | — | — | — | — | ✓ | 隐式奖励边际筛样与迭代偏好数据自举；迭代持续偏好优化（扩展阅读，非序贯多任务遗忘基准） | [PDF](pre_alignment/2025_ICLR_SeRA.pdf) |

## 数据与复现

`continual_learning_manifest.json` 保存标题、方法、会议、年份、归类、分类依据、正式来源页、原始 PDF URL、本地相对路径、字节数及 SHA-256。每个下载文件验证 `%PDF-` 文件头，并通过 `pdftotext` 完整解析。`collect_cl_papers.py` 可在服务器使用 Python 3、requests 和 Poppler 的 pdftotext 重新收集；运行会更新本索引及 manifest，不删除原有文件。

本清单是有明确筛选口径的精选阅读列表，非 2024–2026 全量系统综述。建议先阅读 InfLoRA、MoE-Adapters、GIFT 和 CPPO，再按模型与场景比较其他方法。
