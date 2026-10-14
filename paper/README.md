# 2024–2026 顶会大语言模型持续学习论文

逐篇检索与正文核验日期：**2026-10-11**。年份按正式会议届次，而非 arXiv 首次上传年份或网站上线年份。仅收录已进入正式主会论文集的论文，不含 workshop、Findings、仅录用名单或未发表预印本。

共 **29 篇**：持续预训练 6 篇、持续微调 **20 篇**、持续对齐 3 篇。年份：2026 年 21 篇、2025 年 7 篇、2024 年 1 篇。会议：CVPR 4 篇、ICCV 4 篇、ACL 7 篇、ICML 6 篇、ICLR 6 篇、NeurIPS 1 篇、AAAI 1 篇。各列按年份降序，同年优先 CVPR / ICCV。

## 目录

```text
paper/
├── README.md
├── verification_audit.md          # 逐篇发表与任务核验；旧清单处理记录
├── continual_learning_manifest.json
├── collect_cl_papers.py
├── pre_training/                  # 6
├── pre_finetuning/                # 20
│   ├── method_replay/             # 4
│   ├── method_regularization/     # 6
│   └── method_architecture/       # 10
└── pre_alignment/                # 3
```

服务器路径：`/data2/liyapeng_grp/program/LLMCL/paper`。原有顶层医学论文不计入本清单。

## 收录与分类口径

- **模型与任务**：文本 LLM 或带 LLM 解码器的生成式 MLLM，实际研究持续适配、序贯指令学习或持续能力保持。排除 CLIP / ViT 分类、图像分割、扩散模型、普通视觉自监督，以及仅把分类标签转成文本的 LLM 实验。混合评测论文必须包含实质性的生成实验；本表只以生成、代码、数学、开放式问答、摘要或描述任务为收录依据。
- **持续预训练**：使用语言建模或知识蒸馏目标继续训练。Diffract、Data Mixing Agent 等研究领域适配与能力保持；藏语 CPT、Cite Pretrain、Glyph 属语言、知识或上下文扩展型 CPT，**不能据此宣称具有标准序贯多任务抗遗忘实验**。DiSC 是对已后训练模型进行文档知识更新，按训练目标归入此类。
- **持续微调**：序贯监督指令或任务适配。回放、正则化、架构按主要抗遗忘机制归档，每篇只存一份；混合机制在详情中注明。ECA 的 alignment 是跨模态映射适配，归入微调的架构类。
- **持续对齐**：LifeAlign / CPPO 是序贯偏好或奖励对齐；HPA 研究已安全对齐模型在持续视觉指令微调中的安全保持，属于持续安全对齐保持，**不是新的 RLHF 算法**。
- **较新优先**：优先 2026，适当保留 2025 的代表方法。2024 仅保留明确研究动态偏好持续 RLHF 的 CPPO。

## 方法 × 方向二维表

横轴为方向，持续微调下设三个子类；纵向列出方法（点击方法名查看正式发表页面）。同一行不同列之间没有对应关系。

**文献收录与实验选择分开记录：回放类论文继续完整保留在此二维表、PDF 目录和源码索引中，仅不参加本轮实验比较。** 当前实验方法及小预算见 [010 执行协议](../rules/010_equal_budget_execution.md)。

<table>
<thead>
<tr><th rowspan="2">持续预训练（6）</th><th colspan="3">持续微调（20）</th><th rowspan="2">持续对齐（3）</th></tr>
<tr><th>基于回放（4）</th><th>基于正则化（6）</th><th>基于架构（10）</th></tr>
</thead>
<tbody>
<tr>
<td><a href="https://aclanthology.org/2026.acl-long.427/" title="Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training"><strong>Data Mixing Agent</strong></a><br>ACL 2026<br><a href="pre_training/2026_ACL_Data_Mixing_Agent.pdf">PDF</a></td>
<td><a href="https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Re-evaluating_Continual_VQA_Toward_Fair_and_Robust_Evaluation_for_Multimodal_CVPR_2026_paper.html" title="Re-evaluating Continual VQA: Toward Fair and Robust Evaluation for Multimodal Continual Learning"><strong>MaDQ / UCo-VQA</strong></a><br>CVPR 2026<br><a href="pre_finetuning/method_replay/2026_CVPR_MaDQ_UCo_VQA.pdf">PDF</a></td>
<td><a href="https://openaccess.thecvf.com/content/CVPR2026/html/Li_Multimodal_Continual_Instruction_Tuning_with_Dynamic_Gradient_Guidance_CVPR_2026_paper.html" title="Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance"><strong>DGG</strong></a><br>CVPR 2026<br><a href="pre_finetuning/method_regularization/2026_CVPR_DGG.pdf">PDF</a></td>
<td><a href="https://aclanthology.org/2026.acl-long.1474/" title="PASs-MoE: Mitigating Misaligned Co-drift among Router and Experts via Pathway Activation Subspaces for Continual Learning"><strong>PASs-MoE</strong></a><br>ACL 2026<br><a href="pre_finetuning/method_architecture/2026_ACL_PASs_MoE.pdf">PDF</a></td>
<td><a href="https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Harmonious_Parameter_Adaptation_in_Continual_Visual_Instruction_Tuning_for_Safety-Aligned_CVPR_2026_paper.html" title="Harmonious Parameter Adaptation in Continual Visual Instruction Tuning for Safety-Aligned MLLMs"><strong>HPA</strong></a><br>CVPR 2026<br><a href="pre_alignment/2026_CVPR_HPA.pdf">PDF</a></td>
</tr>
<tr>
<td><a href="https://aclanthology.org/2026.acl-long.1722/" title="Glyph: Scaling Context Windows via Visual-Text Compression"><strong>Glyph</strong></a><br>ACL 2026<br><a href="pre_training/2026_ACL_Glyph.pdf">PDF</a></td>
<td><a href="https://aclanthology.org/2026.acl-long.1144/" title="FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning"><strong>FOREVER</strong></a><br>ACL 2026<br><a href="pre_finetuning/method_replay/2026_ACL_FOREVER.pdf">PDF</a></td>
<td><a href="https://aclanthology.org/2026.acl-long.1244/" title="Harmonizing the Past, Present, and Future: A Null-Space Constrained Region-Specific Method for Continual Learning in LLMs"><strong>PaRSP</strong></a><br>ACL 2026<br><a href="pre_finetuning/method_regularization/2026_ACL_PaRSP.pdf">PDF</a></td>
<td><a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/0dedca10992d34035c9a9776254cfa8c-Abstract-Conference.html" title="Merge before Forget: A Single LoRA Continual Learning via Continual Merging"><strong>SLAO / Merge before Forget</strong></a><br>ICLR 2026<br><a href="pre_finetuning/method_architecture/2026_ICLR_SLAO_Merge_before_Forget.pdf">PDF</a></td>
<td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/40428" title="LifeAlign: Lifelong Alignment for Large Language Models with Memory-Augmented Focalized Preference Optimization"><strong>LifeAlign</strong></a><br>AAAI 2026<br><a href="pre_alignment/2026_AAAI_LifeAlign.pdf">PDF</a><br>作者扩展版</td>
</tr>
<tr>
<td><a href="https://aclanthology.org/2026.acl-long.1866/" title="From Curated Data to Scalable Models: Continual Pre-training of Dense and MoE Large Language Models for Tibetan"><strong>Tibetan Dense / MoE CPT</strong></a><br>ACL 2026<br><a href="pre_training/2026_ACL_Tibetan_Dense_MoE_CPT.pdf">PDF</a></td>
<td><a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html" title="Forget Forgetting: Continual Learning in a World of Abundant Memory"><strong>Weight Space Consolidation</strong></a><br>ICLR 2026<br><a href="pre_finetuning/method_replay/2026_ICLR_Weight_Space_Consolidation.pdf">PDF</a></td>
<td><a href="https://aclanthology.org/2026.acl-long.334/" title="Spectral Disentanglement: Rank-Aware Task Adaptation for Rehearsal-free Continual Learning in LLMs"><strong>SpaRTA</strong></a><br>ACL 2026<br><a href="pre_finetuning/method_regularization/2026_ACL_SpaRTA.pdf">PDF</a></td>
<td><a href="https://proceedings.mlr.press/v306/wang26le.html" title="CoPE: Continual Probe-guided Expansion for Large Vision-Language Models"><strong>CoPE</strong></a><br>ICML 2026<br><a href="pre_finetuning/method_architecture/2026_ICML_CoPE.pdf">PDF</a></td>
<td><a href="https://proceedings.iclr.cc/paper_files/paper/2024/hash/6246f93ebf4f2d76ad2bcb3158220d34-Abstract-Conference.html" title="CPPO: Continual Learning for Reinforcement Learning with Human Feedback"><strong>CPPO</strong></a><br>ICLR 2024<br><a href="pre_alignment/2024_ICLR_CPPO.pdf">PDF</a></td>
</tr>
<tr>
<td><a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/528f0a943a54903939e77509f874d8e8-Abstract-Conference.html" title="Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models"><strong>Cite Pretrain / Active Indexing</strong></a><br>ICLR 2026<br><a href="pre_training/2026_ICLR_Cite_Pretrain_Active_Indexing.pdf">PDF</a></td>
<td><a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/087b346ca4d6dd6fcd0663ff6c9a6b69-Abstract-Conference.html" title="Self-Evolving Pseudo-Rehearsal for Catastrophic Forgetting with Task Similarity in LLMs"><strong>SERS</strong></a><br>NeurIPS 2025<br><a href="pre_finetuning/method_replay/2025_NeurIPS_SERS.pdf">PDF</a></td>
<td><a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html" title="Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning"><strong>OSFT</strong></a><br>ICLR 2026<br><a href="pre_finetuning/method_regularization/2026_ICLR_OSFT.pdf">PDF</a></td>
<td><a href="https://proceedings.mlr.press/v306/kong26h.html" title="ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation"><strong>ECA</strong></a><br>ICML 2026<br><a href="pre_finetuning/method_architecture/2026_ICML_ECA.pdf">PDF</a></td>
<td>—</td>
</tr>
<tr>
<td><a href="https://proceedings.mlr.press/v306/padmanabhan26a.html" title="Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities"><strong>DiSC</strong></a><br>ICML 2026<br><a href="pre_training/2026_ICML_DiSC.pdf">PDF</a></td>
<td>—</td>
<td><a href="https://proceedings.mlr.press/v306/xia26j.html" title="Adversarial Latent Embedding Repair for LLM Continual Learning"><strong>ALER</strong></a><br>ICML 2026<br><a href="pre_finetuning/method_regularization/2026_ICML_ALER.pdf">PDF</a></td>
<td><a href="https://proceedings.mlr.press/v306/xu26cj.html" title="Hyper-LLaVA: Hyperbolic Uncertainty-aware Modality-Balanced Routing for Multimodal Continual Instruction Tuning"><strong>Hyper-LLaVA</strong></a><br>ICML 2026<br><a href="pre_finetuning/method_architecture/2026_ICML_Hyper_LLaVA.pdf">PDF</a></td>
<td>—</td>
</tr>
<tr>
<td><a href="https://proceedings.mlr.press/v306/borodin26a.html" title="Diffract: Spectral View of LLM Domain Adaptation"><strong>Diffract</strong></a><br>ICML 2026<br><a href="pre_training/2026_ICML_Diffract.pdf">PDF</a></td>
<td>—</td>
<td><a href="https://proceedings.iclr.cc/paper_files/paper/2025/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html" title="Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning"><strong>FV-guided Training</strong></a><br>ICLR 2025<br><a href="pre_finetuning/method_regularization/2025_ICLR_FV_guided_Training.pdf">PDF</a></td>
<td><a href="https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html" title="CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering"><strong>CL-MoE</strong></a><br>CVPR 2025<br><a href="pre_finetuning/method_architecture/2025_CVPR_CL_MoE.pdf">PDF</a></td>
<td>—</td>
</tr>
<tr>
<td>—</td>
<td>—</td>
<td>—</td>
<td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Tong_Any-SSR_How_Recursive_Least_Squares_Works_in_Continual_Learning_of_ICCV_2025_paper.html" title="Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Model"><strong>Any-SSR</strong></a><br>ICCV 2025<br><a href="pre_finetuning/method_architecture/2025_ICCV_Any_SSR.pdf">PDF</a></td>
<td>—</td>
</tr>
<tr>
<td>—</td>
<td>—</td>
<td>—</td>
<td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Federated_Continual_Instruction_Tuning_ICCV_2025_paper.html" title="Federated Continual Instruction Tuning"><strong>FCIT</strong></a><br>ICCV 2025<br><a href="pre_finetuning/method_architecture/2025_ICCV_FCIT.pdf">PDF</a></td>
<td>—</td>
</tr>
<tr>
<td>—</td>
<td>—</td>
<td>—</td>
<td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Instruction-Grounded_Visual_Projectors_for_Continual_Learning_of_Generative_Vision-Language_Models_ICCV_2025_paper.html" title="Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models"><strong>MVP / Instruction-Grounded Visual Projectors</strong></a><br>ICCV 2025<br><a href="pre_finetuning/method_architecture/2025_ICCV_MVP_Instruction_Grounded_Visual_Projectors.pdf">PDF</a></td>
<td>—</td>
</tr>
<tr>
<td>—</td>
<td>—</td>
<td>—</td>
<td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html" title="SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning"><strong>SMoLoRA</strong></a><br>ICCV 2025<br><a href="pre_finetuning/method_architecture/2025_ICCV_SMoLoRA.pdf">PDF</a></td>
<td>—</td>
</tr>
</tbody>
</table>

## 逐篇详情

<details>
<summary>展开完整标题、模型、生成任务和分类依据</summary>

| 方法与正式论文 | 会议 | 模型及生成任务 | 主要机制与归类 |
|---|---|---|---|
| **Data Mixing Agent** — [Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](https://aclanthology.org/2026.acl-long.427/) | ACL 2026 | LLaMA-DCLM、LLaMA-FWE、LLaMA-Nemo；领域数据混合持续预训练，评测领域、数学等能力与通用知识保持 | 持续预训练：通过强化学习训练数据域混合策略；LLM 领域持续预训练与通用能力保持 |
| **Glyph** — [Glyph: Scaling Context Windows via Visual-Text Compression](https://aclanthology.org/2026.acl-long.1722/) | ACL 2026 | GLM-4.1V-9B-Base → Glyph；视觉化长文 CPT，长上下文问答、摘要与文本生成；上下文扩展型 CPT | 持续预训练：将长文本渲染为视觉页，持续预训练压缩表征；生成式 VLM 长上下文扩展；能力扩展型 CPT |
| **Tibetan Dense / MoE CPT** — [From Curated Data to Scalable Models: Continual Pre-training of Dense and MoE Large Language Models for Tibetan](https://aclanthology.org/2026.acl-long.1866/) | ACL 2026 | Qwen2.5-7B、50B-A10B MoE；藏／中／英混合语料 CPT，藏语理解与生成；语言扩展型 CPT | 持续预训练：藏／中／英平衡混合预训练及 MoE 扩展；Qwen2.5-7B 与 50B-A10B 藏语生成模型；语言扩展型 CPT |
| **Cite Pretrain / Active Indexing** — [Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/528f0a943a54903939e77509f874d8e8-Abstract-Conference.html) | ICLR 2026 | Qwen2.5-3B/7B；知识语料 CPT，短／长答案生成与来源引用；知识扩展型 CPT | 持续预训练：以双向合成数据建立事实与来源 ID 的参数化绑定；Qwen2.5 3B/7B 持续知识预训练，短／长文本引用生成 |
| **DiSC** — [Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities](https://proceedings.mlr.press/v306/padmanabhan26a.html) | ICML 2026 | Qwen2.5-3B/7B-Instruct、Llama3.1-8B-Instruct、Qwen3-8B；文档知识更新（KUP / BioASQ）；保留代码、数学及指令能力 | 持续预训练：分割上下文产生师生分布，KL 蒸馏语言建模；post-trained LLM 以文档语料持续知识适配，保留指令／推理能力 |
| **Diffract** — [Diffract: Spectral View of LLM Domain Adaptation](https://proceedings.mlr.press/v306/borodin26a.html) | ICML 2026 | 1B / 7B / 13B 生成式 LLM；领域持续预训练，数学、代码与文本领域能力适配及保持 | 持续预训练：CPT 参数谱分析与低重要性注意力头回退；十亿参数 LLM 数学／代码／文本领域持续预训练 |
| **MaDQ / UCo-VQA** — [Re-evaluating Continual VQA: Toward Fair and Robust Evaluation for Multimodal Continual Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Re-evaluating_Continual_VQA_Toward_Fair_and_Robust_Evaluation_for_Multimodal_CVPR_2026_paper.html) | CVPR 2026 | BLIP-2（冻结视觉编码器与通用 LLM 解码器）；UCo-VQA 生成式序贯 VQA；采用大模型扩展实验（表 3），非仅 BLIP 小模型 | 基于回放：仅问题回放；兼有答案蒸馏和动量适配器；仅采用 BLIP-2 大模型生成式序贯 VQA 实验，非固定答案分类 |
| **FOREVER** — [FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](https://aclanthology.org/2026.acl-long.1144/) | ACL 2026 | Qwen3-0.6B/4B、LLaMA-3.1-8B 等；SuperNI 序贯生成任务（摘要、对话等）；不以分类流为依据 | 基于回放：按优化器更新幅度定义模型时间，按遗忘曲线调度回放；LLM 持续指令微调，生成任务与知识保持 |
| **Weight Space Consolidation** — [Forget Forgetting: Continual Learning in a World of Abundant Memory](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html) | ICLR 2026 | LLaMA-3.2-1B；TRACE 序贯指令任务，含会议摘要、代码与数学；不取图像分类实验 | 基于回放：充足历史样本回放；兼有秩重置与权重平均；仅取本文 LLM 持续指令微调实验，排除图像分类实验 |
| **SERS** — [Self-Evolving Pseudo-Rehearsal for Catastrophic Forgetting with Task Similarity in LLMs](https://proceedings.neurips.cc/paper_files/paper/2025/hash/087b346ca4d6dd6fcd0663ff6c9a6b69-Abstract-Conference.html) | NeurIPS 2025 | LLaMA2-7B、ChatGLM-6B；SuperNI 摘要、翻译等生成任务序列 | 基于回放：自生成伪输入和自演化标签回放；兼有任务相似性正则；LLM 持续任务微调 |
| **DGG** — [Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Multimodal_Continual_Instruction_Tuning_with_Dynamic_Gradient_Guidance_CVPR_2026_paper.html) | CVPR 2026 | LLaVA-7B；CoIN / UCIT 多模态序贯指令，含开放式 VQA 和图像描述 | 基于正则化：历史最优参数方向近似旧任务梯度；兼有小回放缓冲和 Bernoulli 采样；MLLM 多模态生成式持续指令微调 |
| **PaRSP** — [Harmonizing the Past, Present, and Future: A Null-Space Constrained Region-Specific Method for Continual Learning in LLMs](https://aclanthology.org/2026.acl-long.1244/) | ACL 2026 | Llama-3.1-8B、Qwen2.5-7B；TRACE 序贯生成任务、代码、摘要和推理能力保持；另含分类流 | 基于正则化：任务区域划分和零空间约束；兼有参数分区；LLM 序贯任务适配 |
| **SpaRTA** — [Spectral Disentanglement: Rank-Aware Task Adaptation for Rehearsal-free Continual Learning in LLMs](https://aclanthology.org/2026.acl-long.334/) | ACL 2026 | LLaMA2、LLaMA3.1、Qwen2.5（另有 T5）；TRACE 的 MeetingBank 摘要及其他生成任务；数学能力保持 | 基于正则化：秩谱分解隔离共享／任务特定子空间；兼有自适应 LoRA；LLM rehearsal-free 持续微调 |
| **OSFT** — [Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) | ICLR 2026 | LLaMA-2-7B、Mistral-7B（另有 T5）；TRACE 指令、数学推理及领域微调；不以文本分类为依据 | 基于正则化：自适应 SVD 保留重要子空间，约束参数更新；LLM 持续领域适配和生成能力保持 |
| **ALER** — [Adversarial Latent Embedding Repair for LLM Continual Learning](https://proceedings.mlr.press/v306/xia26j.html) | ICML 2026 | Qwen3-4B-Instruct-2507（另有 8B 实验）；化学等领域 SFT 后保留 GSM8K 数学、HumanEval 代码和 IFEval 指令能力 | 基于正则化：对抗搜索潜在提示嵌入，参考模型在线蒸馏；LLM 领域微调与通用生成／推理能力保持 |
| **FV-guided Training** — [Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html) | ICLR 2025 | 多种 decoder-only LLM；SuperNI 纯生成和混合任务序列；功能激活与通用生成能力保持 | 基于正则化：正则约束任务 function vector，稳定功能激活；LLM 持续指令微调与生成能力保持 |
| **PASs-MoE** — [PASs-MoE: Mitigating Misaligned Co-drift among Router and Experts via Pathway Activation Subspaces for Continual Learning](https://aclanthology.org/2026.acl-long.1474/) | ACL 2026 | LLaVA-v1.5-7B；多模态序贯视觉问答指令任务与遗忘评测 | 基于架构：基于专家通路激活子空间的路由；兼有秩方向稳定约束；MLLM 持续指令微调 |
| **SLAO / Merge before Forget** — [Merge before Forget: A Single LoRA Continual Learning via Continual Merging](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0dedca10992d34035c9a9776254cfa8c-Abstract-Conference.html) | ICLR 2026 | Llama-2-7B/13B、Llama-3.2-3B、Qwen2.5-3B/7B；SuperNI 对话生成、摘要、问答序列；不取另行分类实验 | 基于架构：保持单 LoRA 的序贯合并；兼有正交初始化及时间缩放；LLM 持续微调 |
| **CoPE** — [CoPE: Continual Probe-guided Expansion for Large Vision-Language Models](https://proceedings.mlr.press/v306/wang26le.html) | ICML 2026 | LLaVA-v1.5-7B（另有 13B）；CoIN 八种生成式 VQA 任务序列 | 基于架构：探针引导专家扩展与概率任务定位；大视觉语言模型，CoIN 多任务生成式 VQA |
| **ECA** — [ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation](https://proceedings.mlr.press/v306/kong26h.html) | ICML 2026 | BLIP-2 / OPT-2.7B，补充 LLaVA-v0 实验；按主题序贯训练 COCO / TextCaps 图像描述、VQAv2 / TextVQA 开放式问答 | 基于架构：查询专家混合与 Fisher 信息引导扩展；兼有字典回放；生成式图像到文本持续适配；此处 alignment 指跨模态对齐而非偏好对齐 |
| **Hyper-LLaVA** — [Hyper-LLaVA: Hyperbolic Uncertainty-aware Modality-Balanced Routing for Multimodal Continual Instruction Tuning](https://proceedings.mlr.press/v306/xu26cj.html) | ICML 2026 | LLaVA-v1.5-7B / Vicuna-7B；CoIN / UCIT 开放式 VQA、描述、数学等多模态持续指令任务 | 基于架构：双曲空间不确定性建模和模态平衡参数路由；MLLM 多模态持续指令微调 |
| **CL-MoE** — [CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) | CVPR 2025 | LLaVA / Vicuna-7B；多模态序贯生成式 VQA 任务 | 基于架构：双层路由和动量专家；MLLM 持续视觉问答 |
| **Any-SSR** — [Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Model](https://openaccess.thecvf.com/content/ICCV2025/html/Tong_Any-SSR_How_Recursive_Least_Squares_Works_in_Continual_Learning_of_ICCV_2025_paper.html) | ICCV 2025 | Llama2-7B-chat、Qwen2.5-7B、GLM-4-9B；TRACE 的 MeetingBank 摘要、Py150 代码生成等序贯任务 | 基于架构：解析子空间路由和递归最小二乘路由更新；LLM 序贯任务微调 |
| **FCIT** — [Federated Continual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Federated_Continual_Instruction_Tuning_ICCV_2025_paper.html) | ICCV 2025 | LLaVA-1.5-7B；联邦序贯图像描述（Flickr30k）、数学和视觉问答；不以 ImageNet-R 为依据 | 基于架构：动态知识组织和子空间选择性激活；MLLM 联邦持续指令微调；图像描述、开放式视觉问答与推理 |
| **MVP / Instruction-Grounded Visual Projectors** — [Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models](https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Instruction-Grounded_Visual_Projectors_for_Continual_Learning_of_Generative_Vision-Language_Models_ICCV_2025_paper.html) | ICCV 2025 | Vicuna-7B，补充 LLaMA-2-7B；Flickr30k / COCO 图像描述与视觉问答序列；不取分类实验 | 基于架构：指令语义驱动投影器专家推荐与剪枝；生成式 VLM 持续多任务学习 |
| **SMoLoRA** — [SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) | ICCV 2025 | LLaVA-v1.5-7B；持续视觉指令，含 TextCaps / Flickr30k 图像描述与开放式问答；另有识别任务 | 基于架构：视觉／语言专家路由和低秩适配；MLLM 生成式持续视觉指令微调 |
| **HPA** — [Harmonious Parameter Adaptation in Continual Visual Instruction Tuning for Safety-Aligned MLLMs](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Harmonious_Parameter_Adaptation_in_Continual_Visual_Instruction_Tuning_for_Safety-Aligned_CVPR_2026_paper.html) | CVPR 2026 | LLaVA-v1.5-7B；六种视觉指令任务序列中保持安全拒答和任务表现；安全对齐保持 | 持续对齐：参数分区、平衡选择和正交调整；安全已对齐 MLLM 的持续视觉指令微调，保持安全对齐与任务性能 |
| **LifeAlign** — [LifeAlign: Lifelong Alignment for Large Language Models with Memory-Augmented Focalized Preference Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/40428) | AAAI 2026 | Qwen2.5-7B-Instruct（另有 Llama3.1 / Mistral）；跨领域、偏好类型及安全偏好序贯对齐 | 持续对齐：focalized preference optimization 与短到长期偏好记忆；LLM 跨领域／偏好类型序贯对齐 |
| **CPPO** — [CPPO: Continual Learning for Reinforcement Learning with Human Feedback](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6246f93ebf4f2d76ad2bcb3158220d34-Abstract-Conference.html) | ICLR 2024 | Llama-7B 策略与奖励模型（附录扩展实验）；主实验另用 GPT2-XL；Reddit TL;DR 摘要与 SHP 跨域人类偏好序列 RLHF | 持续对齐：奖励／生成概率加权持续 PPO 与知识保持；生成式 LLM 动态偏好序贯 RLHF |

</details>

<!-- finetuning-author-source-20261012 -->

持续微调方法的作者源码、固定版本和可用性见 [源码索引](../code/README.md)。各方法独立存放在服务器 `code/<method>/source/`；未公开源码或只有 README 的项目如实标注。源样本量见 [实验准备报告](../rules/008_current_task_counts_and_methods.md)，当前预算与实验选择见 [012](../rules/012_small_data_results_only.md)。

## 核验与下载

每篇的正式来源、发表年份依据、正文模型／任务定位、SHA-256 和本地 PDF 路径见 [逐篇核验记录](verification_audit.md) 及 [manifest](continual_learning_manifest.json)。旧清单 28 篇全部重新检索，保留和移除理由逐项记录。归类由整理者按方法正文判断。

**LifeAlign** 已通过 [AAAI 正式论文页](https://ojs.aaai.org/index.php/AAAI/article/view/40428) 核实发表（2026-03-14，DOI: 10.1609/aaai.v40i37.40428）。服务器无法访问 AAAI 下载端点，本地保存正式页面注明的 arXiv 作者扩展版，表中明确标记；并非将未发表预印本计入。其余 PDF 均来自正式会议论文集或 PMLR 官方资产仓库。

复现：Python 3 与 Poppler 的 `pdftotext`，下载时另需 requests。运行 `python3 collect_cl_papers.py` 校验所有 PDF；`--download` 补充缺失文件，`--render` 从人工核验过的 manifest 重建 README。脚本不会自动认定录用、替换筛选标准或删除其他文件。

这是按上述口径整理的精选清单，不是全部相关论文的穷尽性综述。
