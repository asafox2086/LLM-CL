# 逐篇发表与 LLM 任务核验

核验日期：2026-10-11。每篇均检索正式会议论文页；年份按会议届次。新收录逐篇下载正文，复核模型、生成任务、训练设置与主要抗遗忘机制。定位页码为 PDF 页码。

## 当前清单：29 篇

| 方法／正式发表来源 | 年份与发表依据 | 模型与生成任务（人工复核） | 正文定位 | 分类 |
|---|---|---|---|---|
| [DGG](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Multimodal_Continual_Instruction_Tuning_with_Dynamic_Gradient_Guidance_CVPR_2026_paper.html) | CVPR **2026**；正式 CVPR 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026。 | LLaVA-7B；CoIN / UCIT 多模态序贯指令，含开放式 VQA 和图像描述 | LLaVA-7B：p.6；UCIT：p.6,7,8 | 基于正则化 |
| [HPA](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Harmonious_Parameter_Adaptation_in_Continual_Visual_Instruction_Tuning_for_Safety-Aligned_CVPR_2026_paper.html) | CVPR **2026**；正式 CVPR 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026。 | LLaVA-v1.5-7B；六种视觉指令任务序列中保持安全拒答和任务表现；安全对齐保持 | LLaVA-v1.5-7B：p.6；safety：p.1,2,3,4,5,6,7,8,9,10 | 持续对齐 |
| [MaDQ / UCo-VQA](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Re-evaluating_Continual_VQA_Toward_Fair_and_Robust_Evaluation_for_Multimodal_CVPR_2026_paper.html) | CVPR **2026**；正式 CVPR 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026。 | BLIP-2（冻结视觉编码器与通用 LLM 解码器）；UCo-VQA 生成式序贯 VQA；采用大模型扩展实验（表 3），非仅 BLIP 小模型 | BLIP2：p.8；classification-based：p.5；Tab. 3：p.8 | 基于回放 |
| [LifeAlign](https://ojs.aaai.org/index.php/AAAI/article/view/40428) | AAAI **2026**；AAAI 正式主会论文页，卷 40(37)，31618–31626，DOI 10.1609/aaai.v40i37.40428；2026-03-14 已发表，网页工具核验。 | Qwen2.5-7B-Instruct（另有 Llama3.1 / Mistral）；跨领域、偏好类型及安全偏好序贯对齐 | Qwen-2.5-7b：p.5,7；preference：p.1,2,3,4,5,7,8,9,10,11,17 | 持续对齐 |
| [Data Mixing Agent](https://aclanthology.org/2026.acl-long.427/) | ACL **2026**；正式 ACL 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/7。 | LLaMA-DCLM、LLaMA-FWE、LLaMA-Nemo；领域数据混合持续预训练，评测领域、数学等能力与通用知识保持 | LLaMA-DCLM：p.5,6,7,8,17,20,21,22,23,24,25,26；continual pre-training：p.1,2,3,5,9,10,11,19,22,26 | 持续预训练 |
| [FOREVER](https://aclanthology.org/2026.acl-long.1144/) | ACL **2026**；正式 ACL 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/7。 | Qwen3-0.6B/4B、LLaMA-3.1-8B 等；SuperNI 序贯生成任务（摘要、对话等）；不以分类流为依据 | SuperNI：p.5,6,7,8,13,14,15,16,18,19,20,21；Qwen3：p.5,6,8,13,14,15,16 | 基于回放 |
| [Glyph](https://aclanthology.org/2026.acl-long.1722/) | ACL **2026**；正式 ACL 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/7。 | GLM-4.1V-9B-Base → Glyph；视觉化长文 CPT，长上下文问答、摘要与文本生成；上下文扩展型 CPT | GLM-4.1V-9B：p.8,12；continual pre-training：p.1,3,4,5,9,12 | 持续预训练 |
| [PASs-MoE](https://aclanthology.org/2026.acl-long.1474/) | ACL **2026**；正式 ACL 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/7。 | LLaVA-v1.5-7B；多模态序贯视觉问答指令任务与遗忘评测 | LLaVA-v1.5：p.6；VQA：p.5,6,7,13,14 | 基于架构 |
| [PaRSP](https://aclanthology.org/2026.acl-long.1244/) | ACL **2026**；正式 ACL 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/7。 | Llama-3.1-8B、Qwen2.5-7B；TRACE 序贯生成任务、代码、摘要和推理能力保持；另含分类流 | TRACE：p.2,6,7,11,15,24,25,26,30；GSM8K：p.7,11,24,28 | 基于正则化 |
| [SpaRTA](https://aclanthology.org/2026.acl-long.334/) | ACL **2026**；正式 ACL 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/7。 | LLaMA2、LLaMA3.1、Qwen2.5（另有 T5）；TRACE 的 MeetingBank 摘要及其他生成任务；数学能力保持 | MeetingBank：p.3,10,13,14；GSM8K：p.7,8,12 | 基于正则化 |
| [Tibetan Dense / MoE CPT](https://aclanthology.org/2026.acl-long.1866/) | ACL **2026**；正式 ACL 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/7。 | Qwen2.5-7B、50B-A10B MoE；藏／中／英混合语料 CPT，藏语理解与生成；语言扩展型 CPT | Qwen2.5-7B：p.1,2,5,6,7,8；continual pre-training：p.1,2,3,5,6,15 | 持续预训练 |
| [Cite Pretrain / Active Indexing](https://proceedings.iclr.cc/paper_files/paper/2026/hash/528f0a943a54903939e77509f874d8e8-Abstract-Conference.html) | ICLR **2026**；正式 ICLR 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026-04-20。 | Qwen2.5-3B/7B；知识语料 CPT，短／长答案生成与来源引用；知识扩展型 CPT | Qwen-2.5-7B：p.1,3,23,28；continual pretraining：p.2,3,4,5,7,8,9,10,18,22,23,24 | 持续预训练 |
| [OSFT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) | ICLR **2026**；正式 ICLR 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026-04-20。 | LLaMA-2-7B、Mistral-7B（另有 T5）；TRACE 指令、数学推理及领域微调；不以文本分类为依据 | TRACE：p.2,8,9,10,14,28,30,32；GSM8K：p.10,23,24,26,27 | 基于正则化 |
| [SLAO / Merge before Forget](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0dedca10992d34035c9a9776254cfa8c-Abstract-Conference.html) | ICLR **2026**；正式 ICLR 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026-04-20。 | Llama-2-7B/13B、Llama-3.2-3B、Qwen2.5-3B/7B；SuperNI 对话生成、摘要、问答序列；不取另行分类实验 | SuperNI：p.8,9,10,20,21,22,23,24,25,26,27；dialogue generation：p.8,26 | 基于架构 |
| [Weight Space Consolidation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html) | ICLR **2026**；正式 ICLR 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026-04-20。 | LLaMA-3.2-1B；TRACE 序贯指令任务，含会议摘要、代码与数学；不取图像分类实验 | TRACE：p.2,7,8,14,20,22,23,25,26；Llama-3.2：p.7,25 | 基于回放 |
| [ALER](https://proceedings.mlr.press/v306/xia26j.html) | ICML **2026**；正式 ICML 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/09/29。 | Qwen3-4B-Instruct-2507（另有 8B 实验）；化学等领域 SFT 后保留 GSM8K 数学、HumanEval 代码和 IFEval 指令能力 | Qwen3-4B：p.18,19,20,22；HumanEval：p.6,7,8,17,22,23,24,25 | 基于正则化 |
| [CoPE](https://proceedings.mlr.press/v306/wang26le.html) | ICML **2026**；正式 ICML 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/09/29。 | LLaVA-v1.5-7B（另有 13B）；CoIN 八种生成式 VQA 任务序列 | LLaVA-v1.5：p.20,21；CoIN：p.1,2,3,5,6,7,9,14 | 基于架构 |
| [DiSC](https://proceedings.mlr.press/v306/padmanabhan26a.html) | ICML **2026**；正式 ICML 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/09/29。 | Qwen2.5-3B/7B-Instruct、Llama3.1-8B-Instruct、Qwen3-8B；文档知识更新（KUP / BioASQ）；保留代码、数学及指令能力 | BioASQ：p.2,5,6,7,8,11,13,14,15,16；IFEval：p.2,5,6,7,8,9,13,14 | 持续预训练 |
| [Diffract](https://proceedings.mlr.press/v306/borodin26a.html) | ICML **2026**；正式 ICML 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/09/29。 | 1B / 7B / 13B 生成式 LLM；领域持续预训练，数学、代码与文本领域能力适配及保持 | continual pre-training：p.1,2,3,7,8,9,10,12,15,22,24；13B：p.1,2,3,4,5,6,7,8,9,14,15,17 | 持续预训练 |
| [ECA](https://proceedings.mlr.press/v306/kong26h.html) | ICML **2026**；正式 ICML 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/09/29。 | BLIP-2 / OPT-2.7B，补充 LLaVA-v0 实验；按主题序贯训练 COCO / TextCaps 图像描述、VQAv2 / TextVQA 开放式问答 | OPT-2.7B：p.15；ToS-COCO：p.2,6,7,8,14,15,16,18,19 | 基于架构 |
| [Hyper-LLaVA](https://proceedings.mlr.press/v306/xu26cj.html) | ICML **2026**；正式 ICML 2026 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026/09/29。 | LLaVA-v1.5-7B / Vicuna-7B；CoIN / UCIT 开放式 VQA、描述、数学等多模态持续指令任务 | Vicuna-7B：p.6；UCIT：p.6,7,8,17,18 | 基于架构 |
| [CL-MoE](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) | CVPR **2025**；正式 CVPR 2025 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2025。 | LLaVA / Vicuna-7B；多模态序贯生成式 VQA 任务 | Vicuna：p.6,9；VQA：p.1,2,3,6,7,8,9,10 | 基于架构 |
| [Any-SSR](https://openaccess.thecvf.com/content/ICCV2025/html/Tong_Any-SSR_How_Recursive_Least_Squares_Works_in_Continual_Learning_of_ICCV_2025_paper.html) | ICCV **2025**；正式 ICCV 2025 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2025。 | Llama2-7B-chat、Qwen2.5-7B、GLM-4-9B；TRACE 的 MeetingBank 摘要、Py150 代码生成等序贯任务 | summarization：p.6；Py150：p.6,8 | 基于架构 |
| [FCIT](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Federated_Continual_Instruction_Tuning_ICCV_2025_paper.html) | ICCV **2025**；正式 ICCV 2025 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2025。 | LLaVA-1.5-7B；联邦序贯图像描述（Flickr30k）、数学和视觉问答；不以 ImageNet-R 为依据 | LLaVA-1.5-7b：p.7；Flickr30k：p.2,4,10 | 基于架构 |
| [MVP / Instruction-Grounded Visual Projectors](https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Instruction-Grounded_Visual_Projectors_for_Continual_Learning_of_Generative_Vision-Language_Models_ICCV_2025_paper.html) | ICCV **2025**；正式 ICCV 2025 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2025。 | Vicuna-7B，补充 LLaMA-2-7B；Flickr30k / COCO 图像描述与视觉问答序列；不取分类实验 | Vicuna-7B：p.5；Flickr：p.5,10 | 基于架构 |
| [SMoLoRA](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) | ICCV **2025**；正式 ICCV 2025 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2025。 | LLaVA-v1.5-7B；持续视觉指令，含 TextCaps / Flickr30k 图像描述与开放式问答；另有识别任务 | TextCaps：p.5,7,8,10；Flickr：p.5,6,7,8,10 | 基于架构 |
| [FV-guided Training](https://proceedings.iclr.cc/paper_files/paper/2025/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html) | ICLR **2025**；正式 ICLR 2025 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2025-05-01。 | 多种 decoder-only LLM；SuperNI 纯生成和混合任务序列；功能激活与通用生成能力保持 | SuperNI：p.3,4,18,19,21；pure generation：p.3 | 基于正则化 |
| [SERS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/087b346ca4d6dd6fcd0663ff6c9a6b69-Abstract-Conference.html) | NeurIPS **2025**；正式 NeurIPS 2025 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2026-04-23。 | LLaMA2-7B、ChatGLM-6B；SuperNI 摘要、翻译等生成任务序列 | SuperNI：p.2,6,22；ChatGLM：p.2,6,7,8,9,17,29,31,32 | 基于回放 |
| [CPPO](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6246f93ebf4f2d76ad2bcb3158220d34-Abstract-Conference.html) | ICLR **2024**；正式 ICLR 2024 主会论文集页面；会议年份按来源页/BibTeX 的届次核验。页面日期：2024-05-31。 | Llama-7B 策略与奖励模型（附录扩展实验）；主实验另用 GPT2-XL；Reddit TL;DR 摘要与 SHP 跨域人类偏好序列 RLHF | Llama-7B：p.21,22,23；SHP：p.19,21,22,23 | 持续对齐 |

预训练的语言／知识／上下文扩展型 CPT，以及 HPA 的持续安全对齐保持，均按 README 所述边界标注；不能把它们一概描述成标准序贯抗遗忘或偏好优化算法。分类采用主要机制，混合机制见 manifest。

LifeAlign：正式 AAAI 页面已核验发表，本地 PDF 是该页面注明的作者扩展版；发表状态依据正式页面，不以 arXiv 记录判定。其他论文均有正式主会论文集 PDF。

## 原清单：28 篇逐项复核

所有旧条目的正式会议年份均在 2024–2026；主要问题是模型／任务范围。符合范围但较早的 CPT 论文也单独说明替换原因。移除仅针对本次精选目录，旧文件仍可在 Git 历史中找回。

| 原方法与正式来源 | 核实会议年份／已发表 | 处理 | 原因 |
|---|---|---|---|
| [Knowledge replay](https://openaccess.thecvf.com/content/CVPR2024/html/Duan_Towards_Backward-Compatible_Continual_Learning_of_Image_Compression_CVPR_2024_paper.html) | CVPR 2024；已发表；页面依据 2024 | 移出本清单 | 非 LLM：图像压缩持续适配。 |
| [EIR](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_Beyond_Background_Shift_Rethinking_Instance_Replay_in_Continual_Semantic_Segmentation_CVPR_2025_paper.html) | CVPR 2025；已发表；页面依据 2025 | 移出本清单 | 非 LLM：语义分割持续学习。 |
| [View-batch model](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_Do_Your_Best_and_Get_Enough_Rest_for_Continual_Learning_CVPR_2025_paper.html) | CVPR 2025；已发表；页面依据 2025 | 移出本清单 | 通用传统持续学习／分类设置，没有生成式 LLM 持续学习依据。 |
| [GIFT](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Synthetic_Data_is_an_Elegant_GIFT_for_Continual_Vision-Language_Models_CVPR_2025_paper.html) | CVPR 2025；已发表；页面依据 2025 | 移出本清单 | CLIP 图文编码器持续分类，不是 LLM 生成任务。 |
| [Smart Replay](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Smart_Replay_Adaptive_Scheduling_of_Memory_Rehearsal_for_Computational_Resource-Aware_CVPR_2026_paper.html) | CVPR 2026；已发表；页面依据 2026 | 移出本清单 | 传统分类持续学习回放，不是生成式 LLM。 |
| [UCo-VQA baseline](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Re-evaluating_Continual_VQA_Toward_Fair_and_Robust_Evaluation_for_Multimodal_CVPR_2026_paper.html) | CVPR 2026；已发表；页面依据 2026 | 保留 | 保留并更名为 MaDQ / UCo-VQA：正文表 3 明确采用带通用 LLM 解码器的 BLIP-2 生成式序贯 VQA。 |
| [InfLoRA](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html) | CVPR 2024；已发表；页面依据 2024 | 移出本清单 | ViT 图像类别增量实验；不能因 LoRA 名称推断为 LLM。 |
| [One-for-More](https://openaccess.thecvf.com/content/CVPR2025/html/Li_One-for-More_Continual_Diffusion_Model_for_Anomaly_Detection_CVPR_2025_paper.html) | CVPR 2025；已发表；页面依据 2025 | 移出本清单 | 扩散模型异常检测／适配，不是 LLM。 |
| [ConceptGuard](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_ConceptGuard_Continual_Personalized_Text-to-Image_Generation_with_Forgetting_and_Confusion_Mitigation_CVPR_2025_paper.html) | CVPR 2025；已发表；页面依据 2025 | 移出本清单 | 扩散生成模型概念遗忘，不是 LLM 持续学习。 |
| [CoMasTRe](https://openaccess.thecvf.com/content/CVPR2024/html/Gong_Continual_Segmentation_with_Disentangled_Objectness_Learning_and_Class_Recognition_CVPR_2024_paper.html) | CVPR 2024；已发表；页面依据 2024 | 移出本清单 | 传统语义分割持续学习，不是 LLM。 |
| [LfI](https://openaccess.thecvf.com/content/CVPR2026/html/Gong_Learning_from_Itself_Mining_Internal_Knowledge_from_Vision_Language_Models_CVPR_2026_paper.html) | CVPR 2026；已发表；页面依据 2026 | 移出本清单 | CLIP 持续分类；伪字幕辅助蒸馏不足以构成 LLM 持续生成实验。 |
| [EWC Done Right](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Elastic_Weight_Consolidation_Done_Right_for_Continual_Learning_CVPR_2026_paper.html) | CVPR 2026；已发表；页面依据 2026 | 移出本清单 | 通用持续学习权重正则／分类实验；无本文所需 LLM 生成依据。 |
| [CCL-DC](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Improving_Plasticity_in_Online_Continual_Learning_via_Collaborative_Learning_CVPR_2024_paper.html) | CVPR 2024；已发表；页面依据 2024 | 移出本清单 | 传统在线类别增量学习。 |
| [MoE-Adapters / DDAS](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.html) | CVPR 2024；已发表；页面依据 2024 | 移出本清单 | CLIP 模型持续分类，非生成式 LLM。 |
| [CPrompt](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Consistent_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2024_paper.html) | CVPR 2024；已发表；页面依据 2024 | 移出本清单 | ViT 提示与分类器的类别增量实验。 |
| [S6MOD](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Enhancing_Online_Continual_Learning_with_Plug-and-Play_State_Space_Model_and_CVPR_2025_paper.html) | CVPR 2025；已发表；页面依据 2025 | 移出本清单 | 传统在线类别增量与状态空间插件。 |
| [CHEEM](https://openaccess.thecvf.com/content/CVPR2026/html/Savadikar_CHEEM_Continual_Learning_by_Reuse_New_Adapt_and_Skip_-_CVPR_2026_paper.html) | CVPR 2026；已发表；页面依据 2026 | 移出本清单 | ViT 持续分类和结构搜索，非 LLM。 |
| [GRAPA](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Parameter-efficient_Continual_Learning_for_Enhancing_Plasticity_without_Forgetting_under_Limited_CVPR_2026_paper.html) | CVPR 2026；已发表；页面依据 2026 | 移出本清单 | 传统模型容量／参数分配持续分类，非生成式 LLM。 |
| [MedQwen / Sparse Spectral LoRA](https://openaccess.thecvf.com/content/CVPR2026/html/Nejatimanzari_Sparse_Spectral_LoRA_Routed_Experts_for_Medical_VLMs_CVPR_2026_paper.html) | CVPR 2026；已发表；页面依据 2026 | 移出本清单 | 虽然整体模型有生成能力，但抗遗忘实验主要衡量 FairVLMed 分类 → PathVQA 后的原分类准确率，不以此满足本轮生成式持续学习筛选。 |
| [Learn more, but bother less](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b0bc711f48724237b38823c4d9cee10b-Abstract-Conference.html) | NeurIPS 2024；已发表；页面依据 2024-12-16 | 移出本清单 | 确有 LLM，但主要序贯训练为 AG News / Amazon / DBpedia / GLUE 等分类流；不按所需生成式持续任务收录。 |
| [MedCoSS](https://openaccess.thecvf.com/content/CVPR2024/html/Ye_Continual_Self-supervised_Learning_Towards_Universal_Multi-modal_Medical_Data_Representation_Learning_CVPR_2024_paper.html) | CVPR 2024；已发表；页面依据 2024 | 移出本清单 | 医学多模态视觉自监督表征，不是 LLM 语言建模。 |
| [CoSMIC](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_CoSMIC_Continual_Self-supervised_Learning_for_Multi-Domain_Medical_Imaging_via_Conditional_ICCV_2025_paper.html) | ICCV 2025；已发表；页面依据 2025 | 移出本清单 | 医学视觉多模态自监督，不是 LLM 持续预训练。 |
| [CHIPS](https://openaccess.thecvf.com/content/CVPR2026/html/Zhuang_CHIPS_Efficient_CLIP_Adaptation_via_Curvature-aware_Hybrid_Influence-based_Data_Selection_CVPR_2026_paper.html) | CVPR 2026；已发表；页面依据 2026 | 移出本清单 | CLIP 图文对比预训练，不含 LLM 解码器生成式持续学习。 |
| [128K CPT](https://proceedings.mlr.press/v235/fu24d.html) | ICML 2024；已发表；页面依据 2024/07/08 | 移出本清单 | 符合 LLM 上下文扩展 CPT，但为 2024；本轮优先用 2026 的领域／上下文 CPT 代表替换。 |
| [DiffuGPT / DiffuLLaMA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0fa81c3f0d57f95b8776de3a248ef0ed-Abstract-Conference.html) | ICLR 2025；已发表；页面依据 2025-05-01 | 移出本清单 | LLM 扩散范式转换型继续预训练，但不以标准序贯抗遗忘为重点；本轮换为更新的 2026 CPT 代表。 |
| [Mix-CPT](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc721384c26c0bdff3ec31a7de31d8d5-Abstract-Conference.html) | ICLR 2025；已发表；页面依据 2025-05-01 | 移出本清单 | 符合 LLM 领域适配 CPT；按较新优先原则换为 2026 CPT 代表，并非发表年份或模型不合格。 |
| [CPPO](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6246f93ebf4f2d76ad2bcb3158220d34-Abstract-Conference.html) | ICLR 2024；已发表；页面依据 2024-05-31 | 保留 | 保留：动态偏好下持续 RLHF；附录使用 Llama-7B 策略／奖励模型。 |
| [SeRA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fdcf2565ea86530d65b3622c06d90841-Abstract-Conference.html) | ICLR 2025；已发表；页面依据 2025-05-01 | 移出本清单 | 正式发表但属于偏好数据迭代自举，缺少本轮要求的明确序贯任务／偏好保持设定；换为 LifeAlign 等。 |

机器可读的旧清单核验见 [previous_collection_audit.json](previous_collection_audit.json)。当前来源、模型任务定位、下载 URL、文件大小和 SHA-256 见 [continual_learning_manifest.json](continual_learning_manifest.json)。

## 额外检索但未入选

- MedQwen 的生成模型身份不能代替生成式持续任务证据，因此移除上述分类保持实验。
- Meta-UCF（ICLR 2026）有混合生成／分类流，但为使架构部分更贴近用户优先的计算机视觉顶会与清晰生成任务设置，本轮选用 ICCV 2025 FCIT，不将其错误断言为未发表或完全没有生成任务。
- KPIG（ACL 2025）与 SEEKR（EMNLP 2024）符合 LLM 生成式持续微调范围，但较新优先，本轮分别选择 CVPR 2026 MaDQ 与 ICLR 2025 FV-guided Training。
