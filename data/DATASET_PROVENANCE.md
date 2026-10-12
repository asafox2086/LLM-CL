# 数据集提出者与实际使用证据

更新：2026-10-12。用于论文数据集部分的溯源，**不是十任务已经全部训练就绪的声明**。

原始提出论文与后续使用论文分开列出；训练、回答生成评测、检索评测和基准构建分别注明。只把有实验正文/结果/作者代码依据的工作计入使用记录。原始数据集允许早于 2024；2024–2026 是持续学习方法论文收集范围。

机器可读记录：[dataset_provenance.json](dataset_provenance.json)；引用：[dataset_references.bib](dataset_references.bib)。以下作者均为原论文作者，格式化镜像上传者不是数据集提出者。

| 数据集 / 领域 | 谁提出 / 原始发表 | 谁实际使用 / 用途 | 当前角色 |
|---|---|---|---|
| VQAv2 / 自然/日常场景 | [Yash Goyal 等：Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering](https://openaccess.thecvf.com/content_cvpr_2017/papers/Goyal_Making_the_v_CVPR_2017_paper.pdf)（CVPR 2017） | [Ziqi Wang 等：SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)（ICCV 2025）：continual instruction tuning and evaluation<br>[Ziqin Wang 等：CoPE: Continual Probe-guided Expansion for Large Vision-Language Models](https://proceedings.mlr.press/v306/wang26le.html)（ICML 2026）：continual instruction tuning and evaluation | candidate; final ten-task protocol not locked |
| GQA / 自然/日常场景 | [Drew A. Hudson 等：GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html)（CVPR 2019） | [Ziqi Wang 等：SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)（ICCV 2025）：continual instruction tuning and evaluation<br>[Ziqin Wang 等：CoPE: Continual Probe-guided Expansion for Large Vision-Language Models](https://proceedings.mlr.press/v306/wang26le.html)（ICML 2026）：continual instruction tuning and evaluation | candidate; final ten-task protocol not locked |
| TextVQA / 自然/日常场景 | [Amanpreet Singh 等：Towards VQA Models That Can Read](https://openaccess.thecvf.com/content_CVPR_2019/html/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html)（CVPR 2019） | [Ziqi Wang 等：SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)（ICCV 2025）：continual instruction tuning and evaluation<br>[Ziqin Wang 等：CoPE: Continual Probe-guided Expansion for Large Vision-Language Models](https://proceedings.mlr.press/v306/wang26le.html)（ICML 2026）：continual instruction tuning and evaluation | candidate; final ten-task protocol not locked |
| SQuAD11 / 自然/通用阅读 | [Pranav Rajpurkar 等：SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/)（EMNLP 2016） | [Jacob Devlin 等：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)（NAACL 2019）：fine-tuning and evaluation<br>[Zhiqing Sun 等：MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://aclanthology.org/2020.acl-main.195/)（ACL 2020）：fine-tuning and evaluation | candidate; final ten-task protocol not locked |
| MedQuAD / 医学 | [Asma Ben Abacha 等：A Question-Entailment Approach to Question Answering](https://pmc.ncbi.nlm.nih.gov/articles/PMC6805558/)（BMC Bioinformatics 2019） | [Jeonghun Cho 等：K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor](https://aclanthology.org/2025.naacl-long.351/)（NAACL 2025）：medical QA training and evaluation<br>[Artem Vazhentsev 等：Unconditional Truthfulness: Learning Unconditional Uncertainty of Large Language Models](https://aclanthology.org/2025.emnlp-main.1807/)（EMNLP 2025）：uncertainty estimation training/evaluation | candidate; final ten-task protocol not locked |
| VQA_RAD / 医学 | [Jason J. Lau 等：A dataset of clinically generated visual questions and answers about radiology images](https://pmc.ncbi.nlm.nih.gov/articles/PMC6244189/)（Scientific Data 2018） | [Xiao Liang 等：Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.pdf)（ICCV 2025）：medical VQA fine-tuning and reliability evaluation<br>[Meihong Pan 等：Dual-Level Confidence based Implicit Self-Refinement for Medical Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.html)（CVPR 2026）：medical VQA and cross-domain adaptation | candidate; final ten-task protocol not locked |
| PathVQA / 医学 | [Xuehai He 等：PathVQA: 30000+ Questions for Medical Visual Question Answering](https://arxiv.org/abs/2003.10286)（arXiv:2003.10286 2020） | [Xiao Liang 等：Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.pdf)（ICCV 2025）：medical VQA fine-tuning and reliability evaluation<br>[Meihong Pan 等：Dual-Level Confidence based Implicit Self-Refinement for Medical Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.html)（CVPR 2026）：medical VQA and cross-domain adaptation | candidate; final ten-task protocol not locked |
| SLAKE / 医学 | [Bo Liu 等：SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering](https://research.polyu.edu.hk/en/publications/slake-a-semantically-labeled-knowledge-enhanced-dataset-for-medic/)（ISBI 2021） | [Xiao Liang 等：Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.pdf)（ICCV 2025）：medical VQA fine-tuning and reliability evaluation<br>[Meihong Pan 等：Dual-Level Confidence based Implicit Self-Refinement for Medical Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.html)（CVPR 2026）：medical VQA and cross-domain adaptation | candidate; final ten-task protocol not locked |
| CUAD / 法律/合同 | [Dan Hendrycks 等：CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/6ea9ab1baa0efb9e19094440c317e21b-Abstract-round1.html)（NeurIPS Datasets and Benchmarks 2021） | [Santosh T.Y.S.S 等：CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](https://aclanthology.org/2025.acl-long.931/)（ACL 2025）：grounded legal answer generation<br>[Joongmin Shin 等：MultiDocFusion: Hierarchical and Multimodal Chunking Pipeline for Enhanced RAG on Long Industrial Documents](https://aclanthology.org/2025.emnlp-main.1062/)（EMNLP 2025）：layout/OCR-based document RAG evaluation | candidate; final ten-task protocol not locked |
| FinQA / 金融 | [Zhiyu Chen 等：FinQA: A Dataset of Numerical Reasoning over Financial Data](https://aclanthology.org/2021.emnlp-main.300/)（EMNLP 2021） | [Jiashuo Sun 等：APOLLO: An Optimized Training Approach for Long-form Numerical Reasoning](https://aclanthology.org/2024.lrec-main.122/)（LREC-COLING 2024）：retriever-generator training and numerical evaluation<br>[Xiao Li 等：DyRRen: A Dynamic Retriever-Reranker-Generator Model for Numerical Reasoning over Tabular and Textual Data](https://ojs.aaai.org/index.php/AAAI/article/view/26543)（AAAI 2023）：numerical QA training and evaluation | candidate; final ten-task protocol not locked |
| TAT_DQA / 金融 | [Fengbin Zhu 等：Towards Complex Document Understanding By Discrete Reasoning](https://arxiv.org/abs/2207.11871)（ACM Multimedia 2022） | [Fengbin Zhu 等：Doc2SoarGraph: Discrete Reasoning over Visually-Rich Table-Text Documents via Semantic-Oriented Hierarchical Graphs](https://aclanthology.org/2024.lrec-main.456/)（LREC-COLING 2024）：document QA training and evaluation<br>[Manuel Faysse 等：ColPali: Efficient Document Retrieval with Vision Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf)（ICLR 2025）：document retrieval evaluation, not QA generation | candidate; final ten-task protocol not locked |
| AQUA / 艺术与文化遗产 | [Noa Garcia 等：A Dataset and Baselines for Visual Question Answering on Art](https://github.com/noagarcia/ArtVQA)（ECCV VISART workshop 2020） | [Tibor Bleidt 等：ArtQuest: Countering Hidden Language Biases in ArtVQA](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)（WACV 2024）：ArtVQA evaluation and question-only bias analysis | fifth-domain candidate; QA files downloaded; image download in progress |
| SemArt / 艺术与文化遗产 | [Noa Garcia 等：How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Garcia_How_to_Read_Paintings_Semantic_Art_Understanding_with_Multi-Modal_Retrieval_ECCVW_2018_paper.pdf)（ECCV workshop 2018） | [Noa Garcia 等：A Dataset and Baselines for Visual Question Answering on Art](https://github.com/noagarcia/ArtVQA)（ECCV VISART workshop 2020）：source corpus for QA benchmark construction<br>[Tibor Bleidt 等：ArtQuest: Countering Hidden Language Biases in ArtVQA](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)（WACV 2024）：source corpus for art QA benchmark construction | supporting image/comment corpus, not a separate CL task |
| ChartQA / 辅助图表评测，不自动归金融 | [Ahmed Masry 等：ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning](https://aclanthology.org/2022.findings-acl.177/)（Findings of ACL 2022） | [Kenton Lee 等：Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://proceedings.mlr.press/v202/lee23g.html)（ICML 2023）：fine-tuning and QA evaluation<br>[Fangyu Liu 等：DePlot: One-shot visual language reasoning by plot-to-table translation](https://aclanthology.org/2023.findings-acl.660/)（Findings of ACL 2023）：plot-to-table and numerical QA evaluation | auxiliary; downloaded |
| DocVQA / 辅助文档评测，不自动归法律 | [Minesh Mathew 等：DocVQA: A Dataset for VQA on Document Images](https://openaccess.thecvf.com/content/WACV2021/html/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.html)（WACV 2021） | [Kenton Lee 等：Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://proceedings.mlr.press/v202/lee23g.html)（ICML 2023）：document QA fine-tuning/evaluation<br>[Geewook Kim 等：OCR-free Document Understanding Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880493.pdf)（ECCV 2022）：document QA fine-tuning/evaluation | auxiliary; downloaded |
| GeoChat_Instruct / 遥感，当前五域外备用 | [Kartik Kuckreja 等：GeoChat: Grounded Large Vision-Language Model for Remote Sensing](https://github.com/mbzuai-oryx/GeoChat)（CVPR 2024） | [Jeremy Irvin 等：TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ac3af725ae398b6184faae0828bdbd6c-Abstract-Conference.html)（ICLR 2025）：multimodal instruction training | backup raw assets; not in current five domains |

## 各数据集的证据与处理界限

### VQAv2

提出者：Yash Goyal、Tejas Khot、Douglas Summers-Stay、Dhruv Batra、Devi Parikh。原始论文：[Yash Goyal 等：Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering](https://openaccess.thecvf.com/content_cvpr_2017/papers/Goyal_Making_the_v_CVPR_2017_paper.pdf)（CVPR 2017）。

官方资源：https://visualqa.org/

划分：优先公开官方划分；hidden test 不造标签；开放式任务处理需记录与论文的差异。

限制与版本：目前本地 JSONL 为开放式派生视图，不等同于论文全部任务或原始完整计数；下载镜像不是提出者。

后续实际使用证据：

- [Ziqi Wang 等：SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)（ICCV 2025）。用途：continual instruction tuning and evaluation。定位：Main paper: upstream continual learning section; results table 1。证据概述：TextVQA, GQA and VQAv2 are sequential upstream tasks with per-dataset results; includes additional classification tasks that we do not adopt. [依据](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)
- [Ziqin Wang 等：CoPE: Continual Probe-guided Expansion for Large Vision-Language Models](https://proceedings.mlr.press/v306/wang26le.html)（ICML 2026）。用途：continual instruction tuning and evaluation。定位：Main paper: table 1 and task-order table 3。证据概述：CoIN results explicitly report TextVQA, GQA and VQAv2; not just references in the bibliography. [依据](https://proceedings.mlr.press/v306/wang26le.html)

### GQA

提出者：Drew A. Hudson、Christopher D. Manning。原始论文：[Drew A. Hudson 等：GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html)（CVPR 2019）。

官方资源：https://cs.stanford.edu/people/dorarad/gqa/

划分：优先公开官方划分；hidden test 不造标签；开放式任务处理需记录与论文的差异。

限制与版本：目前本地 JSONL 为开放式派生视图，不等同于论文全部任务或原始完整计数；下载镜像不是提出者。

后续实际使用证据：

- [Ziqi Wang 等：SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)（ICCV 2025）。用途：continual instruction tuning and evaluation。定位：Main paper: upstream continual learning section; results table 1。证据概述：TextVQA, GQA and VQAv2 are sequential upstream tasks with per-dataset results; includes additional classification tasks that we do not adopt. [依据](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)
- [Ziqin Wang 等：CoPE: Continual Probe-guided Expansion for Large Vision-Language Models](https://proceedings.mlr.press/v306/wang26le.html)（ICML 2026）。用途：continual instruction tuning and evaluation。定位：Main paper: table 1 and task-order table 3。证据概述：CoIN results explicitly report TextVQA, GQA and VQAv2; not just references in the bibliography. [依据](https://proceedings.mlr.press/v306/wang26le.html)

### TextVQA

提出者：Amanpreet Singh、Vivek Natarajan、Meet Shah、Yu Jiang、Xinlei Chen、Dhruv Batra、Devi Parikh、Marcus Rohrbach。原始论文：[Amanpreet Singh 等：Towards VQA Models That Can Read](https://openaccess.thecvf.com/content_CVPR_2019/html/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html)（CVPR 2019）。

官方资源：https://textvqa.org/

划分：优先公开官方划分；hidden test 不造标签；开放式任务处理需记录与论文的差异。

限制与版本：目前本地 JSONL 为开放式派生视图，不等同于论文全部任务或原始完整计数；下载镜像不是提出者。

后续实际使用证据：

- [Ziqi Wang 等：SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)（ICCV 2025）。用途：continual instruction tuning and evaluation。定位：Main paper: upstream continual learning section; results table 1。证据概述：TextVQA, GQA and VQAv2 are sequential upstream tasks with per-dataset results; includes additional classification tasks that we do not adopt. [依据](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)
- [Ziqin Wang 等：CoPE: Continual Probe-guided Expansion for Large Vision-Language Models](https://proceedings.mlr.press/v306/wang26le.html)（ICML 2026）。用途：continual instruction tuning and evaluation。定位：Main paper: table 1 and task-order table 3。证据概述：CoIN results explicitly report TextVQA, GQA and VQAv2; not just references in the bibliography. [依据](https://proceedings.mlr.press/v306/wang26le.html)

### SQuAD11

提出者：Pranav Rajpurkar、Jian Zhang、Konstantin Lopyrev、Percy Liang。原始论文：[Pranav Rajpurkar 等：SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/)（EMNLP 2016）。

官方资源：https://rajpurkar.github.io/SQuAD-explorer/

划分：官方 train/dev；公开 dev 只能固定承担一种开发或最终评测用途，不能同时调参和报最终成绩。

限制与版本：SQuAD 是广泛主题的 Wikipedia 阅读集，不是纯自然科学；此前人工 General/Geography 分域不作为当前正式方案。

后续实际使用证据：

- [Jacob Devlin 等：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)（NAACL 2019）。用途：fine-tuning and evaluation。定位：Abstract; SQuAD experiment subsection。证据概述：Explicit SQuAD v1.1 QA fine-tuning and test F1; also v2.0, which is a distinct version. [依据](https://aclanthology.org/N19-1423/)
- [Zhiqing Sun 等：MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://aclanthology.org/2020.acl-main.195/)（ACL 2020）。用途：fine-tuning and evaluation。定位：SQuAD v1.1 results table。证据概述：SQuAD v1.1 is used to evaluate the compressed model. [依据](https://aclanthology.org/2020.acl-main.195/)

### MedQuAD

提出者：Asma Ben Abacha、Dina Demner-Fushman。原始论文：[Asma Ben Abacha 等：A Question-Entailment Approach to Question Answering](https://pmc.ncbi.nlm.nih.gov/articles/PMC6805558/)（BMC Bioinformatics 2019）。

官方资源：https://github.com/abachaa/MedQuAD

划分：已按 K-COMP 随机 80/10/10 思路建立 seed 42 版本：13,125 / 1,641 / 1,641。论文种子与逐条 ID 未核实，不能声称精确相同；本项目 QA 行字节不变，清单单列。

限制与版本：原论文 collection 47,457 QA 不等于当前公开有答案数量；部分答案因版权移除，当前约 16,407 非空公开回答，不能补造。

后续实际使用证据：

- [Jeonghun Cho 等：K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor](https://aclanthology.org/2025.naacl-long.351/)（NAACL 2025）。用途：medical QA training and evaluation。定位：Appendix A.2。证据概述：MedQuAD has no provided split in this protocol; authors randomly split 80/10/10. [依据](https://aclanthology.org/2025.naacl-long.351/)
- [Artem Vazhentsev 等：Unconditional Truthfulness: Learning Unconditional Uncertainty of Large Language Models](https://aclanthology.org/2025.emnlp-main.1807/)（EMNLP 2025）。用途：uncertainty estimation training/evaluation。定位：Appendix dataset descriptions; tables 20/21。证据概述：MedQuAD is a long-answer QA dataset, with a 500 training / 1000 evaluation budget in this work. [依据](https://aclanthology.org/2025.emnlp-main.1807/)

### VQA_RAD

提出者：Jason J. Lau、Soumya Gayen、Asma Ben Abacha、Dina Demner-Fushman。原始论文：[Jason J. Lau 等：A dataset of clinically generated visual questions and answers about radiology images](https://pmc.ncbi.nlm.nih.gov/articles/PMC6244189/)（Scientific Data 2018）。

官方资源：https://osf.io/89kps/

划分：沿用明确公开的划分和 open-ended 标签；格式化镜像版本须单列。

限制与版本：医学开放式 QA 子集和 yes/no 子集不同；原论文计数、镜像计数、派生子集计数须分开。PathVQA 2020 introduction 的记录不伪称顶会发表。

后续实际使用证据：

- [Xiao Liang 等：Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.pdf)（ICCV 2025）。用途：medical VQA fine-tuning and reliability evaluation。定位：Experiments; section 5.2 figure 3; official repository datasets。证据概述：VQA-RAD, SLAKE and PathVQA are evaluated; open and closed questions have separate evaluation treatment. [依据](https://github.com/ecoxial2007/Expert-CFG)
- [Meihong Pan 等：Dual-Level Confidence based Implicit Self-Refinement for Medical Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.html)（CVPR 2026）。用途：medical VQA and cross-domain adaptation。定位：Section 4.1; open/closed per-dataset results table, proceedings p.17219。证据概述：All three datasets are evaluated. Standard split and cross-domain adaptation are distinct settings; pseudo-label adaptation is not our default evaluation protocol. [依据](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.pdf)

### PathVQA

提出者：Xuehai He、Yichen Zhang、Luntian Mou、Eric Xing、Pengtao Xie。原始论文：[Xuehai He 等：PathVQA: 30000+ Questions for Medical Visual Question Answering](https://arxiv.org/abs/2003.10286)（arXiv:2003.10286 2020）。

官方资源：https://github.com/UCSD-AI4H/PathVQA

划分：沿用明确公开的划分和 open-ended 标签；格式化镜像版本须单列。

限制与版本：医学开放式 QA 子集和 yes/no 子集不同；原论文计数、镜像计数、派生子集计数须分开。PathVQA 2020 introduction 的记录不伪称顶会发表。

后续实际使用证据：

- [Xiao Liang 等：Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.pdf)（ICCV 2025）。用途：medical VQA fine-tuning and reliability evaluation。定位：Experiments; section 5.2 figure 3; official repository datasets。证据概述：VQA-RAD, SLAKE and PathVQA are evaluated; open and closed questions have separate evaluation treatment. [依据](https://github.com/ecoxial2007/Expert-CFG)
- [Meihong Pan 等：Dual-Level Confidence based Implicit Self-Refinement for Medical Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.html)（CVPR 2026）。用途：medical VQA and cross-domain adaptation。定位：Section 4.1; open/closed per-dataset results table, proceedings p.17219。证据概述：All three datasets are evaluated. Standard split and cross-domain adaptation are distinct settings; pseudo-label adaptation is not our default evaluation protocol. [依据](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.pdf)

### SLAKE

提出者：Bo Liu、Li-Ming Zhan、Li Xu、Lin Ma、Yan Yang、Xiao-Ming Wu。原始论文：[Bo Liu 等：SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering](https://research.polyu.edu.hk/en/publications/slake-a-semantically-labeled-knowledge-enhanced-dataset-for-medic/)（ISBI 2021）。

官方资源：https://www.med-vqa.com/slake/

划分：沿用明确公开的划分和 open-ended 标签；格式化镜像版本须单列。

限制与版本：医学开放式 QA 子集和 yes/no 子集不同；原论文计数、镜像计数、派生子集计数须分开。PathVQA 2020 introduction 的记录不伪称顶会发表。

后续实际使用证据：

- [Xiao Liang 等：Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.pdf)（ICCV 2025）。用途：medical VQA fine-tuning and reliability evaluation。定位：Experiments; section 5.2 figure 3; official repository datasets。证据概述：VQA-RAD, SLAKE and PathVQA are evaluated; open and closed questions have separate evaluation treatment. [依据](https://github.com/ecoxial2007/Expert-CFG)
- [Meihong Pan 等：Dual-Level Confidence based Implicit Self-Refinement for Medical Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.html)（CVPR 2026）。用途：medical VQA and cross-domain adaptation。定位：Section 4.1; open/closed per-dataset results table, proceedings p.17219。证据概述：All three datasets are evaluated. Standard split and cross-domain adaptation are distinct settings; pseudo-label adaptation is not our default evaluation protocol. [依据](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.pdf)

### CUAD

提出者：Dan Hendrycks、Collin Burns、Anya Chen、Spencer Ball。原始论文：[Dan Hendrycks 等：CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/6ea9ab1baa0efb9e19094440c317e21b-Abstract-round1.html)（NeurIPS Datasets and Benchmarks 2021）。

官方资源：https://zenodo.org/records/4595826

划分：官方 train/test；图像连续训练任务仍需独立核验，不能把测试文档全部用于训练。

限制与版本：510 原始合同 PDF 已下载。MultiDocFusion 文档解析管线不等于已验证端到端 VLM 训练协议；文本与 PDF 使用同一合同需严格保持 split 一致。

后续实际使用证据：

- [Santosh T.Y.S.S 等：CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](https://aclanthology.org/2025.acl-long.931/)（ACL 2025）。用途：grounded legal answer generation。定位：Dataset description and CUAD results。证据概述：Extractive answer spans are concatenated for reference generation; multiple spans can be required components, not alternative answers. [依据](https://aclanthology.org/2025.acl-long.931/)
- [Joongmin Shin 等：MultiDocFusion: Hierarchical and Multimodal Chunking Pipeline for Enhanced RAG on Long Industrial Documents](https://aclanthology.org/2025.emnlp-main.1062/)（EMNLP 2025）。用途：layout/OCR-based document RAG evaluation。定位：Section 4; tables 2/3; appendix A.1。证据概述：CUAD official test contracts are used for retrieval and answer generation; parsed text chunks feed the final LLM. [依据](https://aclanthology.org/2025.emnlp-main.1062/)

### FinQA

提出者：Zhiyu Chen、Wenhu Chen、Charese Smiley、Sameena Shah、Iana Borova、Dylan Langdon、Reema Moussa、Matt Beane、Ting-Hao Huang、Bryan Routledge、William Yang Wang。原始论文：[Zhiyu Chen 等：FinQA: A Dataset of Numerical Reasoning over Financial Data](https://aclanthology.org/2021.emnlp-main.300/)（EMNLP 2021）。

官方资源：https://github.com/czyssrs/FinQA

划分：官方 train/dev/test，不自行重划分。

限制与版本：qa.answer 与 exe_ans 可能尺度不同；保留原始单位和程序标注，不能把 gold_inds 或参考程序放入问答输入。

后续实际使用证据：

- [Jiashuo Sun 等：APOLLO: An Optimized Training Approach for Long-form Numerical Reasoning](https://aclanthology.org/2024.lrec-main.122/)（LREC-COLING 2024）。用途：retriever-generator training and numerical evaluation。定位：Abstract; FinQA/ConvFinQA experiment results。证据概述：Evaluates FinQA and ConvFinQA leaderboard tasks; program/execution measures are task-specific. [依据](https://aclanthology.org/2024.lrec-main.122/)
- [Xiao Li 等：DyRRen: A Dynamic Retriever-Reranker-Generator Model for Numerical Reasoning over Tabular and Textual Data](https://ojs.aaai.org/index.php/AAAI/article/view/26543)（AAAI 2023）。用途：numerical QA training and evaluation。定位：AAAI abstract; official repository training/inference scripts。证据概述：FinQA is the actual training/evaluation dataset for dynamic retriever-reranker-generation. [依据](https://github.com/nju-websoft/dyrren)

### TAT_DQA

提出者：Fengbin Zhu、Wenqiang Lei、Fuli Feng、Chao Wang、Haozhou Zhang、Tat-Seng Chua。原始论文：[Fengbin Zhu 等：Towards Complex Document Understanding By Discrete Reasoning](https://arxiv.org/abs/2207.11871)（ACM Multimedia 2022）。

官方资源：https://github.com/NExTplusplus/TAT-DQA

划分：官方 train/dev/test_gold，保留 answer、scale 与页对应关系。

限制与版本：TAT-QA 纯文本表格任务与 TAT-DQA 文档图像任务不能混淆；原始数据 16,558 QA，派生视图计数另报。

后续实际使用证据：

- [Fengbin Zhu 等：Doc2SoarGraph: Discrete Reasoning over Visually-Rich Table-Text Documents via Semantic-Oriented Hierarchical Graphs](https://aclanthology.org/2024.lrec-main.456/)（LREC-COLING 2024）。用途：document QA training and evaluation。定位：Abstract; TAT-DQA experimental section/results。证据概述：Discrete reasoning on financial document QA with semantic-oriented hierarchical graphs. [依据](https://aclanthology.org/2024.lrec-main.456/)
- [Manuel Faysse 等：ColPali: Efficient Document Retrieval with Vision Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf)（ICLR 2025）。用途：document retrieval evaluation, not QA generation。定位：Table 1; appendix dataset descriptions。证据概述：ViDoRe contains 1600 TAT-DQA query/document pairs for retrieval. This supports reuse, not a generation-score comparison. [依据](https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf)

### AQUA

提出者：Noa Garcia、Chentao Ye、Zihua Liu、Qingtao Hu、Mayu Otani、Chenhui Chu、Yuta Nakashima、Teruko Mitamura。原始论文：[Noa Garcia 等：A Dataset and Baselines for Visual Question Answering on Art](https://github.com/noagarcia/ArtVQA)（ECCV VISART workshop 2020）。

官方资源：https://github.com/noagarcia/ArtVQA

划分：官方 train/val/test，knowledge 标记和原始 QA 保留。

限制与版本：原始工作是 ECCV workshop；已核实 1 篇独立后续 QA 评测，加原始论文共 2 篇实验工作。知识题不能未经协议核实就当成独立文本阅读任务；不能宣称所有视觉题必须看图。

后续实际使用证据：

- [Tibor Bleidt 等：ArtQuest: Countering Hidden Language Biases in ArtVQA](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)（WACV 2024）。用途：ArtVQA evaluation and question-only bias analysis。定位：Sections on visual and knowledge AQUA questions; comparison tables。证据概述：Evaluates existing AQUA rather than merely citing it; finds substantial language bias. [依据](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)

### SemArt

提出者：Noa Garcia、George Vogiatzis。原始论文：[Noa Garcia 等：How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Garcia_How_to_Read_Paintings_Semantic_Art_Understanding_with_Multi-Modal_Retrieval_ECCVW_2018_paper.pdf)（ECCV workshop 2018）。

官方资源：https://researchdata.aston.ac.uk/380/1/SemArt.zip

划分：沿用对应 AQUA/SemArt 官方划分关系。

限制与版本：SemArt 本身是图文检索/理解资源，不能单独冒充人工开放问答集；不采用 ArtQuest 固定属性识别来凑分类任务。

后续实际使用证据：

- [Noa Garcia 等：A Dataset and Baselines for Visual Question Answering on Art](https://github.com/noagarcia/ArtVQA)（ECCV VISART workshop 2020）。用途：source corpus for QA benchmark construction。定位：Official AQUA README; task definition。证据概述：Provides original paintings/comments underlying AQUA questions. [依据](https://github.com/noagarcia/ArtVQA)
- [Tibor Bleidt 等：ArtQuest: Countering Hidden Language Biases in ArtVQA](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)（WACV 2024）。用途：source corpus for art QA benchmark construction。定位：Section 5。证据概述：Uses SemArt paintings and structured attributes to construct ArtQuest. [依据](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)

### ChartQA

提出者：Ahmed Masry、Do Xuan Long、Jia Qing Tan、Shafiq Joty、Enamul Hoque。原始论文：[Ahmed Masry 等：ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning](https://aclanthology.org/2022.findings-acl.177/)（Findings of ACL 2022）。

官方资源：https://github.com/vis-nlp/ChartQA

划分：官方 train/val/test，human/augmented 分别报告。

限制与版本：图表形式不等于金融领域；目前仅辅助资源。

后续实际使用证据：

- [Kenton Lee 等：Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://proceedings.mlr.press/v202/lee23g.html)（ICML 2023）。用途：fine-tuning and QA evaluation。定位：Main results table; visual QA setup。证据概述：ChartQA is a downstream visual QA benchmark. [依据](https://proceedings.mlr.press/v202/lee23g.html)
- [Fangyu Liu 等：DePlot: One-shot visual language reasoning by plot-to-table translation](https://aclanthology.org/2023.findings-acl.660/)（Findings of ACL 2023）。用途：plot-to-table and numerical QA evaluation。定位：ChartQA human/augmented query results。证据概述：Distinguishes human-written and augmented ChartQA questions; original publication is Findings of ACL 2023, not ICML. [依据](https://aclanthology.org/2023.findings-acl.660/)

### DocVQA

提出者：Minesh Mathew、Dimosthenis Karatzas、C. V. Jawahar。原始论文：[Minesh Mathew 等：DocVQA: A Dataset for VQA on Document Images](https://openaccess.thecvf.com/content/WACV2021/html/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.html)（WACV 2021）。

官方资源：https://www.docvqa.org/

划分：官方可公开标注划分；hidden test 不造标签。

限制与版本：单页 DocVQA 与 Multi-page DocVQA 不是同一个实验；文档外观不能据此改成法律领域。

后续实际使用证据：

- [Kenton Lee 等：Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://proceedings.mlr.press/v202/lee23g.html)（ICML 2023）。用途：document QA fine-tuning/evaluation。定位：Main results table and official dataset preparation。证据概述：DocVQA is an actual downstream QA benchmark. [依据](https://proceedings.mlr.press/v202/lee23g.html)
- [Geewook Kim 等：OCR-free Document Understanding Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880493.pdf)（ECCV 2022）。用途：document QA fine-tuning/evaluation。定位：DocVQA experiment subsection/results。证据概述：OCR-free document understanding is evaluated on DocVQA. [依据](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880493.pdf)

### GeoChat_Instruct

提出者：Kartik Kuckreja、Muhammad Sohail Danish、Muzammal Naseer、Abhijit Das、Salman Khan、Fahad Shahbaz Khan。原始论文：[Kartik Kuckreja 等：GeoChat: Grounded Large Vision-Language Model for Remote Sensing](https://github.com/mbzuai-oryx/GeoChat)（CVPR 2024）。

官方资源：https://github.com/mbzuai-oryx/GeoChat

划分：当前不构建第五域任务；如以后使用须采用论文明确的任务协议。

限制与版本：原始混合集含分类和定位，不能整包算开放 QA；本次 pinned JSON 实际 308,861 条与 318k brochure 不同。

后续实际使用证据：

- [Jeremy Irvin 等：TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ac3af725ae398b6184faae0828bdbd6c-Abstract-Conference.html)（ICLR 2025）。用途：multimodal instruction training。定位：Section 3.1.1 and training-data appendix。证据概述：Explicitly incorporates GeoChat instructions in training. [依据](https://proceedings.iclr.cc/paper_files/paper/2025/file/ac3af725ae398b6184faae0828bdbd6c-Paper-Conference.pdf)

## 不纳入正式实验的资源

- SEC_Contracts_VQA：No verified original peer-reviewed introduction and multiple independent experimental-use evidence; automatic field labels are not used as the formal legal task.
- QASPER / SPIQA：User rejected the computer-science domain; downloads stopped; not adopted.
- SQuAD11_General / SQuAD11_Geography：Our former manual derived subsets are not standalone author-proposed benchmarks and are not the current formal protocol.

## 写论文时可以怎样表述

“我们选用已有公开问答资源构建五领域、文本与图像交替的持续学习序列，并分别引用数据集提出工作和后续实际评测工作。自然图像问答资源 VQAv2、GQA 和 TextVQA 已被 SMoLoRA（ICCV 2025）与 CoPE（ICML 2026）用于持续视觉指令实验；医学 VQA 资源 VQA-RAD、SLAKE 和 PathVQA 已被 Expert-CFG（ICCV 2025）和 DuCoR（CVPR 2026）用于医学问答评测。”

这段可描述选源依据，但文章正式实验设置必须另外补上实际采用的任务清单、版本、划分、开放式子集和计数。不能提前写“我们采用完全相同的划分”，也不能把备选/下载中资源写成已完成训练数据。艺术域的独立文本任务与法律图像训练协议尚未锁定。
