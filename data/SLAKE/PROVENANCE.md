# SLAKE 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Bo Liu、Li-Ming Zhan、Li Xu、Lin Ma、Yan Yang、Xiao-Ming Wu。原始论文：[Bo Liu 等：SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering](https://research.polyu.edu.hk/en/publications/slake-a-semantically-labeled-knowledge-enhanced-dataset-for-medic/)（ISBI 2021）。

官方资源：https://www.med-vqa.com/slake/

划分：沿用明确公开的划分和 open-ended 标签；格式化镜像版本须单列。

限制与版本：医学开放式 QA 子集和 yes/no 子集不同；原论文计数、镜像计数、派生子集计数须分开。PathVQA 2020 introduction 的记录不伪称顶会发表。

后续实际使用证据：

- [Xiao Liang 等：Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.pdf)（ICCV 2025）。用途：medical VQA fine-tuning and reliability evaluation。定位：Experiments; section 5.2 figure 3; official repository datasets。证据概述：VQA-RAD, SLAKE and PathVQA are evaluated; open and closed questions have separate evaluation treatment. [依据](https://github.com/ecoxial2007/Expert-CFG)
- [Meihong Pan 等：Dual-Level Confidence based Implicit Self-Refinement for Medical Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.html)（CVPR 2026）。用途：medical VQA and cross-domain adaptation。定位：Section 4.1; open/closed per-dataset results table, proceedings p.17219。证据概述：All three datasets are evaluated. Standard split and cross-domain adaptation are distinct settings; pseudo-label adaptation is not our default evaluation protocol. [依据](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.pdf)
