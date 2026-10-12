# MedQuAD 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Asma Ben Abacha、Dina Demner-Fushman。原始论文：[Asma Ben Abacha 等：A Question-Entailment Approach to Question Answering](https://pmc.ncbi.nlm.nih.gov/articles/PMC6805558/)（BMC Bioinformatics 2019）。

官方资源：https://github.com/abachaa/MedQuAD

划分：可参考 K-COMP 的 80/10/10；随机种子未核实，不能声称逐条精确复现。

限制与版本：原论文 collection 47,457 QA 不等于当前公开有答案数量；部分答案因版权移除，当前约 16,407 非空公开回答，不能补造。

后续实际使用证据：

- [Jeonghun Cho 等：K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor](https://aclanthology.org/2025.naacl-long.351/)（NAACL 2025）。用途：medical QA training and evaluation。定位：Appendix A.2。证据概述：MedQuAD has no provided split in this protocol; authors randomly split 80/10/10. [依据](https://aclanthology.org/2025.naacl-long.351/)
- [Artem Vazhentsev 等：Unconditional Truthfulness: Learning Unconditional Uncertainty of Large Language Models](https://aclanthology.org/2025.emnlp-main.1807/)（EMNLP 2025）。用途：uncertainty estimation training/evaluation。定位：Appendix dataset descriptions; tables 20/21。证据概述：MedQuAD is a long-answer QA dataset, with a 500 training / 1000 evaluation budget in this work. [依据](https://aclanthology.org/2025.emnlp-main.1807/)
