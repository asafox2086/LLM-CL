# MedQuAD 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Asma Ben Abacha、Dina Demner-Fushman。原始论文：[Asma Ben Abacha 等：A Question-Entailment Approach to Question Answering](https://pmc.ncbi.nlm.nih.gov/articles/PMC6805558/)（BMC Bioinformatics 2019）。

官方资源：https://github.com/abachaa/MedQuAD

划分：已按 K-COMP 随机 80/10/10 思路建立 seed 42 版本：13,125 / 1,641 / 1,641。论文种子与逐条 ID 未核实，不能声称精确相同；本项目 QA 行字节不变，清单单列。

限制与版本：原论文 collection 47,457 QA 不等于当前公开有答案数量；部分答案因版权移除，当前约 16,407 非空公开回答，不能补造。

后续实际使用证据：

- [Jeonghun Cho 等：K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor](https://aclanthology.org/2025.naacl-long.351/)（NAACL 2025）。用途：medical QA training and evaluation。定位：Appendix A.2。证据概述：MedQuAD has no provided split in this protocol; authors randomly split 80/10/10. [依据](https://aclanthology.org/2025.naacl-long.351/)
- [Artem Vazhentsev 等：Unconditional Truthfulness: Learning Unconditional Uncertainty of Large Language Models](https://aclanthology.org/2025.emnlp-main.1807/)（EMNLP 2025）。用途：uncertainty estimation training/evaluation。定位：Appendix dataset descriptions; tables 20/21。证据概述：MedQuAD is a long-answer QA dataset, with a 500 training / 1000 evaluation budget in this work. [依据](https://aclanthology.org/2025.emnlp-main.1807/)
