# FinQA 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Zhiyu Chen、Wenhu Chen、Charese Smiley、Sameena Shah、Iana Borova、Dylan Langdon、Reema Moussa、Matt Beane、Ting-Hao Huang、Bryan Routledge、William Yang Wang。原始论文：[Zhiyu Chen 等：FinQA: A Dataset of Numerical Reasoning over Financial Data](https://aclanthology.org/2021.emnlp-main.300/)（EMNLP 2021）。

官方资源：https://github.com/czyssrs/FinQA

划分：官方 train/dev/test，不自行重划分。

限制与版本：qa.answer 与 exe_ans 可能尺度不同；保留原始单位和程序标注，不能把 gold_inds 或参考程序放入问答输入。

后续实际使用证据：

- [Jiashuo Sun 等：APOLLO: An Optimized Training Approach for Long-form Numerical Reasoning](https://aclanthology.org/2024.lrec-main.122/)（LREC-COLING 2024）。用途：retriever-generator training and numerical evaluation。定位：Abstract; FinQA/ConvFinQA experiment results。证据概述：Evaluates FinQA and ConvFinQA leaderboard tasks; program/execution measures are task-specific. [依据](https://aclanthology.org/2024.lrec-main.122/)
- [Xiao Li 等：DyRRen: A Dynamic Retriever-Reranker-Generator Model for Numerical Reasoning over Tabular and Textual Data](https://ojs.aaai.org/index.php/AAAI/article/view/26543)（AAAI 2023）。用途：numerical QA training and evaluation。定位：AAAI abstract; official repository training/inference scripts。证据概述：FinQA is the actual training/evaluation dataset for dynamic retriever-reranker-generation. [依据](https://github.com/nju-websoft/dyrren)
