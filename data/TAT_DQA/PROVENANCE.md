# TAT_DQA 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Fengbin Zhu、Wenqiang Lei、Fuli Feng、Chao Wang、Haozhou Zhang、Tat-Seng Chua。原始论文：[Fengbin Zhu 等：Towards Complex Document Understanding By Discrete Reasoning](https://arxiv.org/abs/2207.11871)（ACM Multimedia 2022）。

官方资源：https://github.com/NExTplusplus/TAT-DQA

划分：官方 train/dev/test_gold，保留 answer、scale 与页对应关系。

限制与版本：TAT-QA 纯文本表格任务与 TAT-DQA 文档图像任务不能混淆；原始数据 16,558 QA，派生视图计数另报。

后续实际使用证据：

- [Fengbin Zhu 等：Doc2SoarGraph: Discrete Reasoning over Visually-Rich Table-Text Documents via Semantic-Oriented Hierarchical Graphs](https://aclanthology.org/2024.lrec-main.456/)（LREC-COLING 2024）。用途：document QA training and evaluation。定位：Abstract; TAT-DQA experimental section/results。证据概述：Discrete reasoning on financial document QA with semantic-oriented hierarchical graphs. [依据](https://aclanthology.org/2024.lrec-main.456/)
- [Manuel Faysse 等：ColPali: Efficient Document Retrieval with Vision Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf)（ICLR 2025）。用途：document retrieval evaluation, not QA generation。定位：Table 1; appendix dataset descriptions。证据概述：ViDoRe contains 1600 TAT-DQA query/document pairs for retrieval. This supports reuse, not a generation-score comparison. [依据](https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf)
