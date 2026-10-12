# ChartQA 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Ahmed Masry、Do Xuan Long、Jia Qing Tan、Shafiq Joty、Enamul Hoque。原始论文：[Ahmed Masry 等：ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning](https://aclanthology.org/2022.findings-acl.177/)（Findings of ACL 2022）。

官方资源：https://github.com/vis-nlp/ChartQA

划分：官方 train/val/test，human/augmented 分别报告。

限制与版本：图表形式不等于金融领域；目前仅辅助资源。

后续实际使用证据：

- [Kenton Lee 等：Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://proceedings.mlr.press/v202/lee23g.html)（ICML 2023）。用途：fine-tuning and QA evaluation。定位：Main results table; visual QA setup。证据概述：ChartQA is a downstream visual QA benchmark. [依据](https://proceedings.mlr.press/v202/lee23g.html)
- [Fangyu Liu 等：DePlot: One-shot visual language reasoning by plot-to-table translation](https://aclanthology.org/2023.findings-acl.660/)（Findings of ACL 2023）。用途：plot-to-table and numerical QA evaluation。定位：ChartQA human/augmented query results。证据概述：Distinguishes human-written and augmented ChartQA questions; original publication is Findings of ACL 2023, not ICML. [依据](https://aclanthology.org/2023.findings-acl.660/)
