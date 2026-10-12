# SQuAD11 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Pranav Rajpurkar、Jian Zhang、Konstantin Lopyrev、Percy Liang。原始论文：[Pranav Rajpurkar 等：SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/)（EMNLP 2016）。

官方资源：https://rajpurkar.github.io/SQuAD-explorer/

划分：官方 train/dev；公开 dev 只能固定承担一种开发或最终评测用途，不能同时调参和报最终成绩。

限制与版本：SQuAD 是广泛主题的 Wikipedia 阅读集，不是纯自然科学；此前人工 General/Geography 分域不作为当前正式方案。

后续实际使用证据：

- [Jacob Devlin 等：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)（NAACL 2019）。用途：fine-tuning and evaluation。定位：Abstract; SQuAD experiment subsection。证据概述：Explicit SQuAD v1.1 QA fine-tuning and test F1; also v2.0, which is a distinct version. [依据](https://aclanthology.org/N19-1423/)
- [Zhiqing Sun 等：MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://aclanthology.org/2020.acl-main.195/)（ACL 2020）。用途：fine-tuning and evaluation。定位：SQuAD v1.1 results table。证据概述：SQuAD v1.1 is used to evaluate the compressed model. [依据](https://aclanthology.org/2020.acl-main.195/)
