# AQUA 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Noa Garcia、Chentao Ye、Zihua Liu、Qingtao Hu、Mayu Otani、Chenhui Chu、Yuta Nakashima、Teruko Mitamura。原始论文：[Noa Garcia 等：A Dataset and Baselines for Visual Question Answering on Art](https://github.com/noagarcia/ArtVQA)（ECCV VISART workshop 2020）。

官方资源：https://github.com/noagarcia/ArtVQA

划分：官方 train/val/test，knowledge 标记和原始 QA 保留。

限制与版本：原始工作是 ECCV workshop；已核实 1 篇独立后续 QA 评测，加原始论文共 2 篇实验工作。知识题不能未经协议核实就当成独立文本阅读任务；不能宣称所有视觉题必须看图。

后续实际使用证据：

- [Tibor Bleidt 等：ArtQuest: Countering Hidden Language Biases in ArtVQA](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)（WACV 2024）。用途：ArtVQA evaluation and question-only bias analysis。定位：Sections on visual and knowledge AQUA questions; comparison tables。证据概述：Evaluates existing AQUA rather than merely citing it; finds substantial language bias. [依据](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)
