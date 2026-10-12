# SemArt 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Noa Garcia、George Vogiatzis。原始论文：[Noa Garcia 等：How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Garcia_How_to_Read_Paintings_Semantic_Art_Understanding_with_Multi-Modal_Retrieval_ECCVW_2018_paper.pdf)（ECCV workshop 2018）。

官方资源：https://researchdata.aston.ac.uk/380/1/SemArt.zip

划分：沿用对应 AQUA/SemArt 官方划分关系。

限制与版本：SemArt 本身是图文检索/理解资源，不能单独冒充人工开放问答集；不采用 ArtQuest 固定属性识别来凑分类任务。

后续实际使用证据：

- [Noa Garcia 等：A Dataset and Baselines for Visual Question Answering on Art](https://github.com/noagarcia/ArtVQA)（ECCV VISART workshop 2020）。用途：source corpus for QA benchmark construction。定位：Official AQUA README; task definition。证据概述：Provides original paintings/comments underlying AQUA questions. [依据](https://github.com/noagarcia/ArtVQA)
- [Tibor Bleidt 等：ArtQuest: Countering Hidden Language Biases in ArtVQA](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)（WACV 2024）。用途：source corpus for art QA benchmark construction。定位：Section 5。证据概述：Uses SemArt paintings and structured attributes to construct ArtQuest. [依据](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)
