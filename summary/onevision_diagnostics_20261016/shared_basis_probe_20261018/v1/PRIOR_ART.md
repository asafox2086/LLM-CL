# 共享表示轻量化：已有工作与本次边界

2026-10-18核验原始来源，不作为新增基线或本模型F1结果。

| 原始来源 | 可借鉴点 | 不能直接推出 |
| --- | --- | --- |
| [EigenLoRAx，CVPR Workshops 2025](https://openaccess.thecvf.com/content/CVPR2025W/TCV/html/Kaushik_EigenLoRAx_Recycling_Adapters_to_Find_Principal_Subspaces_for_Resource-Efficient_Adaptation_CVPRW_2025_paper.html) | 回收已有适配器构建共享主子空间，以较小系数适配；论文也考虑补充正交方向。 | 它是已发表workshop，不能写成CVPR主会。当前探针不是其完整复现，也未证明小数据多领域图文场景有效。 |
| [LoRA-XS，ECAI 2025论文集](https://journals.sagepub.com/doi/10.3233/FAIA251185)（出版社2026-08-25上线；[2024预印本](https://arxiv.org/abs/2405.17604)） | 由预训练权重SVD固定两侧矩阵，只训练中间小矩阵R，减少适配参数。 | 其预训练权重子空间不是我们的三个已学功能；不能把参数减少、作者语言任务成绩换成本模型F1或常驻内存优势。 |

本次只做权重可行性：从实际`deltaW=4BA`而非A/B元素相似度构造B加权联合右子空间，固定rank8候选及rank24数值控制，保存共享A与状态各自B。线性代数压缩不是我们的新理论。

当前rank8权重载荷由6.1875降至3.5625MiB，但三个状态函数重构误差约4.95%／1.45%／1.62%，状态差分误差约6.54%／6.59%／9.18%。只按整体相似度或能量不能宣称新方向／F1无损；实际部署CPU/GPU和F1均尚未测。原匹配试验源码／队列不变，本压缩没有准入GPU。

[实际数值及CPU快照](EVIDENCE.json) · [六栏探究记录](RESEARCH_FINDINGS.md)
