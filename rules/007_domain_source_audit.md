# 五领域来源核验与最新领域决定

2026-10-12。当前为准备阶段，尚未开始新实验；不能把下载原始数据等同于十任务全部训练就绪。

## 领域顺序

自然/日常场景 → 医学 → 法律 → 金融 → 艺术与文化遗产（候选核验中）。每领域先文本、后图像，共十个阶段。科学论文域已被用户否决，QASPER/SPIQA 下载进程已停止；不继续下载或纳入任务。

艺术域限定绘画、艺术史、艺术作品解读与文化藏品，不能混入日常自然照片、医学影像、合同或财报。不同领域完全无概念交集不现实，实际需要检验样本来源与任务语义，避免仅改名称。

## 第五域候选：AQUA / ArtpediaVQA

- [AQUA 作者仓库](https://github.com/noagarcia/ArtVQA)发布于 ECCV 2020 **VISART workshop**，不能标成 ECCV 主会。作者已有 train/val/test，69,812 / 5,124 / 4,912 QA；标注含原始 question、answer、image、need_external_knowledge。
- [官方数据说明](https://github.com/noagarcia/ArtVQA/blob/master/AQUA/README.md)明确区分艺术评论产生的知识问答和绘画视觉问答。原始 QA 不需要重新生成。
- [ArtQuest，WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf)实际评测 AQUA，并指出其语言偏置。因此多论文使用有依据，但不能据此宣称每个视觉问题必须看图；需要在实验中记录 question-only 对照。
- [ArtpediaVQA 官方仓库](https://github.com/miccunifi/ArtpediaVQA)提供人工视觉和上下文问答，并列出 ECCV 2022 相关文化遗产 VQA 工作；需进一步核验具体实验样本和可下载图像。

**尚未锁定文本/图像任务的具体来源。** AQUA 的知识题不自动等于独立文本阅读任务，ArtQuest 的艺术家/流派等固定属性也不直接满足用户排除朴素分类任务的要求。必须核实论文的文本输入协议和原始标注后再锁定。原始库自动构建的问题与我们新增合成问题是不同事项；不新增合成问题。

## 划分和真实性

优先官方 train/dev/test。没有官方划分则沿用已发表顶会论文的使用方式，保留比例、随机种子、源代码版本及未公开细节。例如 [K-COMP，NAACL 2025](https://aclanthology.org/2025.naacl-long.351/)采用 MedQuAD 的 80/10/10；未核实随机种子时不能声称精确复现。禁止把自己设计的全局 hash 划分说成论文原始划分。

法律图像仍须核验。[MultiDocFusion，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1062/)使用 CUAD 合同文档做布局/OCR/RAG，但最终文本 chunk 输入 LLM 的结果不能直接声称是端到端 VLM 图像问答的既有协议。官方 510 份合同 PDF 已下载，原始问答不变；SEC_Contracts_VQA 不纳入正式任务。

## 网络与资源

服务器 Clash 已关闭，后续 requests 使用 trust_env=False。下载走直连；直连失败不能自动退回代理。当前无科学论文下载进程，原先计划的 SPIQA 32 GB 图像包尚未开始传输。已下载 GeoChat 原始包保留为备用，不为更名而重新下载。所有大文件仅存服务器，不进入 GitHub 提交。
