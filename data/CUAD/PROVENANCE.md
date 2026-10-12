# CUAD 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Dan Hendrycks、Collin Burns、Anya Chen、Spencer Ball。原始论文：[Dan Hendrycks 等：CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/6ea9ab1baa0efb9e19094440c317e21b-Abstract-round1.html)（NeurIPS Datasets and Benchmarks 2021）。

官方资源：https://zenodo.org/records/4595826

划分：官方 train/test；图像连续训练任务仍需独立核验，不能把测试文档全部用于训练。

限制与版本：510 原始合同 PDF 已下载。MultiDocFusion 文档解析管线不等于已验证端到端 VLM 训练协议；文本与 PDF 使用同一合同需严格保持 split 一致。

后续实际使用证据：

- [Santosh T.Y.S.S 等：CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](https://aclanthology.org/2025.acl-long.931/)（ACL 2025）。用途：grounded legal answer generation。定位：Dataset description and CUAD results。证据概述：Extractive answer spans are concatenated for reference generation; multiple spans can be required components, not alternative answers. [依据](https://aclanthology.org/2025.acl-long.931/)
- [Joongmin Shin 等：MultiDocFusion: Hierarchical and Multimodal Chunking Pipeline for Enhanced RAG on Long Industrial Documents](https://aclanthology.org/2025.emnlp-main.1062/)（EMNLP 2025）。用途：layout/OCR-based document RAG evaluation。定位：Section 4; tables 2/3; appendix A.1。证据概述：CUAD official test contracts are used for retrieval and answer generation; parsed text chunks feed the final LLM. [依据](https://aclanthology.org/2025.emnlp-main.1062/)
