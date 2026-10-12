# DocVQA 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Minesh Mathew、Dimosthenis Karatzas、C. V. Jawahar。原始论文：[Minesh Mathew 等：DocVQA: A Dataset for VQA on Document Images](https://openaccess.thecvf.com/content/WACV2021/html/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.html)（WACV 2021）。

官方资源：https://www.docvqa.org/

划分：官方可公开标注划分；hidden test 不造标签。

限制与版本：单页 DocVQA 与 Multi-page DocVQA 不是同一个实验；文档外观不能据此改成法律领域。

后续实际使用证据：

- [Kenton Lee 等：Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://proceedings.mlr.press/v202/lee23g.html)（ICML 2023）。用途：document QA fine-tuning/evaluation。定位：Main results table and official dataset preparation。证据概述：DocVQA is an actual downstream QA benchmark. [依据](https://proceedings.mlr.press/v202/lee23g.html)
- [Geewook Kim 等：OCR-free Document Understanding Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880493.pdf)（ECCV 2022）。用途：document QA fine-tuning/evaluation。定位：DocVQA experiment subsection/results。证据概述：OCR-free document understanding is evaluated on DocVQA. [依据](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880493.pdf)
