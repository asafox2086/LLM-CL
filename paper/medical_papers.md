# 医学 LLM 持续学习相关论文

以下文件统一保存在 `paper/`，下载于 2026-10-07。论文的过程展示与指标定义对照见 [评测阅读报告](evaluation_and_process.md)。

| 本地 PDF | 原始来源 |
|---|---|
| [MedCL-Bench (2026)](MedCL_Bench_2603.16738.pdf) | [原文](https://arxiv.org/abs/2603.16738) |
| [Replay-free Sequential Fine-tuning of Medical VLMs (ML4H 2025)](Medical_CL_ML4H_2025.pdf) | [原文](https://lihe50hz.github.io/ML4H_2025_Replay_free.pdf.pdf) |
| [MLLM-CL (2025)](MLLM_CL_2506.05453.pdf) | [原文](https://arxiv.org/abs/2506.05453) |
| [Countering Catastrophic Forgetting … via Weight-Space Model Merging (2026)](Medical_Model_Merging_2604.01538.pdf) | [原文](https://arxiv.org/abs/2604.01538) |
| [Medical foundation large language models for comprehensive text analysis and beyond (2025)](Me_LLaMA_2025.pdf) | [原文](https://www.nature.com/articles/s41746-025-01533-1) |

注意：MedCL-Bench 为标签分类；Medical-CL、MLLM-CL 为多模态；模型合并论文使用五个临床生成任务，但没有将五任务串为 CL 序列；Me-LLaMA 为持续预训练。

本轮选择模型合并论文中的 MTS-Dialog 和 IU X-ray 文本摘要，构建自己的两阶段医学续训协议。
