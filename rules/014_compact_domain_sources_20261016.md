# 2026-10-16：单来源几万规模与多领域文献依据

覆盖 013 的大规模来源选择，其他要求保留：Qwen2-VL-2B、seed42、一轮、无验证集、不存 checkpoint；GPU0 保留、GPU3 全量、GPU1/2 为 1% 训练，两组完整同测试；九方法和内存、loss、梯度记录均不变。

- T2 仅使用 TextVQA，不再叠加 VQAv2／GQA 大池。
- T4 使用原 VQA-RAD 镜像和 SLAKE，退出 LLaVA-Med；不因图像 URL 失效而悄悄挑取它的小子集。
- T6 使用 RSVQA-LR 原始发布，退出 GeoChat。作者 JSON 的 active 掩码指定 train 57,223／val 10,005／test 10,004；本轮不使用 val。保留原题型和数字答案，不照搬其他论文的计数分箱或题型排除。
- 其余 T1／3／5／7／8／9／10 来源保持。单任务完整训练 QA 必须小于 100,000，脚本断言防止误回大池。十任务合计仍为 281,958 训练／37,998 测试；1% 训练 2,825，不把单任务规模与总规模混淆。
- 本轮 split_version 为 `ten_task_cl_20261015_full_1pct_compact_20261016_v1`；不能读取旧 native_v1 预测／清单。旧大规模方案没有正式训练结果，元数据保存在 summary/ago/dataset_protocol_20261015_large_sources。
- 研究证据见 data/MULTIDOMAIN_DATASET_SURVEY_20261016.md。逐项记录使用论文、发表版本、正文位置和监督训练／仅评测用途。多领域与多任务、原始发表与后续顶会复用、QA 与图像／会话计数必须区分。
- 医学来源原始发表如实写为 Scientific Data 2018 和 ISBI 2021，不因 ICCV 2025 后续使用而改写其发表 venue。未证明所有五域十任务来自同一个现成 benchmark。
