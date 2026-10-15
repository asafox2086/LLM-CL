# 2026-10-15：全量与1%原始构成实验

本条覆盖012的100/0/50预算及011对朴素控制的排除；旧轮停止并归档到exp/ago、summary/ago。

1. Qwen2-VL-2B，seed42，一轮，无验证、无checkpoint。五领域、文本后图像共十任务。
2. 全量组GPU3；1%训练组GPU1/2；GPU0保留。不得干预其他用户任务。
3. 全量使用明确原始配置的完整训练／公开有标注评测池。1%只抽训练，测试与全量组逐字节相同。原始分类、是非、选择、定位等题型全部保留，禁止沿用旧open_ended过滤视图。
4. PathVQA退出，使用NeurIPS2023 Datasets and Benchmarks正式发表的LLaVA-Med原始训练／评测；GEMeX因缺MIMIC授权不采用。
5. 八篇方法加标准SeqFT-LoRA控制，共九方法，两个条件都执行。不能改名冒充论文机制。
6. 所有方法、两个条件都必须记录CPU和GPU训练前／后具体内存字节值、增长与峰值、逐阶段变化、loss与裁剪前后实际梯度。结果缺失须明确，不能估填。
7. 原始问答／选项／答案／图像不改写；同数据集只列一个名字并标注模态。官方划分优先；研究划分和公开validation代作测试须注明。

本轮路径：exp/ten_task_cl_20261015_full_1pct；summary同名；数据data/TenTaskCL同名。详细可执行协议在该exp/README.md与execution_policy.json。
