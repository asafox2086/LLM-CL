# 002：T5-Large 统一七任务试跑

2026-10-07，用户授权先用 T5-Large 试跑。本文件覆盖 001 协议中的基座、精度、评价 batch 和模型架构适配条款；001 的数据划分、训练预算、评分和防泄漏约束继续有效。数据协议标识仍为 `superni_generation7_v2`，新的 run label 含 `t5large`，与 7B、未启动的 Phi-2 配置分开。

## 论文中的 T5 任务

- O-LoRA：标准文本分类持续学习，以及 15 任务长序列。数据包括 AG News、Amazon、Yelp、DBpedia、Yahoo，以及 GLUE、SuperGLUE 和 IMDB；不同短序列设置采用其中的子集和顺序。见 [§4.1、附录 A](https://aclanthology.org/2023.findings-emnlp.715/)。
- Progressive Prompts：T5 短序列使用 AG News、Amazon、Yahoo、DBpedia；长序列使用五个分类数据集加 GLUE 的 MNLI/QQP/RTE/SST-2、SuperGLUE 的 WiC/CB/COPA/MultiRC/BoolQ，以及 IMDB，共 15 个任务。见 [§4.1](https://arxiv.org/abs/2301.12314)。
- MIGU：T5-Large 使用四任务标准分类序列，以及 15 任务长序列；另有 RoBERTa 和 Llama2 实验，不能混作 T5 结果。见 [§4.1](https://arxiv.org/abs/2406.17245)。
- SAPT：SuperNI 中对话、信息抽取、问答、摘要、情感分析各选三个任务，合计 15 个；另跑 15 个分类任务的 Long Sequence Benchmark。分类看 Accuracy，其余看 ROUGE-L。见 [§5.1](https://aclanthology.org/2024.acl-long.625/)。
- 仓库综述的 §IV-E 和表 IV 统一比较使用 T5-Large，在 LSB 与 SuperNI 上各取两个任务顺序。见 [综述](https://arxiv.org/html/2603.12658v2)。

## 本轮配置

原版 `google-t5/t5-large`，revision `150ebc2c4b72291e770f58e6057481c8d2ed331a`，不替换成 FLAN-T5。四个方法分别为 O-LoRA、MIGU-LoRA、SAPT-LoRA、SeqLoRA，独立基座实例和方法状态。保持原七任务生成集合，不切换到上述分类数据集，也不宣称复现论文的 15 任务表格。

七任务仍为 XSUM → Quoref → EVALution → PersonaChat → GLUCOSE → Reddit TIFU → SciQ，每任务 Train/Dev/Test=1000/200/500。仍先测基座，再每学一任务测全部七任务，产生 8×7 ROUGE-L 矩阵；逐条保存 Exact Match、Token F1 和完整回答。

FP32 GPU、冻结基座、无 AMP/量化；LoRA 为全部 attention 的 q/v（包括 cross-attention），rank 8、alpha 32、dropout 0.1。学习率 1e-4，AdamW，训练 batch=16、micro batch=1、1 epoch，warmup 3%，每任务 63 更新；Dev 按 ROUGE-L 选模。encoder prompt≤768（包含 EOS），decoder target≤256（包含 EOS），greedy eval batch=8、max_new_tokens=256。

Checkpoint、原子写入、逐 batch 预测恢复、每步优化器/RNG 恢复和完整阶段恢复均沿用原机制。旧 7B 结果保留，但不能与 T5 行合并计算 AP/FWT/BWT。
