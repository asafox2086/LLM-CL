# 当前修订：每领域一个图文来源，保存最终参数

2026-10-16 用户要求减少数据集数量，优先每个领域一个图文来源。原生文本问题必须有足够上下文，不能直接删除图像构造不可回答的“文本任务”。用户已确认五个图文任务；本十任务目录停用，当前入口为 exp/five_domain_cl_20261016_full_1pct。旧任务清单不重用。

候选：自然 TextVQA；医学 SLAKE；遥感 RSVQA-LR；金融 MME-Finance；艺术 AQUA。候选来源和实际使用论文见 ../../data/MULTIDOMAIN_DATASET_SURVEY_20261016.md。是否具备独立文本视图需要逐项判断，不能把“有文字问题”当“有文本任务”。

最新保存规则覆盖此前无最终权重要求：不存阶段／优化器 checkpoint，每个正式方法完成全部任务后，保存 runs/<condition>/<method>/<run_id>/final_parameters/，含完整 model_state.pt、method_state.pt、模型与处理器配置、任务清单、运行代码与来源校验。包含冻结旧专家和合并后的基座权重。最后训练后的内存记录在导出前完成；导出大小和 SHA256 写 artifact_manifest.json。不是原生 Transformers 目录，必须重建自定义方法架构，正式后续评定前完成新进程重载等价检验。

以下十任务方案是本次合并前的记录，不能直接作为最新任务数启动。

---

# 原始数据全量与 1% 持续学习实验

2026-10-15 用户新要求覆盖此前 100/0/50 预算与排除 SeqLoRA 控制的限制。旧实验已停止并归档到 `exp/ago/ten_task_cl_20261014_geo_fin` 与 `summary/ago/ten_task_cl_20261014_geo_fin`，不再追加结果。

基座 Qwen2-VL-2B-Instruct；五领域按自然／通用、医学、地理／遥感、金融、艺术文化排列，每领域文本后图像，共十任务。seed42；每任务一轮；无验证集；无模型、优化器或恢复 checkpoint。

## 两组与调度

| 条件 | 训练 | 测试 | GPU |
|---|---|---|---|
| full | 所选原始发布配置的完整训练池 | 完整有标注公开评测池 | 3，方法串行 |
| pct01 | 相同训练池的 1% | 与 full 逐字节相同的完整测试池 | 1、2，方法并行 |

GPU0 保留，不安排任务，不停止其他用户进程。不是把训练和测试合并用于训练。某些基准隐藏测试答案，因此使用其完整公开 dev／validation／testdev；不冒称官方隐藏 test 结果。

所有分类、是非、选择、定位、开放回答等原始题型均保留；选项、原始问答、图像不改写。不能读取旧 `open_ended` 题型过滤版本生成本轮数据。仅不能取得公开金标准答案的记录无法训练／评分，必须列数量和原因。官方配置本身的边界（例如 GQA balanced、MME-Finance 英文 MMfin）明确记录，不能把它冒称所有语言／版本的完整集合。

官方分割优先。无官方分割的 MedQuAD 按原始文档、GeoQuestions1089 按原始查询组、GeoChat 按图像、MME-Finance 按底层图表组划分 80%／20% 的组。确定性 SHA256 排序、seed42；所有可用记录归入一个角色，无额外保留验证池。该规则是本研究划分，不是作者官方划分；不同大小组会让 QA 比例略偏离 80/20。

1% 在各来源数据集内取 ceil(0.01×原训练数)，按原始题型比例用最大余数分配，再按固定 SHA256 选 ID；稀有题型可能因取整在 1% 中为零，不是人工排除。full 与 pct01 测试文件 SHA 必须相同；所有方法训练顺序按 `SHA256(42:stage:QA_ID)` 排列。

## 数据集表：同名只列一次

| 数据集 | 模态／任务 | 原始来源／说明 |
|---|---|---|
| SQuAD 1.1 | 文本，T1 | EMNLP 2016；train / dev |
| VQAv2 | 多模态，T2 | CVPR 2017；全部原始题型；train / validation |
| GQA balanced | 多模态，T2 | CVPR 2019；train / 有答案的 testdev |
| TextVQA | 多模态，T2 | CVPR 2019；train / validation |
| MedQuAD | 文本，T3 | BMC Bioinformatics 2019；原始 XML，公开答案缺失情况单列 |
| VQA-RAD | 多模态，T4 | Scientific Data 2018；原始 train / test，不删 yes/no |
| SLAKE | 多模态，T4 | ISBI 2021；原始 train / test，保留 OPEN/CLOSED 和两种语言 |
| LLaVA-Med | 多模态，T4 | 替代 PathVQA；NeurIPS 2023 Datasets and Benchmarks；作者 instruct60k 和独立 qa50 评测，原始多轮 QA，非生成新标注 |
| GeoQuestions1089 | 文本，T5 | ISWC 2023；全部 1089 问题类型及原始结果 CSV，不只取数值题 |
| GeoChat | 多模态，T6 | CVPR 2024；原始指令配置全部题型及原始回答，包括分类／定位 |
| FinQA | 文本与原表格，T7 | EMNLP 2021；原始 train / test，包括 yes/no |
| MME-Finance | 多模态，T8 | ACM MM 2025；原始英文 MMfin 全部任务、样式、图像种类；不继续限 PC 图表 |
| AQUA | 文本 T9、多模态 T10 | ECCV VISART Workshop 2020；按作者 need_external_knowledge 标记区分两个任务，官方 train / test，同一名字不重复列 |

LLaVA-Med 原始发表：[NeurIPS 论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/5abcdf8ecdcacba028c6662789194572-Paper-Datasets_and_Benchmarks.pdf)、[作者数据与评测](https://github.com/microsoft/LLaVA-Med)。独立复用示例：[STLLaVA-Med，NeurIPS 2024 Datasets and Benchmarks](https://papers.neurips.cc/paper_files/paper/2024/file/aec33ab89b5986605cd7c331396e7e5c-Paper-Datasets_and_Benchmarks_Track.pdf)，其中使用作者 50 图像／193 QA 的评测。原训练／参考回答含作者 GPT-4 生成标注，不能称全人工临床标签。

## 方法与最简单控制

九种：SeqFT-LoRA 控制，加 SMoLoRA、PASs-MoE、Hyper-LLaVA、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER。两组同样跑全部九种。

**SeqFT-LoRA** 是标准朴素顺序微调：冻结视觉和语言基座，对 q/v 使用同一组 LoRA（r8、alpha32、dropout0.1），按 T1→T10 继续更新；没有回放、保护约束、额外专家或模型平均，不把它当作新提出的方法。原论文基线依据：[SMoLoRA 的 SeqLoRA 对照](https://arxiv.org/html/2411.13949)。作者机制适配继承旧代码并取消固定100条断言；不把普通 LoRA 改名成论文方法。

PaRSP 的作者校准设置上限256，只从当前训练池固定取 min(256,N) 做归因／协方差，正式训练仍用全部 N。FV 保留100均值试验／25因果试验；当前 1% 池不足11条时演示数取 min(10,N−1)，不读全量组补数据、不重复填样本。各辅助调用及实际ID另记；两组辅助来源均限制当前条件的训练池。

## 所有方法、两组共同强制记录

- 每个更新：总训练 loss、裁剪前／后实际梯度 L2 范数、裁剪阈值、学习率、样本数、输入／输出 token、耗时和显存；辅助计算沿用方法日志。
- `resources.jsonl`：每5秒及阶段边界记录 CPU RSS/PSS/USS、交换量，以及 GPU allocated/reserved/峰值和 NVML 进程占用，单位为字节，不以整机已用内存替代进程内存。
- `memory_summary.json`：模型和方法加载后的训练前、最后阶段训练后释放优化器和梯度后的具体占用，以及差值与观测峰值。两个条件、所有方法均保留，不只展示峰值。
- 每阶段另外保留：辅助准备前后、训练结束含优化器、释放优化器后的持久状态、评测后的占用。这样区分计算临时开销、缓存与跨任务持久增长。PSS/USS取 `/proc/self/smaps_rollup`；NVML 缺失记 null，不伪造0。
- 训练前和每训练完一个任务完整评测全部十任务；保存逐题预测／参考及三种分数、完整11×10矩阵、AP／ForgetRate／FWT／BWT。统一词面分数不冒称每个原始任务的官方语义、定位或数值指标。

脚本：`prepare_assets.py`（公开医学资源直连）、`prepare_more_assets.py`（GeoChat／SemArt原图）、`prepare_data.py`（原始构成、全量与1%冻结）、`run.py`、`resource_monitor.py`、`pipeline.py`。

资产未齐全、两组测试 SHA 不一致、训练／测试 ID 重叠、内存记录或新图像反传预检失败时，不发布正式结果。下载失败不会通过静默删除样本处理。队列进度在 `summary/ten_task_cl_20261015_full_1pct/progress.md`。
