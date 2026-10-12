# 本轮执行协议：等量十任务，单 seed 42

2026-10-12 用户确认：每任务等量，先只用 seed 42；采样参考论文、规模适中；清理旧脚本并由脚本启动。本文件覆盖 009 的数据预算、多种子计划及准备状态。

## 固定预算与来源

全部 T1–T10：**train 2,000 / dev 100 / test 200，一轮，seed 42**。没有复制样本补齐。十阶段每个方法一共曝光 20,000 条当前任务训练 QA；方法原有历史样本及辅助计算另报。全部方法使用同一 ID manifest，不能按方法重新抽样。

| T | 领域 / 模态 | 数据与配额 | 划分和输入依据 |
|---|---|---|---|
| T1 | 自然 / 文本 | SQuAD 1.1：2000/100/200 | 官方 train 取训练；train 内按 passage 固定留出开发候选；官方 dev 用作固定测试。问题、参考答案和 passage 不改。 |
| T2 | 自然 / 图像 | VQAv2、GQA、TextVQA；train 800/800/400，dev 40/40/20，test 80/80/40 | 数据组合参考 SMoLoRA 与 CoPE 的实际使用。GQA 用官方 train/validation/testdev；VQAv2、TextVQA 的官方 validation 做测试，开发集来自 train 内按图像留出。 |
| T3 | 医学 / 文本 | MedQuAD：2000/100/200 | 使用已经固定的 80/10/10 源划分后再取样；划分思路来自 [K-COMP，NAACL 2025](https://aclanthology.org/2025.naacl-long.351/)，seed/逐条 ID 是本项目的公开复现选择，不冒称论文原 ID。 |
| T4 | 医学 / 图像 | RAD/PathVQA/SLAKE；train 300/1000/700，dev 0/60/40，test 50/100/50 | 开放问答来源及官方任务划分参考 Expert-CFG、DuCoR。RAD 没有独立官方 dev，不借用其 test 调参；SLAKE 原有双语题目保留。 |
| T5 | 法律 / 文本 | CUAD：2000/100/200 | 官方合同 train/test；官方 train 合同内固定留开发合同；长文本用问题驱动的 BM25 选原文块，不输入 gold span。保留多个必要答案 span 的原始文本。 |
| T6 | 法律 / 图像 | CUAD 原始 PDF、原始 QA：2000/100/200 | CUAD 的 PDF 文档 QA 使用依据见 [MultiDocFusion，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1062/)。本轮是新的 Qwen2-VL 页面图像适配，**不是该论文端到端 VLM 的精确复现**：在当前合同内，仅用问题 BM25 选前两页，按页序输入；不凭答案选页，不用 OCR 文字代替图片。 |
| T7 | 金融 / 文本 | FinQA：2000/100/200 | 作者官方 train/dev/test；原始文本和表格作输入，program、gold_inds 与 exe_ans 不进入输入。目标保留现有原始人工回答视图。 |
| T8 | 金融 / 图像 | TAT-DQA：2000/100/200 | 官方 train/dev/test；真实原始页图，保留必要回答部分及 scale 元数据；参照 Doc2Soar 的来源协议，但本轮生成回答不是原论文专用推理系统。 |
| T9 | 艺术 / 文本 | AQUA knowledge：2000/100/200 | [ArtQuest，WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf) 明确评估不输入图像的 closed-book knowledge QA，本轮沿用这一输入形式，不以答案或图像描述补充上下文。 |
| T10 | 艺术 / 图像 | AQUA visual：2000/100/200 | [AQUA 作者数据说明](https://github.com/noagarcia/ArtVQA/tree/master/AQUA) 的原 visual 标记、官方 train/val/test 和 SemArt 原始绘画；自由生成原答案，不改成选项分类。 |

取样是源划分内固定 seed 的伪随机顺序取前 N（实现为 SHA256 排序）。论文来源提供任务/划分/输入依据；**2,000/100/200、混合配额与 seed 42 哈希 ID 是本项目计算预算，不是声称这些论文均采用这个数量或这些 ID**。无官方 dev 时留出规则与源划分在 manifest 明示，不称为官方 dev。

法律两个模态任务的 QA ID 在每个角色内互不重复；训练/开发/测试遵循同一组合同角色。十任务之间没有 QA ID 跨 train/dev/test 重复。医学 VQA 原官方协议可能复用图像；保留官方角色并报告复用，不能把“无 QA ID 重复”写成患者/图像完全隔离。对同一合同/图像的模态迁移属于此十任务设计，需要在文章中说明。

原始文件没有修改。选中的 QA 是实验视图，图像采用原文件或原 PDF 栅格化页面；艺术只取固定清单需要的原图，直连、不启用 Clash。

## 真实接入范围

目前启动队列为 **SeqLoRA + SMoLoRA，seed 42**。DGG、PaRSP、PASs-MoE、Hyper-LLaVA 保留为待适配项，不以简化约束或普通 LoRA 替代它们。回放类论文继续保留在二维文献表，但不做本轮实验。

SMoLoRA 的专家类与 forward 直接从已核验作者源文件加载，保留视觉/语言侧路由、指令侧路由、双路专家及加权融合。沿用作者八专家、总 rank 128、alpha 256、全语言线性层和首阶段后冻结路由的设置。作者 `SMoLoraModel.add_adapter()` 最后调用 `mark_only_lora_as_trainable()`，实际冻结非 LoRA 参数（包含 projector）；本轮按这一有效源码设置冻结 Qwen2-VL 视觉编码器和 merger，不能仅凭 shell 的 projector-lr 参数推断其被训练。纯文本任务和 Qwen2-VL 替换是本轮适配扩展；没有伪图像输入，也没有删除专家/路由核心模块。

作者源码的 hard top-1 路由通过 argmax 形成离散选择；本轮保留其原数学过程，没有擅自换成可微 soft routing。指令向量按作者 all-MiniLM-L6-v2 均值池化生成；仅由每个任务的可观察输入指令决定，不使用测试答案。基座仍为 Qwen2-VL-2B-Instruct，MiniLM 是路由辅助编码器。

SeqLoRA 使用 q/v、rank 8、alpha 32 基础对照。两者均真实图像输入，视觉模块按上述源码设置冻结。每任务一轮、effective batch 16、micro batch 1、适配器 lr 1e-4、3% warmup、FP16 基座与 FP32 适配参数，bias/norm 不做 weight decay。模型/任务输入差异、可训练参数量及耗时单独报告，样本数相等不等于 token 或算力相等。

小型真实 Qwen2-VL 的文本和图像更新、零适配器与基座一致、保存恢复后下一步 Adam 一致已通过；完整 2B 的显存/数值检查在 GPU 分配后执行。工程预检修改的模型会丢弃，正式运行重新加载基座，不把预检样本计作正式训练。

冻结视觉模块的输出可按原图、预处理网格和固定基座版本缓存，避免每次重算；不缓存答案或可训练模块。原图与缓存的 loss 等价已通过小模型检查，缓存上限 12 GiB，达到上限后继续计算而不减少数据。长回答的 ROUGE-L 使用与常规动态规划结果一致的加速实现，避免评分耗时成为瓶颈。

## 启动与保存

服务器仓库根目录：`/data2/liyapeng_grp/program/LLMCL`。

```bash
cd /data2/liyapeng_grp/program/LLMCL
/tmp/llmcl-assets-env/bin/python exp/ten_task_cl_20261012/launch.py
```

入口可重复调用：已有启动器运行时不重复排队。先准备固定数据和选中图片，再等待连续两次空闲的 GPU；不同方法分配不同卡。不会停止其他用户进程。完整模型预检失败的作业记为失败，不悄悄缩小正式样本数或换掉方法。

本轮运行目录：`exp/ten_task_cl_20261012/runs/<method>/seed_42/<run_id>/`。总控/准备日志在 `pipeline_logs/<run_id>/`。模型、优化器、调度器、混合精度、方法状态、随机状态和十任务评测全部保存；T1–T10 独立保留，阶段内每 25 个 update 另存 last；完成阶段原子写 COMPLETE 后才标完成。

训练前及每阶段后测试全部十任务，各 200 QA，构成 11×10 矩阵。每阶段额外测当前任务 100 dev，只作监测。矩阵主分数为统一生成回答 ROUGE-L F1（0–100），同时保留 token F1/EM；不是 FinQA execution accuracy、CUAD AUPR 或 VQAv2 原排行榜指标。AP、ForgetRate、FWT、BWT 沿用已有矩阵定义，ForgetRate 是分数差，不是比值。只有 seed 42，不报告多种子均值或标准差。

报告在 `summary/ten_task_cl_20261012/`；逐条预测、日志和权重不推 GitHub。旧根目录 43 项脚本/配置/旧测试移入 `exp/ago/legacy_scripts/` 与 `summary/ago/legacy_scripts/`，没有删除原数据、模型或作者源码。
