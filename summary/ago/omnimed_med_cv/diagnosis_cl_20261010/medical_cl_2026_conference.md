# 2026 正式顶会医学持续学习方法

核对日期：2026-10-10。

用户已确定 **MedQwen、RA-LDL 均作为本实验基线**。注册信息保存在 `exp/omnimed_med_cv/diagnosis_cl_20261010/baselines.json`，源码版本与文件校验记录保存在同目录 `source_audit.json`。

| 新增基线 | 对照实现 | 本三任务序列成绩 |
|---|---|---|
| MedQwen（CVPR 2026） | 在统一 Qwen2-VL-2B 上按论文移植谱 LoRA 专家；增加容量匹配的普通 LoRA 对照 | 已启动，见运行快照 |
| RA-LDL（ICLR 2026） | 官方算法移植到同一 Qwen 的图像特征；明确特征池化和分类读出变化 | 三种子已完成，见对比表 |

两行均加入主对比表，分别标明输出形式、参数容量和额外计算；测试时使用同一批图像和候选标签。原论文成绩与本实验成绩分栏保存。

## 与 Qwen＋LoRA 最贴近的论文：MedQwen

**Sparse Spectral LoRA: Routed Experts for Medical VLMs**，CVPR 2026 主会议，35351–35362 页。

- [CVF 正式论文记录](https://openaccess.thecvf.com/content/CVPR2026/html/Nejatimanzari_Sparse_Spectral_LoRA_Routed_Experts_for_Medical_VLMs_CVPR_2026_paper.html)
- [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Nejatimanzari_Sparse_Spectral_LoRA_Routed_Experts_for_Medical_VLMs_CVPR_2026_paper.pdf)
- [作者全文](https://arxiv.org/html/2604.01310v1)
- [官方仓库](https://github.com/IMPACT-L/MedQwen)

它把预训练权重的不同 SVD 谱段用于初始化不同 LoRA 专家，按输入选择专家，并用残差补偿和缩放稳定更新。原文第 5.5 节先训练 Harvard-FairVLMed，再训练 PathVQA，回测前一个任务；报告 MedQwen 约 5% 的准确率下降，普通 LoRA 超过 50%，MoELoRA 超过 20%。这些是原文的下降描述，不能直接当作本实验的百分点或最终准确率。

MedQwen 是本次医学 LoRA 方法对照的首选来源。复核时官方仓库仅有 README，未提供训练实现。本次移植到 Qwen2-VL-2B 标记为独立公式实现，记录谱段、专家数、激活数、缩放、残差和负载均衡设置；完整设置见本实验 README 和 protocol.json。

## 简洁医学分类 CL 方法：RA-LDL

**Random Anchors with Low-rank Decorrelated Learning: A Minimalist Pipeline for Class-Incremental Medical Image Classification**，ICLR 2026 正式会议论文。

- [正式会议记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/68a3919db3858f548dea769f2dbba611-Abstract-Conference.html)
- [会议 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/68a3919db3858f548dea769f2dbba611-Paper-Conference.pdf)
- [官方完整代码](https://github.com/CUHK-BMEAI/RA-LDL)

该方法使用随机锚点与首阶段训练的低秩投影校准特征，之后累积统计量解析更新分类器。它可以作为诊断特征读出的医学 CL 参照。其输出为分类器，需单独注明结构和候选类别范围；语言模型内部 LoRA 与分类读出方法分别报告。

## 公平对比约束及实际状态

| 项目 | 本次对比要求 |
|---|---|
| 基座 | 所有 Qwen 内部 LoRA 方法从同一 Qwen2-VL-2B 原始权重开始 |
| 数据 | 同一 ISIC2019→OCT-C8→Fitzpatrick17k 划分、顺序、训练样本和测试题 |
| 主训练预算 | 同轮数、有效 batch、学习率策略、样本顺序及种子；额外训练或计算单列 |
| 参数容量 | 报告总 adapter 参数、活跃参数与随任务增长情况；多专家设置增加容量匹配的普通 LoRA 对照 |
| 评测 | 每个任务结束回测全部任务，统一选项准确率、大小写等价规则和平衡准确率 |
| 模型选择 | 统一验证集选择规则，测试集仅用于评测 |
| 外部资源 | 单列回放样本、教师、统计量和外部数据的使用 |

2026-10-10 本组三任务实验已在服务器运行。运行状态见 run_status.json，完成种子的成绩见 baseline_comparison.csv；不足三种子的行是阶段性结果。EWC/KD 作为通用 CL 参考，2026 正式医学论文来源使用本文件核实的记录。
