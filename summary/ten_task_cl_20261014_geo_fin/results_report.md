# 已完成结果：逐阶段十任务性能与最终指标

[性能下降原因诊断：识别、知识、推理与遗忘](failure_diagnosis_20261015/README.md)

生成时间：2026-10-15T16:03:39.288368+08:00。本报告是独立快照，实时队列不会覆盖。

已完成 **6/8** 个方法，每个方法核验 **110 个评测单元、5500 条逐题预测**，ID 与冻结测试集一致；逐题分数均值与矩阵、最终指标交叉核对通过。

基座 Qwen2-VL-2B-Instruct；每任务训练 100／测试 50，无验证集；一轮训练；仅 seed 42；不保存 checkpoint。所有方法使用相同冻结样本及训练顺序。

## 图怎么读

每个完成的方法一张图，**10 条折线分别代表 T1–T10，每条 11 个节点**。横轴 S0–S10 是训练阶段；S0 是训练前，Si 是完成任务 Ti 训练后的评测。颜色和标记在所有方法间一致；纵轴统一为 0–100 的 ROUGE-L F1，越高越好。

沿同一条线从左向右看，就是固定任务随持续学习的真实性能轨迹。Sj 是任务 Tj 刚学完的位置，之后的下降体现遗忘；之前的变化反映未学习任务的迁移。未学习任务也保留实际评测值。没有插值、平滑或缺失值补齐。此前以任务为横轴的阶段快照图不用于判断时间变化，主图现已纠正。

| 任务 | 领域 | 模态 | 数据集 |
|---|---|---|---|
| T1 | 自然／通用 | 文本 | SQuAD 1.1 |
| T2 | 自然／通用 | 图像 | VQAv2 / GQA / TextVQA |
| T3 | 医学 | 文本 | MedQuAD |
| T4 | 医学 | 图像 | VQA-RAD / PathVQA / SLAKE |
| T5 | 地理／遥感 | 文本 | GeoQuestions1089 |
| T6 | 地理／遥感 | 图像 | GeoChat |
| T7 | 金融 | 文本 | FinQA |
| T8 | 金融 | 图像 | MME-Finance PC |
| T9 | 艺术／文化 | 文本 | AQUA knowledge |
| T10 | 艺术／文化 | 图像 | AQUA visual |

## 论文二维表中的定义

本轮方法均属于二维表的**持续微调**大类。基于架构的方法主要通过专家、模块或路由结构分离和组合任务能力；基于正则化的方法通过子空间、梯度或表征／输出约束保护已有知识。这里沿用仓库论文表分类，具体方法可同时包含多种组件。

| 方法 | 大类 → 小类 | 主要机制 | 当前状态 |
|---|---|---|---|
| SMoLoRA | 持续微调 → 基于架构 | 双路路由与低秩专家，区分任务指令和视觉信息。 | 尚未完成，不发布最终指标 |
| PASs-MoE | 持续微调 → 基于架构 | 路径激活子空间约束路由器与专家的协同变化。 | 已完成 11×10 |
| Hyper-LLaVA | 持续微调 → 基于架构 | 双曲不确定性与模态平衡驱动专家路由。 | 已完成 11×10 |
| PaRSP | 持续微调 → 基于正则化 | 区域定位及历史输入子空间的零空间梯度投影。 | 尚未完成，不发布最终指标 |
| OSFT | 持续微调 → 基于正则化 | 约束全量微调的可更新子空间，以正交约束保护已有知识。 | 已完成 11×10 |
| SpaRTA | 持续微调 → 基于正则化 | 秩感知的任务适配及正交约束，分离可塑与稳定部分。 | 已完成 11×10 |
| FV-guided Training | 持续微调 → 基于正则化 | 函数向量引导及 KL 约束，稳定任务相关的内部功能。 | 已完成 11×10 |
| ALER | 持续微调 → 基于正则化 | 对抗潜在嵌入搜索及修复约束，抑制遗忘。 | 已完成 11×10 |

论文链接见[仓库二维表](../../README.md)，实现与原作者设置的差异见[方法适配记录](../../exp/ten_task_cl_20261014_geo_fin/METHOD_ADAPTATIONS.md)。以下是统一小模型、小样本设置下的适配结果，不是原论文排行榜数值。OSFT、ALER 包含更重的全量参数更新，不能据此假定各方法训练成本相同。

## 最终指标汇总

### ROUGE-L

T1–T10 均为**完成全部十阶段训练后（S10）**，在各任务固定 50 条测试 QA 上的分数。EM 为归一化答案完全匹配准确率（%）；Token F1、ROUGE-L 为对应分数（0–100）。AP 是本行十个任务最终分数的算术平均；其余持续学习指标仍由完整 11×10 矩阵计算。

| 方法 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | AP ↑ | ForgetRate ↓ | FWT ↑ | BWT ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PASs-MoE | 70.282 | 66.667 | 1.533 | 13.222 | 2.000 | 59.445 | 5.667 | 7.160 | 5.711 | 60.000 | 29.169 | 8.565 | -4.818 | -7.745 |
| Hyper-LLaVA | 79.926 | 82.333 | 15.530 | 12.346 | 8.000 | 59.589 | 7.133 | 14.395 | 10.043 | 38.000 | 32.730 | 0.697 | -1.539 | -0.243 |
| OSFT | 71.149 | 73.333 | 5.076 | 12.139 | 14.000 | 63.853 | 7.000 | 7.227 | 12.573 | 58.000 | 32.435 | 6.599 | -2.821 | -5.271 |
| SpaRTA | 76.756 | 78.667 | 12.719 | 13.029 | 10.000 | 57.205 | 5.000 | 8.793 | 11.162 | 50.000 | 32.333 | 3.775 | -2.078 | -3.266 |
| FV-guided Training | 79.302 | 80.333 | 15.454 | 9.657 | 8.000 | 25.202 | 5.889 | 13.322 | 10.745 | 32.000 | 27.990 | 3.291 | -1.151 | -2.729 |
| ALER | 37.000 | 39.333 | 0.484 | 2.333 | 0.000 | 1.129 | 2.667 | 0.995 | 0.000 | 52.000 | 13.594 | 21.012 | -9.382 | -20.621 |

### Normalized EM

T1–T10 均为**完成全部十阶段训练后（S10）**，在各任务固定 50 条测试 QA 上的分数。EM 为归一化答案完全匹配准确率（%）；Token F1、ROUGE-L 为对应分数（0–100）。AP 是本行十个任务最终分数的算术平均；其余持续学习指标仍由完整 11×10 矩阵计算。

| 方法 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | AP ↑ | ForgetRate ↓ | FWT ↑ | BWT ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PASs-MoE | 56.000 | 64.000 | 0.000 | 12.000 | 2.000 | 8.000 | 0.000 | 2.000 | 0.000 | 60.000 | 20.400 | 4.667 | 2.444 | -3.556 |
| Hyper-LLaVA | 68.000 | 80.000 | 0.000 | 8.000 | 8.000 | 8.000 | 0.000 | 4.000 | 0.000 | 38.000 | 21.400 | 0.444 | 3.778 | -0.222 |
| OSFT | 58.000 | 72.000 | 0.000 | 10.000 | 14.000 | 8.000 | 0.000 | 0.000 | 2.000 | 58.000 | 22.200 | 3.556 | 3.111 | -2.444 |
| SpaRTA | 62.000 | 76.000 | 0.000 | 10.000 | 10.000 | 6.000 | 0.000 | 2.000 | 0.000 | 50.000 | 21.600 | 2.889 | 3.778 | -2.222 |
| FV-guided Training | 68.000 | 78.000 | 0.000 | 6.000 | 8.000 | 4.000 | 0.000 | 4.000 | 0.000 | 32.000 | 20.000 | 0.444 | 2.000 | 0.222 |
| ALER | 16.000 | 38.000 | 0.000 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 52.000 | 10.800 | 10.444 | -0.667 | -10.444 |

### Token F1

T1–T10 均为**完成全部十阶段训练后（S10）**，在各任务固定 50 条测试 QA 上的分数。EM 为归一化答案完全匹配准确率（%）；Token F1、ROUGE-L 为对应分数（0–100）。AP 是本行十个任务最终分数的算术平均；其余持续学习指标仍由完整 11×10 矩阵计算。

| 方法 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | AP ↑ | ForgetRate ↓ | FWT ↑ | BWT ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PASs-MoE | 70.282 | 66.667 | 1.533 | 13.222 | 2.000 | 60.630 | 5.667 | 7.791 | 5.711 | 60.000 | 29.350 | 10.151 | -6.223 | -9.270 |
| Hyper-LLaVA | 79.926 | 82.333 | 24.734 | 12.346 | 8.000 | 61.227 | 7.133 | 16.903 | 11.081 | 38.000 | 34.168 | 0.859 | -1.864 | -0.334 |
| OSFT | 71.149 | 73.333 | 5.594 | 12.139 | 14.000 | 65.510 | 7.000 | 8.577 | 12.573 | 58.000 | 32.787 | 8.019 | -3.163 | -6.588 |
| SpaRTA | 76.756 | 78.667 | 15.809 | 13.029 | 10.000 | 59.313 | 5.000 | 9.780 | 11.803 | 50.000 | 33.016 | 4.935 | -2.282 | -4.345 |
| FV-guided Training | 79.302 | 80.333 | 24.041 | 9.657 | 8.000 | 26.606 | 5.889 | 15.205 | 11.697 | 32.000 | 29.273 | 3.434 | -1.205 | -2.866 |
| ALER | 37.000 | 39.333 | 0.484 | 2.333 | 0.000 | 1.129 | 2.667 | 0.995 | 0.000 | 52.000 | 13.594 | 22.041 | -11.076 | -21.533 |

### 指标定义与解释

令 R[i,j] 为完成第 i 阶段后在任务 j 上的分数（i=0…10，j=1…10）。每一种评分分别从自己的完整 11×10 矩阵计算：

- AP = mean(j=1…10) R[10,j]：最终平均性能。
- ForgetRate = mean(j=1…9) (max(k=j…9) R[k,j] − R[10,j])：沿用当前实验定义的历史最佳与最终性能之差，不截断负值。
- BWT = mean(j=1…9) (R[10,j] − R[j,j])：后续训练对已学任务的影响。
- FWT = mean(j=2…10) (R[j−1,j] − R[0,j])：学习某任务前，相对本方法初始模型的变化。

**ForgetRate 是分数点差，不是归一化比例**；FWT、BWT 也为分数点差。负 BWT 表示平均遗忘，负 FWT 表示尚未学习的任务平均低于初始参考。统一 EM、词面 Token F1、ROUGE-L 不等于各原始基准的全部官方指标；EM 的正确数／总数保留在任务及数据集明细中。只有一个 seed，不报告统计显著性或跨 seed 误差条。

## 各方法折线图

### PASs-MoE

二维表定位：持续微调 → 基于架构。路径激活子空间约束路由器与专家的协同变化。

![PASs-MoE 的 10 条任务曲线，每条 11 个训练阶段节点](results_review/PASs_MoE_task_trajectories_rougeL.png)

[矢量 PDF](results_review/PASs_MoE_task_trajectories_rougeL.pdf) · [ROUGE-L 11×10 数据](results_review/PASs_MoE_rougeL_11x10.csv) · [EM 矩阵](results_review/PASs_MoE_exact_match_11x10.csv) · [Token F1 矩阵](results_review/PASs_MoE_token_f1_11x10.csv)

### Hyper-LLaVA

二维表定位：持续微调 → 基于架构。双曲不确定性与模态平衡驱动专家路由。

![Hyper-LLaVA 的 10 条任务曲线，每条 11 个训练阶段节点](results_review/Hyper_LLaVA_task_trajectories_rougeL.png)

[矢量 PDF](results_review/Hyper_LLaVA_task_trajectories_rougeL.pdf) · [ROUGE-L 11×10 数据](results_review/Hyper_LLaVA_rougeL_11x10.csv) · [EM 矩阵](results_review/Hyper_LLaVA_exact_match_11x10.csv) · [Token F1 矩阵](results_review/Hyper_LLaVA_token_f1_11x10.csv)

### OSFT

二维表定位：持续微调 → 基于正则化。约束全量微调的可更新子空间，以正交约束保护已有知识。

![OSFT 的 10 条任务曲线，每条 11 个训练阶段节点](results_review/OSFT_task_trajectories_rougeL.png)

[矢量 PDF](results_review/OSFT_task_trajectories_rougeL.pdf) · [ROUGE-L 11×10 数据](results_review/OSFT_rougeL_11x10.csv) · [EM 矩阵](results_review/OSFT_exact_match_11x10.csv) · [Token F1 矩阵](results_review/OSFT_token_f1_11x10.csv)

### SpaRTA

二维表定位：持续微调 → 基于正则化。秩感知的任务适配及正交约束，分离可塑与稳定部分。

![SpaRTA 的 10 条任务曲线，每条 11 个训练阶段节点](results_review/SpaRTA_task_trajectories_rougeL.png)

[矢量 PDF](results_review/SpaRTA_task_trajectories_rougeL.pdf) · [ROUGE-L 11×10 数据](results_review/SpaRTA_rougeL_11x10.csv) · [EM 矩阵](results_review/SpaRTA_exact_match_11x10.csv) · [Token F1 矩阵](results_review/SpaRTA_token_f1_11x10.csv)

### FV-guided Training

二维表定位：持续微调 → 基于正则化。函数向量引导及 KL 约束，稳定任务相关的内部功能。

![FV-guided Training 的 10 条任务曲线，每条 11 个训练阶段节点](results_review/FV_guided_Training_task_trajectories_rougeL.png)

[矢量 PDF](results_review/FV_guided_Training_task_trajectories_rougeL.pdf) · [ROUGE-L 11×10 数据](results_review/FV_guided_Training_rougeL_11x10.csv) · [EM 矩阵](results_review/FV_guided_Training_exact_match_11x10.csv) · [Token F1 矩阵](results_review/FV_guided_Training_token_f1_11x10.csv)

### ALER

二维表定位：持续微调 → 基于正则化。对抗潜在嵌入搜索及修复约束，抑制遗忘。

![ALER 的 10 条任务曲线，每条 11 个训练阶段节点](results_review/ALER_task_trajectories_rougeL.png)

[矢量 PDF](results_review/ALER_task_trajectories_rougeL.pdf) · [ROUGE-L 11×10 数据](results_review/ALER_rougeL_11x10.csv) · [EM 矩阵](results_review/ALER_exact_match_11x10.csv) · [Token F1 矩阵](results_review/ALER_token_f1_11x10.csv)

## 可追溯文件

- [完整主表 CSV](results_review/final_metrics_all_scores.csv)：三种评分各自的最终 T1–T10 分数及 AP、ForgetRate、FWT、BWT，保留完整精度。
- [核验快照与绘图原始数值](results_review/verified_snapshot.json)：实际 run 路径、配置、完整矩阵、逐预测文件 SHA-256。
- [按任务明细](task_metrics.csv)及[按来源数据集明细](dataset_metrics.csv)：EM 正确数／总数及其他分数；这两个文件随队列更新。
- [可复现生成脚本](../../exp/ten_task_cl_20261014_geo_fin/build_results_review.py)：CPU 绘图，不占用训练 GPU。

冻结 manifest SHA-256：`dab25e3940758cfac3e1bb16da5305ce6393dfa4eb22e1c39d4f99a28fdc4208`。未完成方法不使用旧运行、旧划分或中断阶段拼接成最终结果。
