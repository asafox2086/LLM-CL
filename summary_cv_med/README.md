# 医学实验：持续学习与诊断读出适配

从原始 Qwen2-VL-2B-Instruct 开始，依次学习 MedMCQA → MedQA → 胸部图像 → 头部图像 → 腹部图像。四方法使用同一数据和上一轮 LoRA 参数。只用医学 QA 训练；外部通用题仅用于测量知识变化。

**总对照入口：** [基线性能、训练过程与方法来源](../summary/medical_research_comparison.md)。本页先列本研究的诊断适配，再列四方法纯医学持续学习；两组实验各有固定测试集。

## 本研究方法：诊断读出与语言端适配

数据来自 **OmniMedVQA（CVPR 2024）**。固定131张测试图，来自 ISIC2019、Retinal OCT-C8、Fitzpatrick17k，每个来源四类。所有方法共享视觉基座；准确率为 %，提升为百分点。

| 方法／来源 | 训练设置 | 测试准确率 | 相对基础模型 | 去图像准确率 |
|---|---|---|---|---|
| 基础模型／Qwen2-VL | 本轮无训练 | 28.24 | 0.00 | 25.95 |
| 自训练医学 LoRA 对照／前期实验 | VQA-RAD 80张训练图；普通rank-8 | 30.79 | 2.54 | 25.95 |
| 医学论文方法基线／缺失 | 具体方法尚未选定并复现 | 未测 | 未测 | 未测 |
| 本研究：诊断标签监督 LoRA | 372张图；语言q/v普通rank-8；冻结视觉；3轮；3种子 | 47.84 | 19.59 | 27.99 |
| 本研究：冻结特征线性读出 | 同372张图；每来源一个线性分类器；冻结视觉 | 63.36 | 35.11 | 不适用：输入是图像特征 |

诊断标签监督 LoRA 使用普通 LoRA 结构，本研究改变的是诊断标签监督和适配实验设置。线性读出用于检验固定视觉特征中可恢复的类别信息。新 LoRA 相对原 LoRA 提升17.05个百分点，95%区间[7.38, 26.72]。线性读出按来源单独训练，LoRA联合训练，结构差异纳入结果解释。

**医学基线的来源与缺失：** 表中的30.79%来自本研究前期训练的VQA-RAD LoRA。OmniMedVQA论文提供数据和问答基准；医学论文方法在同一131张图上的复现基线尚未完成，表中明确记为“未测”。47.84%和63.36%分别对应本研究的诊断监督LoRA和线性读出。当前提升参照的是基础模型或自训练医学LoRA对照，对医学论文方法的提升尚无测量。

**训练过程：** 同一127张验证图上，第1／2／3轮的三种子均值为40.16%／42.26%／44.09%；各种子由验证集选择第3轮后，在131张测试图得到47.84%。本轮诊断适配尚未测量逐轮旧任务保留和外部知识，因此A–D图展示下面的四方法持续学习实验。

[完整诊断实验报告](../analysis/omnimed_20261009/README.md) · [逐轮训练CSV](../summary/omnimed_training_process.csv)

## 纯医学持续学习：五类真实示例

### MedMCQA

Answer this medical multiple-choice question. Respond only with the letter of the correct answer.

A man has 1x1.5cm pedunculated lesion on the soft palate which has a rough, "warty" surface but is the same colour as adjacent mucosa. Appropriate management of this lesion is to:
A. Perform an incisional biopsy
B. Perform excisional biopsy
C. Scrape for exfoliative cytology
D. Observe for two weeks

参考答案：**B**。

| 方法 | 最新评估阶段 | 模型回答 | EM (%) |
|---|---|---|---|
| SeqLoRA | 5 | B | 100.00 |
| MIGU-LoRA | 5 | B | 100.00 |
| O-LoRA | 5 | B | 100.00 |
| SAPT-LoRA | 5 | B | 100.00 |

### MedQA

Answer this medical multiple-choice question. Respond only with the letter of the correct answer.

A 3-week-old newborn is brought to the physician by his parents because of poor feeding, irritability, and frequent vomiting over the past week. The vomitus is greenish in color and smells strange. His parents have tried to feed him every 4 hours, but the patient often spits up or refuses to eat. The patient was born at term and had his first bowel movement at 50 hours of life. He has since had one bowel movement daily. He is at the 50th percentile for length, 10th percentile for weight, and 40th percentile for head circumference. He does not appear to be in acute distress. His temperature is 36.9°C (98.4°F), pulse is 140/min, respirations are 40/min, and blood pressure is 90/60 mm Hg. Physical examination shows that the patient has small, low-set ears, a broad and flat nasal bridge, and a large space between the first and second toes bilaterally. The abdomen is distended. When the finger is removed following a rectal exam, there is an explosive release of stool from the patient's rectum. An x-ray of the abdomen shows a section of dilated colon followed by a segment of colon without stool or air. Which of the following is most likely to confirm the diagnosis?
A. CT scan of the abdomen
B. Transabdominal ultrasonography
C. Anorectal manometry
D. Rectal suction biopsy

参考答案：**D**。

| 方法 | 最新评估阶段 | 模型回答 | EM (%) |
|---|---|---|---|
| SeqLoRA | 5 | A | 0.00 |
| MIGU-LoRA | 5 | A | 0.00 |
| O-LoRA | 5 | A | 0.00 |
| SAPT-LoRA | 5 | A | 0.00 |

### 胸部图像

Answer the medical question briefly using the image.
Are the lungs normal appearing?

参考答案：**No**。

![胸部图像测试原图](../sample/med/synpic29265.jpg)

**图注｜胸部图像原图。** 固定第一条测试样本，与上述问题配对；原始图片直接复制，不改尺寸或像素，不按模型表现挑选。

| 方法 | 最新评估阶段 | 模型回答 | EM (%) |
|---|---|---|---|
| SeqLoRA | 5 | Yes | 0.00 |
| MIGU-LoRA | 5 | Yes | 0.00 |
| O-LoRA | 5 | Yes | 0.00 |
| SAPT-LoRA | 5 | Yes | 0.00 |

### 头部图像

Answer the medical question briefly using the image.
Is the vertebro-basilar arterial network viewed in this section?

参考答案：**Yes**。

![头部图像测试原图](../sample/med/synpic26925.jpg)

**图注｜头部图像原图。** 固定第一条测试样本，与上述问题配对；原始图片直接复制，不改尺寸或像素，不按模型表现挑选。

| 方法 | 最新评估阶段 | 模型回答 | EM (%) |
|---|---|---|---|
| SeqLoRA | 5 | yes | 100.00 |
| MIGU-LoRA | 5 | Yes | 100.00 |
| O-LoRA | 5 | yes | 100.00 |
| SAPT-LoRA | 5 | Yes | 100.00 |

### 腹部图像

Answer the medical question briefly using the image.
Is there evidence of small bowel obstruction on this image?

参考答案：**Yes**。

![腹部图像测试原图](../sample/med/synpic34515.jpg)

**图注｜腹部图像原图。** 固定第一条测试样本，与上述问题配对；原始图片直接复制，不改尺寸或像素，不按模型表现挑选。

| 方法 | 最新评估阶段 | 模型回答 | EM (%) |
|---|---|---|---|
| SeqLoRA | 5 | No | 0.00 |
| MIGU-LoRA | 5 | No | 0.00 |
| O-LoRA | 5 | No | 0.00 |
| SAPT-LoRA | 5 | No | 0.00 |

## 当前进度与效果

| 方法 | 状态 | 阶段 | 正在处理 |
|---|---|---|---|
| SeqLoRA | completed | 5 | — |
| MIGU-LoRA | completed | 5 | — |
| O-LoRA | completed | 5 | — |
| SAPT-LoRA | completed | 5 | — |

阶段 0 为基座；阶段 1–5 对应上述五任务。每阶段重测固定全部 600 条测试题，而不是只测当前任务。未完成数据留空，不记成 0 分。

### SeqLoRA：逐阶段任务准确率

| 阶段 | MedMCQA | MedQA | 胸部图像 | 头部图像 | 腹部图像 |
|---|---|---|---|---|---|
| 0 | 43.00 | 45.50 | 41.76 | 52.00 | 35.59 |
| 1 | 45.00 | 47.50 | 42.86 | 52.00 | 35.59 |
| 2 | 44.00 | 48.00 | 42.86 | 52.00 | 35.59 |
| 3 | 44.50 | 48.00 | 43.96 | 42.00 | 44.07 |
| 4 | 44.50 | 47.00 | 48.35 | 46.00 | 38.98 |
| 5 | 44.50 | 48.00 | 46.15 | 54.00 | 47.46 |

### MIGU-LoRA：逐阶段任务准确率

| 阶段 | MedMCQA | MedQA | 胸部图像 | 头部图像 | 腹部图像 |
|---|---|---|---|---|---|
| 0 | 43.00 | 45.50 | 41.76 | 52.00 | 35.59 |
| 1 | 44.50 | 48.00 | 41.76 | 52.00 | 35.59 |
| 2 | 45.50 | 48.00 | 42.86 | 52.00 | 35.59 |
| 3 | 46.00 | 48.50 | 41.76 | 54.00 | 38.98 |
| 4 | 46.50 | 48.50 | 41.76 | 50.00 | 40.68 |
| 5 | 46.50 | 49.00 | 42.86 | 52.00 | 44.07 |

### O-LoRA：逐阶段任务准确率

| 阶段 | MedMCQA | MedQA | 胸部图像 | 头部图像 | 腹部图像 |
|---|---|---|---|---|---|
| 0 | 43.00 | 45.50 | 41.76 | 52.00 | 35.59 |
| 1 | 45.00 | 48.00 | 41.76 | 52.00 | 35.59 |
| 2 | 45.00 | 48.00 | 42.86 | 52.00 | 35.59 |
| 3 | 44.00 | 47.50 | 42.86 | 48.00 | 40.68 |
| 4 | 44.50 | 47.50 | 47.25 | 48.00 | 42.37 |
| 5 | 45.00 | 48.00 | 43.96 | 56.00 | 45.76 |

### SAPT-LoRA：逐阶段任务准确率

| 阶段 | MedMCQA | MedQA | 胸部图像 | 头部图像 | 腹部图像 |
|---|---|---|---|---|---|
| 0 | 43.00 | 45.50 | 41.76 | 52.00 | 35.59 |
| 1 | 45.00 | 47.00 | 41.76 | 52.00 | 35.59 |
| 2 | 44.50 | 46.50 | 41.76 | 52.00 | 35.59 |
| 3 | 44.50 | 47.00 | 41.76 | 52.00 | 35.59 |
| 4 | 45.00 | 46.50 | 41.76 | 52.00 | 35.59 |
| 5 | 45.00 | 47.00 | 41.76 | 52.00 | 35.59 |

### 完整 CL 指标

| 方法 | 最终 AP ↑ | 遗忘幅度 ↓ | BWT ↑ | FWT ↑ |
|---|---|---|---|---|
| SeqLoRA | 48.02 | -1.33 | 2.42 | -0.88 |
| MIGU-LoRA | 46.88 | -0.90 | 1.52 | 2.67 |
| O-LoRA | 47.74 | -1.18 | 2.27 | 1.59 |
| SAPT-LoRA | 44.27 | 0.00 | 0.12 | 0.38 |

## 学会多少、保留多少

下表比较同一批题：学习增益＝刚学完 − 基座，保留增益＝最新阶段 − 基座，后续变化＝最新阶段 − 刚学完。正的学习增益才支持“本任务确实改善”；低遗忘且低学习增益可能只是没有学会。最新阶段在学习完成前持续变化。

| 方法 | 任务 | 最新阶段 | 基座 | 刚学完 | 最新 | 学习增益 | 保留增益 | 后续变化 |
|---|---|---|---|---|---|---|---|---|
| SeqLoRA | MedMCQA | 5 | 43.00 | 45.00 | 44.50 | 2.00 | 1.50 | -0.50 |
| SeqLoRA | MedQA | 5 | 45.50 | 48.00 | 48.00 | 2.50 | 2.50 | 0.00 |
| SeqLoRA | 胸部图像 | 5 | 41.76 | 43.96 | 46.15 | 2.20 | 4.40 | 2.20 |
| SeqLoRA | 头部图像 | 5 | 52.00 | 46.00 | 54.00 | -6.00 | 2.00 | 8.00 |
| SeqLoRA | 腹部图像 | 5 | 35.59 | 47.46 | 47.46 | 11.86 | 11.86 | 0.00 |
| MIGU-LoRA | MedMCQA | 5 | 43.00 | 44.50 | 46.50 | 1.50 | 3.50 | 2.00 |
| MIGU-LoRA | MedQA | 5 | 45.50 | 48.00 | 49.00 | 2.50 | 3.50 | 1.00 |
| MIGU-LoRA | 胸部图像 | 5 | 41.76 | 41.76 | 42.86 | 0.00 | 1.10 | 1.10 |
| MIGU-LoRA | 头部图像 | 5 | 52.00 | 50.00 | 52.00 | -2.00 | 0.00 | 2.00 |
| MIGU-LoRA | 腹部图像 | 5 | 35.59 | 44.07 | 44.07 | 8.47 | 8.47 | 0.00 |
| O-LoRA | MedMCQA | 5 | 43.00 | 45.00 | 45.00 | 2.00 | 2.00 | 0.00 |
| O-LoRA | MedQA | 5 | 45.50 | 48.00 | 48.00 | 2.50 | 2.50 | 0.00 |
| O-LoRA | 胸部图像 | 5 | 41.76 | 42.86 | 43.96 | 1.10 | 2.20 | 1.10 |
| O-LoRA | 头部图像 | 5 | 52.00 | 48.00 | 56.00 | -4.00 | 4.00 | 8.00 |
| O-LoRA | 腹部图像 | 5 | 35.59 | 45.76 | 45.76 | 10.17 | 10.17 | 0.00 |
| SAPT-LoRA | MedMCQA | 5 | 43.00 | 45.00 | 45.00 | 2.00 | 2.00 | 0.00 |
| SAPT-LoRA | MedQA | 5 | 45.50 | 46.50 | 47.00 | 1.00 | 1.50 | 0.50 |
| SAPT-LoRA | 胸部图像 | 5 | 41.76 | 41.76 | 41.76 | 0.00 | 0.00 | 0.00 |
| SAPT-LoRA | 头部图像 | 5 | 52.00 | 52.00 | 52.00 | 0.00 | 0.00 | 0.00 |
| SAPT-LoRA | 腹部图像 | 5 | 35.59 | 35.59 | 35.59 | 0.00 | 0.00 | 0.00 |

## 外部知识有没有变化

每阶段测同一组独立 198 道 MMLU 题：医学 96、通用 102。使用 A/B/C/D 下一 token 概率评分，保存四个概率和逐题答案；这些题不参与训练、验证或选 checkpoint。它能显示固定题组的知识变化，不代表全部知识，也不能和自由生成 EM 混为一个指标。

| 方法 | 阶段 | 医学 (%) | 通用 (%) |
|---|---|---|---|
| SeqLoRA | 0 | 56.25 | 54.90 |
| SeqLoRA | 1 | 58.33 | 52.94 |
| SeqLoRA | 2 | 60.42 | 53.92 |
| SeqLoRA | 3 | 60.42 | 52.94 |
| SeqLoRA | 4 | 61.46 | 53.92 |
| SeqLoRA | 5 | 61.46 | 53.92 |
| MIGU-LoRA | 0 | 56.25 | 54.90 |
| MIGU-LoRA | 1 | 58.33 | 51.96 |
| MIGU-LoRA | 2 | 57.29 | 51.96 |
| MIGU-LoRA | 3 | 59.38 | 51.96 |
| MIGU-LoRA | 4 | 59.38 | 50.98 |
| MIGU-LoRA | 5 | 60.42 | 50.98 |
| O-LoRA | 0 | 56.25 | 54.90 |
| O-LoRA | 1 | 58.33 | 51.96 |
| O-LoRA | 2 | 58.33 | 51.96 |
| O-LoRA | 3 | 58.33 | 52.94 |
| O-LoRA | 4 | 58.33 | 54.90 |
| O-LoRA | 5 | 58.33 | 53.92 |
| SAPT-LoRA | 0 | 56.25 | 54.90 |
| SAPT-LoRA | 1 | 58.33 | 52.94 |
| SAPT-LoRA | 2 | 58.33 | 52.94 |
| SAPT-LoRA | 3 | 58.33 | 52.94 |
| SAPT-LoRA | 4 | 58.33 | 52.94 |
| SAPT-LoRA | 5 | 58.33 | 52.94 |

![纯医学 CL 全过程](../sample/med/process_med.png)

**图注｜A–D：四种方法在纯医学持续学习中的变化。** 横轴0＝基础模型，1＝学完MedMCQA，2＝学完MedQA，3＝学完胸部，4＝学完头部，5＝学完腹部。

| 面板／位置 | 测量什么 | 如何读 |
|---|---|---|
| A／左上：总体任务表现 | 固定600题，先算五个任务各自准确率，再等权平均 | 纵轴为准确率%；越高表示固定任务平均表现越好 |
| B／右上：医学知识 | 每阶段重测固定96道外部医学MMLU题 | 纵轴为准确率%；观察医学知识题的变化 |
| C／左下：旧任务变化 | 训练当前任务之后−之前，在同一组已学任务测试题上的宏平均准确率 | 纵轴为百分点；负值表示本次训练后旧任务准确率下降 |
| D／右下：通用知识 | 每阶段重测固定102道外部通用MMLU题 | 纵轴为准确率%；观察通用知识题的变化 |

曲线：蓝色圆点＝SeqLoRA，绿色菱形＝MIGU-LoRA，红色三角＝O-LoRA，紫色方块＝SAPT-LoRA。折线连接已测阶段，单种子；[矢量PDF](figures/process_med.pdf)。

### C图的实验设置：怎样测量旧任务受损

对每种方法，在阶段s−1保存模型并评测固定测试题；按各方法设置学习当前任务的训练集，完成阶段s，再用更新后的模型重测完全相同的旧任务测试题。SAPT还按其实现执行任务边界反思。测试题不参与训练。取每个旧任务准确率的“训练后−训练前”，再对旧任务等权平均。训练超参数及各任务样本量见本页“数据规模、实现与恢复”。

| 横轴阶段 | 本次训练 | 比较模型状态 | 测量哪些旧任务 | 固定旧测试题数 |
|---|---|---|---|---|
| 2 | MedQA | 学完MedMCQA → 学完MedQA | MedMCQA | 200 |
| 3 | 胸部图像 | 学完MedQA → 学完胸部 | MedMCQA、MedQA | 400 |
| 4 | 头部图像 | 学完胸部 → 学完头部 | MedMCQA、MedQA、胸部 | 491 |
| 5 | 腹部图像 | 学完头部 → 学完腹部 | MedMCQA、MedQA、胸部、头部 | 541 |

阶段0／1尚无可测的旧任务训练前后变化，C图留空。不同阶段的旧任务集合随学习增加，各点对应表中各自的集合。

$$\Delta_{\mathrm{old}}(s)=\frac{1}{s-1}\sum_{t=1}^{s-1}\left[R_{s,t}-R_{s-1,t}\right],\quad s=2,3,4,5.$$

**具体例子：** SeqLoRA 在阶段2训练MedQA；旧任务只有MedMCQA的同一200道测试题。训练前准确率45.00%（90题正确），训练后44.00%（88题正确），所以C图阶段2为44.00−45.00＝**−1.00个百分点**。这测量本次更新后旧任务的准确率下降；因果解释受单种子和固定题组范围限制。

| 方法 | 本次训练 | 旧任务平均：训练前 | 训练后 | C图变化／百分点 |
|---|---|---|---|---|
| SeqLoRA | MedQA | 45.00 | 44.00 | -1.00 |
| SeqLoRA | 胸部图像 | 46.00 | 46.25 | 0.25 |
| SeqLoRA | 头部图像 | 45.49 | 46.62 | 1.13 |
| SeqLoRA | 腹部图像 | 46.46 | 48.16 | 1.70 |
| MIGU-LoRA | MedQA | 44.50 | 45.50 | 1.00 |
| MIGU-LoRA | 胸部图像 | 46.75 | 47.25 | 0.50 |
| MIGU-LoRA | 头部图像 | 45.42 | 45.59 | 0.17 |
| MIGU-LoRA | 腹部图像 | 46.69 | 47.59 | 0.90 |
| O-LoRA | MedQA | 45.00 | 45.00 | 0.00 |
| O-LoRA | 胸部图像 | 46.50 | 45.75 | -0.75 |
| O-LoRA | 头部图像 | 44.79 | 46.42 | 1.63 |
| O-LoRA | 腹部图像 | 46.81 | 48.24 | 1.43 |
| SAPT-LoRA | MedQA | 45.00 | 44.50 | -0.50 |
| SAPT-LoRA | 胸部图像 | 45.50 | 45.75 | 0.25 |
| SAPT-LoRA | 头部图像 | 44.42 | 44.42 | 0.00 |
| SAPT-LoRA | 腹部图像 | 46.31 | 46.44 | 0.12 |

[逐任务训练前后明细CSV](old_task_impact_details.csv) · [C图每阶段汇总CSV](old_task_impact_stages.csv)。

## 指标与实验边界

MedMCQA/MedQA 主指标为严格答案字母准确率（允许末尾句点或右括号）；图像主指标为规范化 EM，并另存 Token F1、ROUGE-L 及 OPEN/CLOSED 分组。图像短答案同义表达可能被 EM 判错，ROUGE-L 只是文字重合。AP 为五任务等权均分；图像三个器官共享一个数据集，因此也分别报告文字/图像领域均分，不能称为三个独立数据集。

令 $R_{s,t}$ 为学完 $s$ 个任务后在任务 $t$ 上的准确率（百分数），$T=5$。

$$\mathrm{AP}=\frac{1}{T}\sum_{t=1}^{T}R_{T,t},\qquad \Delta_{\mathrm{learn},t}=R_{t,t}-R_{0,t},\qquad \Delta_{\mathrm{retain},t}=R_{T,t}-R_{0,t}.$$

$$\mathrm{BWT}=\frac{1}{T-1}\sum_{t=1}^{T-1}(R_{T,t}-R_{t,t}),\qquad \mathrm{FWT}=\frac{1}{T-1}\sum_{t=2}^{T}(R_{t-1,t}-R_{0,t}).$$

$$\mathrm{F.Rate}=\frac{1}{T-1}\sum_{t=1}^{T-1}\left(\max_{s\in\{t,\ldots,T-1\}}R_{s,t}-R_{T,t}\right).$$

单种子、固定任务顺序、每任务 1 epoch。视觉编码器冻结，图像最多 128 token；没有同时改路由、正则或分辨率，以便和上一轮对照。医学选择题只训练字母，不训练解释或临床推理过程。图像保持现有 seed42 病例分组切分，非论文标准问题级切分。MedMCQA 的 dev/test 是有标签的官方 validation 子集；MedQA 的 dev 从官方 train 单独划出。数据集间仅去除完全相同的问题，没有完成语义近重复去污染。

## 数据规模、实现与恢复

| 任务 | 训练 | 验证 | 测试 |
|---|---|---|---|
| MedMCQA | 1000 | 100 | 200 |
| MedQA | 1000 | 100 | 200 |
| 胸部图像 | 362 | 28 | 91 |
| 头部图像 | 319 | 34 | 50 |
| 腹部图像 | 319 | 38 | 59 |

模型约 2.21B；LoRA rank=8、alpha=32、dropout=0.1、q_proj/v_proj，学习率 1e-4，有效 batch=16、micro-batch=1、seed=42。方法额外参数和实际数据哈希保存在各自 config/manifest。每次优化更新保存优化器、调度器、AMP、RNG；每阶段保存完整边界，测试回答按批落盘。nohup 独立会话，关闭终端继续运行。显存不足时等待；训练 OOM 最多自动恢复 4 次，其余失败保留日志。

当前原始结果目录：`exp/CV_result_med/qwen2vl2b_pure_med/20261008T102912Z`，每个方法独立子目录；医学数据 `data/medical_qa_med`，图像原件 `CV_data/medical/vqa_rad`，展示图片 `sample/med`。

```bash
# 查看实时状态与报告
.plot-env/bin/python summary_cv_med/collect_med.py
# 中断后恢复同一轮（把下方路径替换为 latest.json 中的 run）
.vision-env/bin/python exp/start_cv_detached_med.py --resume exp/CV_result_med/qwen2vl2b_pure_med/<run>
```

数据来源：[MedMCQA 官方](https://github.com/medmcqa/medmcqa)、[MedQA 官方](https://github.com/jind11/MedQA)、[VQA-RAD 官方](https://osf.io/89kps/)。下载镜像版本、文件 SHA、选择 ID 在独立 manifest；全部配置为 `exp/configs/*_qwen2vl_med.json`，运行代码 `exp/cv_run_med.py`。
