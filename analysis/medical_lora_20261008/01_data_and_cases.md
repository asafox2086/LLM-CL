# 数据、评分和医学实例

本报告区分任务确实没完成、过度压缩和词面评分误伤，避免把所有低分解释成医学知识不足。以下实例为诊断选择；完整平均数来自200题，不由几个实例推断。

## 正常模板基线有多强

| 任务 | 训练集中最常见答案 | 训练频率 | 恒定回答测试ROUGE-L | 恒定回答测试EM |
|---|---|---:|---:|---:|
| medical_mts_dialog_note | none | 3.1% | 3.677 | 3.500 |
| medical_iu_xray_impression | no acute cardiopulmonary abnormality | 8.7% | 39.567 | 8.500 |
| medical_vqa_rad | yes | 29.9% | 27.500 | 27.500 |
| natural_vqav2 | yes | 23.1% | 28.500 | 28.500 |

Impression 任务中，该模板仅占训练目标的8.7%，却占 SeqLoRA 最终预测的58%。恒定回答已达39.567，SeqLoRA最终40.438、刚学完42.609。说明模型强烈偏向常见“无急性异常”句式；不能把相对基座的30余点增长全部解释成新增临床事实。该任务输入是 Findings 文字，没有读取原始影像。

## 短答题与长记录不能只比同一个指标

MTS参考平均38.375个英文词；SeqLoRA阶段0/6/7/9平均输出69.975/15.755/6.950/8.755词。参考很长而预测过短，会漏掉病史、时间、部位、检查和否定关系；反过来，基座长篇复述也不等于正确。测试仅1/200输入截断，5/200目标超过255 token；训练分别13/1000与36/1000。因此截断是局部问题，不能解释所有MTS低分。所有任务均按每条答案CE均值再按样本均值，长答案不会自然获得更多样本权重。

医学 VQA 有136条CLOSED和64条OPEN；SeqLoRA最终分别63.97%和18.75%。加入有限显式等价（One↔1、Right hemisphere↔Right、PA全称）后，OPEN升到23.44%，仍很低。该校正只用于诊断，没有替换原分数。
VQAv2测试75条yes/no里有26条的十位参考同时含yes与no，max-reference EM两种回答都可命中；所有200题中26条同时含yes/no。SeqLoRA的84.5% EM在同一归一化、留一标注者共识计算下为78.15%。此诊断不是官方VQA成绩，因为没有采用完整官方文本处理。参见 [VQA官方评价](https://visualqa.org/evaluation.html) 与 [校正CSV](score_calibration.csv)。

## 临床文字：模型具体遗漏了什么

### 1. mts_test_122

参考：

> He also had a P-Bag placed, but did not have any urine.  Therefore, a straight catheter was done, which was done with ease without complication and there was no leukocytes noted within the urine.  There was a little bit of blood from catheterization but otherwise normal urine.  X-ray noted some stool within the vault.  Child is acting normally.  He is jumping up and down on the bed without any significant findings.

| 方法 | 刚学完回答 | 最终回答 |
|---|---|---|
| SeqLoRA | Normal. | No infection. |
| MIGU-LoRA | Patient is fine. Patient is very active. Patient has stool in the colon. | No infection. |
| O-LoRA | Patient is fine. | Normal stool in the colon. |
| SAPT-LoRA | Patient: Hi. | Patient: Hi. |

这里比较数据集参考中的信息覆盖，不声称从词面评分完成临床正确性裁定。完整输入、全部回答和逐题分数在 [cases.json](cases.json)。

### 2. mts_test_67

参考：

> The patient is a 56-year-old right-handed female with longstanding intermittent right low back pain, who was involved in a motor vehicle accident in September of 2005.  At that time, she did not notice any specific injury.  Five days later, she started getting abnormal right low back pain.  At this time, it radiates into the buttocks down the posterior aspect of her thigh and into the right lateral aspect of her calf.  Symptoms are worse when sitting for any length of time, such as driving a motor vehicle.  Mild symptoms when walking for long periods of time.  Relieved by standing and lying down.  She denies any left leg symptoms or right leg weakness.  No change in bowel or bladder function.  Symptoms have slowly progressed.  She has had Medrol Dosepak and analgesics, which have not been very effective.  She underwent a spinal epidural injection, which was effective for the first few hours, but she had recurrence of the pain by the next day.  This was done four and a half weeks ago. On examination, lower extremities strength is full and symmetric.  Straight leg raising is normal.

| 方法 | 刚学完回答 | 最终回答 |
|---|---|---|
| SeqLoRA | Right lower back pain. | Right lower back pain. |
| MIGU-LoRA | Lower back pain. | Right lower back pain. |
| O-LoRA | Lower back pain. | Lower back pain. |
| SAPT-LoRA | Patient is 56 years old. She has had lower back pain for 5 days after a motor vehicle accident in September 2005. The pain is worse when she is sitting for a long time, such as driving or walking. It is relieved by lying down and standing. She has not had any weakness or pain in her left leg or right leg. She has not had any change in her bowel or urine function. She had a big injection in her back four and a half weeks ago. | Patient is 56 years old. She has had lower back pain for 5 days after a motor vehicle accident. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has had the pain for 5 days. She has |

这里比较数据集参考中的信息覆盖，不声称从词面评分完成临床正确性裁定。完整输入、全部回答和逐题分数在 [cases.json](cases.json)。

### 3. mts_test_152

参考：

> XYZ was seen by Dr. ABC for an FCR tendinitis.  We do not have his reports, but by history she has had two cortisone shots.  She plays musical instruments, and it does bother her from time to time.  She was considering surgery, but she takes ibuprofen and it seems to be well-controlled.  She is here now for consultation.

| 方法 | 刚学完回答 | 最终回答 |
|---|---|---|
| SeqLoRA | F C R tendinitis. | F C R tendinitis. |
| MIGU-LoRA | F C R tendinitis. Cortisone injections. Patient plays a few instruments. | F C R tendinitis. |
| O-LoRA | F C R tendinitis. | F C R tendinitis. |
| SAPT-LoRA | Doctor: I understand that you have been experiencing pain in your forearm. I would like to review your medical history and discuss your treatment options. | Doctor: I understand that you have been experiencing pain in your forearm. I would like to review your medical history and discuss your treatment options. |

这里比较数据集参考中的信息覆盖，不声称从词面评分完成临床正确性裁定。完整输入、全部回答和逐题分数在 [cases.json](cases.json)。

### 4. CXR1233

参考：

> Cardiomegaly without heart failure. Minimal XXXX left basilar scarring/atelectasis.

| 方法 | 刚学完回答 | 最终回答 |
|---|---|---|
| SeqLoRA | No acute cardiopulmonary abnormality. | No acute cardiopulmonary abnormality. |
| MIGU-LoRA | No acute cardiopulmonary abnormality. | No acute or chronic pulmonary or cardiac abnormality. |
| O-LoRA | No acute cardiopulmonary abnormality. | No acute cardiopulmonary abnormality. |
| SAPT-LoRA | The chest radiograph demonstrates an enlarged cardiac silhouette, low lung volumes, and a relative elevation of the right hemidiaphragm. The left base density is noted, but there is no focal consolidation, pneumothorax, or large pleural effusion. The radiograph is negative for acute bone abnormality. | The chest radiograph demonstrates an enlarged cardiac silhouette, low lung volumes, and a relative elevation of the right hemidiaphragm. The left base density is noted, but there is no focal consolidation, pneumothorax, or large pleural effusion. The radiograph is negative for acute bone abnormality. |

这里比较数据集参考中的信息覆盖，不声称从词面评分完成临床正确性裁定。完整输入、全部回答和逐题分数在 [cases.json](cases.json)。

### 5. CXR2139

参考：

> 1. No acute pulmonary abnormality. 2. Mild cardiomegaly without pulmonary edema.

| 方法 | 刚学完回答 | 最终回答 |
|---|---|---|
| SeqLoRA | No acute cardiopulmonary abnormality. | No acute abnormality. |
| MIGU-LoRA | No acute abnormality. | No acute abnormality. |
| O-LoRA | No acute abnormality. | No acute abnormality. |
| SAPT-LoRA | The chest radiograph shows no acute abnormalities in the lungs or pleural spaces. The right hilar lymph node appears to be calcified. The heart size is enlarged, and the pulmonary vascularity is within normal limits. The sternotomy site and prosthetic aortic valve are noted. | The chest radiograph shows no acute abnormalities in the lungs or pleural spaces. The right hilar lymph node appears to be calcified. The heart size is enlarged, and the pulmonary vascularity is within normal limits. The sternotomy site and prosthetic aortic valve are noted. |

这里比较数据集参考中的信息覆盖，不声称从词面评分完成临床正确性裁定。完整输入、全部回答和逐题分数在 [cases.json](cases.json)。

## 医学原图：真正的参考不一致与词面假失败

### 1. CHEST / CLOSED / vqarad_0056

![医学测试原图 vqarad_0056](assets/synpic25821.jpg)

**图 1｜该问题的原始医学图片。** 按字节复制、未修图；部位标签为数据集的 CHEST。问题：Are there >8 ribs shown in this image?；参考：Yes。不能凭该图注认定临床诊断，下面只对照数据集参考与模型输出。

| 方法 | 基座回答 | 最终回答 |
|---|---|---|
| SeqLoRA | No | No |
| MIGU-LoRA | No | No |
| O-LoRA | No | No |
| SAPT-LoRA | No | No |

四方法最终回答的Yes/No与参考相反，是比句式差异更直接的参考不一致；不据此独立裁定图像病变。

### 2. CHEST / OPEN / vqarad_0019

![医学测试原图 vqarad_0019](assets/synpic29265.jpg)

**图 2｜该问题的原始医学图片。** 按字节复制、未修图；部位标签为数据集的 CHEST。问题：How is the patient oriented?；参考：Posterior-Anterior。不能凭该图注认定临床诊断，下面只对照数据集参考与模型输出。

| 方法 | 基座回答 | 最终回答 |
|---|---|---|
| SeqLoRA | Supine | Supine |
| MIGU-LoRA | Supine | Supine |
| O-LoRA | Supine | Supine |
| SAPT-LoRA | Supine | Supine |

Supine是体位词，与参考Posterior-Anterior成像方向并不等价，体现问题语义与任务属性混淆。

### 3. HEAD / CLOSED / vqarad_0131

![医学测试原图 vqarad_0131](assets/synpic26925.jpg)

**图 3｜该问题的原始医学图片。** 按字节复制、未修图；部位标签为数据集的 HEAD。问题：Is there herniation of the brainstem secondary to the lesion；参考：No。不能凭该图注认定临床诊断，下面只对照数据集参考与模型输出。

| 方法 | 基座回答 | 最终回答 |
|---|---|---|
| SeqLoRA | Yes | yes |
| MIGU-LoRA | Yes | yes |
| O-LoRA | Yes | yes |
| SAPT-LoRA | Yes | Yes |

四方法最终回答的Yes/No与参考相反，是比句式差异更直接的参考不一致；不据此独立裁定图像病变。

### 4. HEAD / OPEN / vqarad_0296

![医学测试原图 vqarad_0296](assets/synpic46720.jpg)

**图 4｜该问题的原始医学图片。** 按字节复制、未修图；部位标签为数据集的 HEAD。问题：Which hemisphere is the ischemia located?；参考：Right hemisphere。不能凭该图注认定临床诊断，下面只对照数据集参考与模型输出。

| 方法 | 基座回答 | 最终回答 |
|---|---|---|
| SeqLoRA | Left | Right |
| MIGU-LoRA | Left | Right |
| O-LoRA | Left | Right |
| SAPT-LoRA | Left | Left |

SeqLoRA/MIGU/O-LoRA 的 Right 与参考 Right hemisphere 属于本报告显式等价，EM=0不能代表判断错误；SAPT的Left与参考方向不一致。

### 5. ABD / CLOSED / vqarad_0039

![医学测试原图 vqarad_0039](assets/synpic34515.jpg)

**图 5｜该问题的原始医学图片。** 按字节复制、未修图；部位标签为数据集的 ABD。问题：Is there evidence of small bowel obstruction on this image?；参考：Yes。不能凭该图注认定临床诊断，下面只对照数据集参考与模型输出。

| 方法 | 基座回答 | 最终回答 |
|---|---|---|
| SeqLoRA | No | No |
| MIGU-LoRA | No | No |
| O-LoRA | No | No |
| SAPT-LoRA | No | No |

四方法最终回答的Yes/No与参考相反，是比句式差异更直接的参考不一致；不据此独立裁定图像病变。

### 6. ABD / OPEN / vqarad_0352

![医学测试原图 vqarad_0352](assets/synpic27985.jpg)

**图 6｜该问题的原始医学图片。** 按字节复制、未修图；部位标签为数据集的 ABD。问题：How many instances of intussusception are in the image?；参考：One。不能凭该图注认定临床诊断，下面只对照数据集参考与模型输出。

| 方法 | 基座回答 | 最终回答 |
|---|---|---|
| SeqLoRA | 1 | 1 |
| MIGU-LoRA | 1 | 1 |
| O-LoRA | 1 | 1 |
| SAPT-LoRA | 1 | 1 |

四方法的1与参考One数量相同，属于词面假失败。

## 数据依据与重算

[答案分布](answer_statistics.csv) · [数据长度/截断](dataset_audit.csv) · [分组成绩](medical_vqa_groups.csv) · [等价校正](calibration_protocol.json) · [校正救回的实例](alias_rescued_cases.json)。原图和病例从原测试集选择，没有拿来训练。

```bash
.vision-env/bin/python analysis/medical_lora_20261008/cases.py
.vision-env/bin/python analysis/medical_lora_20261008/calibration.py
```
