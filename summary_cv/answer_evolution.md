# 同一测试实例的回答演变

每个任务沿用固定顺序的第一条示例；完整记录覆盖基线和全部九个学习阶段，未按回答变化或得分挑选。表格只节选前 140 个字符，完整文本和逐题分数见 [JSON](answer_evolution.json)。

## 新闻摘要（XSum）

实例：`task1290-fdb2129f72f349019654d8b07a72c90b`；完整输入与参考见 [实例文档](examples.md)。

| 阶段 | SeqLoRA | MIGU-LoRA | O-LoRA | SAPT-LoRA |
|---:|---|---|---|---|
| 0 | Cross, who has admitted to the murders, said he hoped to "die a martyr" after the verdict was announced. | Cross, who has admitted to the murders, said he hoped to "die a martyr" after the verdict was announced. | Cross, who has admitted to the murders, said he hoped to "die a martyr" after the verdict was announced. | Cross, who has admitted to the murders, said he hoped to "die a martyr" after the verdict was announced. |
| 1 | A jury in Kansas City has convicted a man of killing three people at two Jewish centres in April 2014. | A jury in Kansas City has convicted a man of killing three people at two Jewish centres in April 2014. | A jury in Kansas City has convicted a man of killing three people at two Jewish centres in April 2014. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. |
| 2 | A jury in Kansas City has convicted a man of killing three people at two Jewish centres in April 2014. | A jury in Kansas City has convicted a man of killing three people at two Jewish centres in April 2014. | A jury in Kansas City has convicted a man of killing three people at two Jewish centres in April 2014. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. |
| 3 | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. | A white supremacist who shot dead three people in a Jewish community centre in Kansas City, Missouri, has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been found guilty of murder. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been found guilty of murder. |
| 4 | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been found guilty of murder. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. |
| 5 | A white supremacist who shot dead three people in a Jewish community centre in Kansas City has been found guilty of murder. | A white supremacist who shot dead three people in a Jewish community centre in Kansas City has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. |
| 6 | A white supremacist who shot dead three people in a Jewish community center in Kansas City, Missouri, has been found guilty of murder. | A white supremacist who shot dead three people in a Jewish community centre in Kansas City, Missouri, has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. |
| 7 | A white supremacist who shot dead three people in a Jewish community center in Kansas City, Missouri, has been found guilty of murder. | A white supremacist who shot dead three people in a Jewish community center in Kansas City, Missouri, has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. |
| 8 | A white supremacist who shot dead three people in a Jewish community center in Kansas City, Missouri, has been found guilty of murder. | A white supremacist who shot dead three people in a Jewish community center in Kansas City, Missouri, has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. |
| 9 | A white supremacist who shot dead three people at a Jewish community center in Kansas City, Missouri, has been found guilty of murder. | A white supremacist who shot dead three people in a Jewish community centre in Kansas City, Missouri, has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. | A white supremacist who shot dead three people at a Jewish community centre in Kansas City has been sentenced to death. |

## 阅读理解（Quoref）

实例：`task002-0c57e29b81724af1acd6f4265c9b2bcd`；完整输入与参考见 [实例文档](examples.md)。

| 阶段 | SeqLoRA | MIGU-LoRA | O-LoRA | SAPT-LoRA |
|---:|---|---|---|---|
| 0 | William the Conqueror | William the Conqueror | William the Conqueror | William the Conqueror |
| 1 | William the Conqueror | William the Conqueror | William the Conqueror | William the Conqueror |
| 2 | Rochester. | Rochester. | Rochester. | William the Conqueror |
| 3 | Rochester. | Rochester | Rochester | William the Conqueror |
| 4 | William Rufus. | Rochester. | Rochester. | William the Conqueror |
| 5 | Rochester. | Rochester | Rochester | William the Conqueror |
| 6 | Rochester Castle. | Rochester. | Rochester. | William the Conqueror |
| 7 | Rochester Castle | Rochester. | Rochester. | William the Conqueror |
| 8 | Rochester Castle | Rochester Castle | Rochester. | William the Conqueror |
| 9 | William the Conqueror | Rochester Castle | Rochester | William the Conqueror |

## 关系抽取

实例：`task1510-5f9d683cda37406b8581222d07f79df2`；完整输入与参考见 [实例文档](examples.md)。

| 阶段 | SeqLoRA | MIGU-LoRA | O-LoRA | SAPT-LoRA |
|---:|---|---|---|---|
| 0 | radio is a member of tv | radio is a member of tv | radio is a member of tv | radio is a member of tv |
| 1 | IsA | IsA | IsA | IsA Antonym |
| 2 | IsA | IsA | IsA | IsA Antonym |
| 3 | radio Antonym tv | radio Antonym tv | radio Antonym tv | radio Antonym tv |
| 4 | radio Antonym tv | radio Antonym tv | radio Antonym tv | radio Antonym tv |
| 5 | radio Antonym tv | radio Antonym tv | radio Antonym tv | radio Antonym tv |
| 6 | radio Antonym tv | radio Antonym tv | radio Antonym tv | radio Antonym tv |
| 7 | radio Antonym tv | radio Antonym tv | radio Antonym tv | radio Antonym tv |
| 8 | radio Antonym tv | radio Antonym tv | radio Antonym tv | radio Antonym tv |
| 9 | radio Antonym tv | radio Antonym tv | radio Antonym tv | radio Antonym tv |

## 对话续写（PersonaChat）

实例：`task1729-53143860f49c4b868f90de7fd8baa281`；完整输入与参考见 [实例文档](examples.md)。

| 阶段 | SeqLoRA | MIGU-LoRA | O-LoRA | SAPT-LoRA |
|---:|---|---|---|---|
| 0 | I'm sorry to hear that you don't have a job. It sounds like you're spending a lot of time on the couch. Is there anything you'd like to talk… | I'm sorry to hear that you don't have a job. It sounds like you're spending a lot of time on the couch. Is there anything you'd like to talk… | I'm sorry to hear that you don't have a job. It sounds like you're spending a lot of time on the couch. Is there anything you'd like to talk… | I'm sorry to hear that you don't have a job. It sounds like you're spending a lot of time on the couch. Is there anything you'd like to talk… |
| 1 | I'm sorry to hear that. | I'm sorry to hear that. | I'm sorry to hear that. | I'm sorry to hear that. |
| 2 | I know what you mean. I don't like to leave the house. | I'm sorry to hear that. I'm not sure what you mean by "get out and see the world". | I'm sorry to hear that. | I'm sorry to hear that. |
| 3 | I know, I'm not very social. | I'm sorry to hear that. | I'm sorry to hear that. | I'm sorry to hear that. |
| 4 | I do not have a job. | I know, I don't like to go out. | I know, I don't like to go out. | I'm sorry to hear that. |
| 5 | I do not have a job. | I do not have a job. | I know, I don't like to go out. | I'm sorry to hear that. |
| 6 | I do not have a job. | I do not have a job. | I don't have a job. | I'm sorry to hear that. |
| 7 | I do not have a job. | I do not have a job. | I do not have a job. | I'm sorry to hear that. |
| 8 | I do not have a job. | I do not have a job. | I do not have a job. | I'm sorry to hear that. |
| 9 | I do not have a job. | I do not have a job. | I do not have a job. | I'm sorry to hear that. |

## 因果推理（GLUCOSE）

实例：`task748-2f188cfafd6043ad86d7806cba4a41da`；完整输入与参考见 [实例文档](examples.md)。

| 阶段 | SeqLoRA | MIGU-LoRA | O-LoRA | SAPT-LoRA |
|---:|---|---|---|---|
| 0 | All the towels in the house are wet >causes> Someone must be extra dry. | All the towels in the house are wet >causes> Someone must be extra dry. | All the towels in the house are wet >causes> Someone must be extra dry. | All the towels in the house are wet >causes> Someone must be extra dry. |
| 1 | Someone must be extra dry. | Someone must be extra dry. | Someone must be extra dry. | Someone must be extra dry. |
| 2 | Someone must be extra dry. | Someone must be extra dry. | Someone must be extra dry. | Someone must be extra dry. |
| 3 | Someone >causes/enables> Someone | Someone >causes/enables> Someone was really wet. | Someone >causes/enables> Someone | Someone must be extra dry. |
| 4 | Someone must be extra dry. | Someone must be extra dry. | Someone must be extra dry. | Someone must be extra dry. |
| 5 | All the towels are wet >Causes/Enables> Someone is extra dry | All the towels in the house are wet >Causes/Enables> Someone is extra dry | All the towels in the house are wet >Causes/Enables> Someone must be extra dry | Someone must be extra dry. |
| 6 | All the towels in the house are wet >Causes/Enables> Someone is extra dry | All the towels in the house are wet >Causes/Enables> Someone is extra dry | All the towels in the house are wet >Causes/Enables> Someone is extra dry | Someone must be extra dry. |
| 7 | All the towels are wet >Causes/Enables> Someone is extra dry | All the towels in the house are wet >Causes/Enables> Someone is extra dry | Someone is extra dry >Causes/Enables> Someone is really wet | Someone must be extra dry. |
| 8 | all the towels are wet >Causes/Enables> someone is extra dry | All the towels in the house are wet >Causes/Enables> Someone is extra dry | Someone is extra dry >Causes/Enables> Someone is really wet | Someone must be extra dry. |
| 9 | All the towels are wet >Causes/Enables> Someone is extra dry | All the towels in the house are wet >Causes/Enables> Someone is extra dry | All the towels in the house are wet >Causes/Enables> Someone is extra dry | Someone must be extra dry. |

## 医患对话转临床记录（MTS-Dialog）

实例：`mts_test_32`；完整输入与参考见 [实例文档](examples.md)。

| 阶段 | SeqLoRA | MIGU-LoRA | O-LoRA | SAPT-LoRA |
|---:|---|---|---|---|
| 0 | Clinical Note: Poison Control advised that the patient's ingestion of the liquid was likely a nontoxic ingestion, as it was a relatively sma… | Clinical Note: Poison Control advised that the patient's ingestion of the liquid was likely a nontoxic ingestion, as it was a relatively sma… | Clinical Note: Poison Control advised that the patient's ingestion of the liquid was likely a nontoxic ingestion, as it was a relatively sma… | Clinical Note: Poison Control advised that the patient's ingestion of the liquid was likely a nontoxic ingestion, as it was a relatively sma… |
| 1 | The patient, a 10-year-old girl, was brought to the emergency department after ingesting a liquid substance. The patient was initially vomit… | The patient, a 10-year-old girl, was brought to the emergency department after ingesting a liquid substance. The patient was initially vomit… | The patient, a 10-year-old girl, was brought to the emergency department after ingesting a liquid substance. The patient was initially vomit… | The patient has ingested a small amount of a liquid substance. The substance is not toxic and is not likely to be the cause of the patient's… |
| 2 | The patient, a 10-year-old girl, was brought to the emergency department after ingesting a liquid. The patient was brought to the emergency … | The patient, a 10-year-old girl, was brought to the emergency department after ingesting a liquid substance. The patient was initially vomit… | The patient, a 10-year-old girl, was brought to the emergency department after ingesting a liquid substance. The patient was initially vomit… | The patient has ingested a small amount of a liquid substance. The substance is not toxic and is not likely to be the cause of the patient's… |
| 3 | The patient, a 10-year-old girl, was brought to the emergency department after ingesting a liquid. The patient was brought to the emergency … | The patient, a 10-year-old girl, was brought to the emergency department after ingesting a liquid. The patient was brought to the emergency … | The patient, a 10-year-old girl, was brought to the emergency department after ingesting a liquid substance. The patient was initially vomit… | The patient has ingested a small amount of a liquid substance. The substance is not toxic and is not likely to be the cause of the patient's… |
| 4 | It is important to note that the patient is not experiencing any symptoms of poisoning. | The patient is a 25-year-old woman who presented to the emergency department with a history of a 10-hour history of vomiting and diarrhea. S… | The patient has ingested a small amount of a liquid substance. The patient is not exhibiting any signs of toxicity. | The patient has ingested a small amount of a liquid substance. The substance is not toxic and is not likely to be the cause of the patient's… |
| 5 | Thank you for your help. | Thank you for your time. | Patient has ingested a small amount of a liquid. It is not likely to be a toxic ingestion. Patient is not exhibiting any signs of toxicity. | The patient has ingested a small amount of a liquid substance. The substance is not toxic and is not likely to be the cause of the patient's… |
| 6 | No further action is required. | No further action taken. | No further action is required. | The patient has ingested a small amount of a liquid substance. The substance is not toxic and is not likely to be the cause of the patient's… |
| 7 | No further action. | No further action is required. | Nontoxic ingestion of liquid. | The patient has ingested a small amount of a liquid substance. The substance is not toxic and is not likely to be the cause of the patient's… |
| 8 | No further action. | Nontoxic ingestion. | Nontoxic ingestion of liquid. | The patient has ingested a small amount of a liquid substance. The substance is not toxic and is not likely to be the cause of the patient's… |
| 9 | No further action. | Thank you. | No further action required. | The patient has ingested a small amount of a liquid substance. The substance is not toxic and is not likely to be the cause of the patient's… |

## 影像 Findings 转 Impression（IU-Xray）

实例：`CXR979`；完整输入与参考见 [实例文档](examples.md)。

| 阶段 | SeqLoRA | MIGU-LoRA | O-LoRA | SAPT-LoRA |
|---:|---|---|---|---|
| 0 | The chest radiograph reveals a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear hyperexpanded without focal consolidat… | The chest radiograph reveals a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear hyperexpanded without focal consolidat… | The chest radiograph reveals a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear hyperexpanded without focal consolidat… | The chest radiograph reveals a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear hyperexpanded without focal consolidat… |
| 1 | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… |
| 2 | The chest radiograph demonstrates a normal cardiac silhouette without evidence of cardiomegaly or cardiomegaly. The lungs appear hyperexpand… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… |
| 3 | The chest radiograph demonstrates a normal cardiac silhouette without evidence of cardiomegaly or cardiomegaly. The lungs appear normal with… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… |
| 4 | The chest radiograph is unremarkable. There is no evidence of acute or chronic pulmonary disease. The cardiac silhouette is normal in size a… | The chest radiograph is unremarkable. The cardiac silhouette is normal in size and contour. There is no evidence of hyperexpansion of the lu… | The chest radiograph demonstrates a normal cardiomegaly with no abnormality in the cardiac silhouette. The lungs appear hyperexpanded withou… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… |
| 5 | The chest radiograph is unremarkable. There is no evidence of acute or chronic pulmonary disease. The cardiac silhouette is normal in size a… | The chest radiograph demonstrates a normal cardiac silhouette with no evidence of cardiomegaly or cardiomegaly. The lungs appear hyperexpand… | The chest radiograph demonstrates a normal cardiac silhouette with no evidence of cardiomegaly or cardiomegaly. The lungs appear hyperexpand… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… |
| 6 | The chest radiograph is unremarkable. There is no evidence of acute or chronic bone abnormality. | The chest radiograph is unremarkable. There is no evidence of acute or chronic pulmonary disease. The cardiac silhouette is normal in size a… | The chest radiograph is unremarkable. The cardiac silhouette is normal in size and contour. There is no evidence of cardiomegaly, cardiomega… | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… |
| 7 | No acute cardiopulmonary abnormality. | No acute cardiopulmonary abnormality. | No acute cardiopulmonary abnormality. | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… |
| 8 | No acute cardiopulmonary abnormality. | No acute cardiopulmonary abnormality. | No acute cardiopulmonary abnormality. | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… |
| 9 | No acute cardiopulmonary abnormality. | No acute cardiopulmonary abnormality. | No acute cardiopulmonary abnormality. | The chest radiograph demonstrates a normal cardiomegaly with no evidence of cardiomegaly. The lungs appear normal without evidence of consol… |

## 自然图像问答（VQAv2）

实例：`vqav2_388829000`；完整输入与参考见 [实例文档](examples.md)。

![自然图像问答（VQAv2）](../sample/natural_vqav2_COCO_val2014_000000388829.jpg)

| 阶段 | SeqLoRA | MIGU-LoRA | O-LoRA | SAPT-LoRA |
|---:|---|---|---|---|
| 0 | yes | yes | yes | yes |
| 1 | yes | yes | yes | yes |
| 2 | yes | yes | yes | yes |
| 3 | yes | yes | yes | yes |
| 4 | yes | yes | yes | yes |
| 5 | yes | yes | yes | yes |
| 6 | yes | yes | yes | yes |
| 7 | yes | yes | yes | yes |
| 8 | yes | yes | yes | yes |
| 9 | yes | yes | yes | yes |

## 医学图像问答（VQA-RAD）

实例：`vqarad_0001`；完整输入与参考见 [实例文档](examples.md)。

![医学图像问答（VQA-RAD）](../sample/medical_vqa_rad_synpic29265.jpg)

| 阶段 | SeqLoRA | MIGU-LoRA | O-LoRA | SAPT-LoRA |
|---:|---|---|---|---|
| 0 | Yes | Yes | Yes | Yes |
| 1 | No | No | No | No |
| 2 | No | No | No | Yes |
| 3 | Yes | No | No | Yes |
| 4 | Yes | No | Yes | No |
| 5 | Yes | No | Yes | No |
| 6 | Yes | No | No | No |
| 7 | Yes | No | No | No |
| 8 | yes | yes | yes | No |
| 9 | No | No | No | No |
