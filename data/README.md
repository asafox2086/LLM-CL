# 文本数据：任务示例与实验效果

这里的任务要求模型生成文字答案，包括摘要、阅读理解、对话、关系和因果推理，以及医学记录生成。

## 先看数据长什么样

| 任务 | 输入节选 | 参考答案节选 | SeqLoRA 最终回答节选 |
|---|---|---|---|
| 新闻摘要（XSum） | Cross, who has admitted to the murders, said he hoped to "die a martyr" after the verdict … | White supremacist Frazier Glenn Cross has been found guilty of murdering three people at t… | A white supremacist who shot dead three people at a Jewish community center in Kansas City… |
| 阅读理解（Quoref） | Passage: It was probably William the Conqueror who gave the city and its castle to Bishop … | Rochester. | William the Conqueror |
| 关系抽取 | radio can be used as the opposite of tv | radio Antonym tv | radio Antonym tv |
| 对话续写（PersonaChat） | Personality: I try to limit how much I eat. I whine a lot. I've a golden retriever puppy. … | I like being at home with the puppy. I eat a lot and should really limit that. | I do not have a job. |
| 因果推理（GLUCOSE） | story: All the towels in the house are wet. There are only four of us living here. There a… | The towels are wet &gt;Causes/Enables&gt; The towels dry out | All the towels are wet &gt;Causes/Enables&gt; Someone is extra dry |
| 医患对话转临床记录（MTS-Dialog） | Doctor: I spoke with Poison Control regarding the possible ingestion of the liquid. They l… | I discussed the case with Poison Control and apparently this is actually relatively small … | No further action. |
| 影像 Findings 转 Impression（IU-Xray） | The cardiomediastinal silhouette is normal in size and contour. Hyperexpanded lungs withou… | Negative for acute abnormality. | No acute cardiopulmonary abnormality. |

表内为真实输入与回答节选；完整通用语言示例见 [七任务说明](../summary/README.md#先看七个任务的真实示例)，医学实例及四方法的学习前后回答见 [九任务实例](../summary_cv/examples.md)。T5 轮次另外保留 Reddit TIFU 摘要和 SciQ 科学问答任务。

## 这些数据上得到什么效果

T5 七任务最终平均 ROUGE-L：SeqLoRA 25.634、MIGU-LoRA 23.084、O-LoRA 21.926、SAPT-LoRA 12.030。之后医学续训中，SeqLoRA 的医学任务平均分提高 9.654 分，MIGU-LoRA 提高 2.658 分；另外两方法未获得平均提升。完整结果见 [语言报告](../summary/README.md)。

这些分数来自固定数据、一个种子、短训练预算。Qwen 使用五个通用语言和两个医学任务，配额不同，成绩单独见 [含图像报告](../summary_cv/README.md)。

## 指标定义与数据范围

ROUGE-L 看答案和参考的有序文字重合；EM 看归一化后的完全匹配比例；Token F1 看词覆盖。分数均为 0–100，多参考时分别取最大值。AP 是最终各任务 ROUGE-L 的等权平均；遗忘率、BWT 和 FWT 看学习过程中的变化，完整定义见 [指标说明](../summary/README.md#指标定义)。

| 轮次 | 文本任务 | 每任务训练 / 验证 / 测试 |
|---|---|---|
| T5 七任务 | 7 通用语言 | 1000 / 200 / 500 |
| T5 医学续训 | 2 医学语言 | 1000 / 100 / 200；旧任务测试不变 |
| Qwen 九任务的语言部分 | 5 通用语言 + 2 医学语言 | 1000 / 100 / 200 |

## 数据来源、准备与实现


### Full Super-NaturalInstructions (SuperNI)

- `SuperNI_full/`: complete official repository snapshot, including `tasks/`, upstream splits, documentation and licenses; ignored by Git via the root `.gitignore`.
- Source: https://github.com/allenai/natural-instructions
- Downloaded on 2026-10-06; upstream revision: `55a365637381ce7f3748fa2eac7aef1a113bbb82`.
- The snapshot contains 1,613 task JSON files. The user has selected seven generation tasks, with 1000 train / 200 dev / 500 test instances per task, drawn directly from this complete snapshot.
- Download metadata and integrity results are stored locally as `upstream_commit.json`, `upstream_tree.json`, and `verification.json` inside `SuperNI_full/`. Verification compares file contents against the official Git blob hashes and parses every task JSON.
- Upstream task-level splits differ from this project's instance-level continual-learning splits. The deterministic algorithm and verified fingerprints are documented in [the experiment protocol](../rules/001_experiment_protocol.md) and [data manifest](../rules/001_data_manifest.json). SAPT's instance splits are reference material only.

### Existing paper benchmark subsets

- `SAPT_CL_Benchmark/Long_Sequence/`: Long Sequence Benchmark task data (15 tasks).
- `SAPT_CL_Benchmark/SuperNI/`: SuperNI/SuperNatural Instructions task data used by SAPT.
- `Long_Sequence_configs/` and `SuperNI_configs/`: task split/configuration files.

These files are copied from the official SAPT repository:
https://github.com/circle-hit/SAPT
- `MAC_datasets/`: StreamingQA test set and SQuAD test set/configs from the official MAC repository. The MAC README links the full StreamingQA download; `archivalqa.yaml` is included as its loader configuration.

### 医学文本数据

MTS-Dialog 医患对话与 IU-Xray findings-to-impression 的固定划分在 `prepared/medical_generation2_after_superni7_v1/`；原始来源在 `medical_raw/`。划分与来源指纹见 [医学清单](../rules/003_medical_manifest.json)，准备流程见 [续训协议](../rules/003_medical_continuation.md)。

```bash
.conda-env/bin/python exp/prepare_medical.py
```
