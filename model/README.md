# 0.5B–3B 多模态持续学习基座

服务器目录：`/data2/liyapeng_grp/program/LLMCL/model/`。新增下载 5 个多模态模型，另复用并重新核验已有 Qwen2-VL-2B-Instruct。原有文本与医学模型保留。

4 个是真正的 pretrained Base；2 个为 Instruct 版，用作持续指令微调的起点，表中明确区分。所有模型都带视觉编码器，可处理图像；不把纯文本 LLM 当作能直接接收图像的模型。

| 独立模型目录与官方来源 | 阶段 | 标称 B | 实际存储张量参数 B（含视觉） | 下载 GB | 状态 |
|---|---|---|---|---|---|
| [SmolVLM-500M-Base](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Base) | 预训练 Base | 0.5 | 0.507 | 1.02 | 完整新增下载 |
| [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | 预训练 Base | 0.8 | 0.873 | 1.77 | 完整新增下载 |
| [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | 预训练 Base | 2 | 2.274 | 4.57 | 完整新增下载 |
| [SmolVLM-Base](https://huggingface.co/HuggingFaceTB/SmolVLM-Base) | 预训练 Base | 2.25 | 2.246 | 4.50 | 完整新增下载 |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Instruct：持续微调起点 | 2 | 2.128 | 4.27 | 完整新增下载 |
| [Qwen2-VL-2B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct) | Instruct：持续微调起点 | 2 | 2.209 | 4.43 | 已有模型重新核验 |

实际参数量按 safetensors 头部的存储张量形状计算，包含视觉部分；绑定／未重复存储的权重按存储记录计数。所有表中模型均落在约 0.5B–3B 范围。

每个文件夹包含完整权重、config、tokenizer 和 image processor 配置，不是仅下载卡片或空目录。权重大小及上游 SHA-256 已核验，分片 index 引用全部存在，视觉配置存在。模型文件没有用于新的训练，未对下载后的问答性能作实验结论。

建议以 Qwen3.5-0.8B-Base / Qwen3.5-2B-Base 做同系列规模比较，以 SmolVLM-500M-Base / SmolVLM-Base 做另一系列对照。已有 Qwen2-VL-2B-Instruct 和新增 Qwen3-VL-2B-Instruct 可作为指令微调起点。Base 版本应经过任务微调后使用，不把预训练 Base 与 Instruct 的初始指令能力混为一谈。

加载需要匹配模型版本的 Transformers；尤其 Qwen3.5 应使用支持其架构的较新版本。没有为下载修改已有训练环境或占用 GPU。逐文件哈希、源版本和许可证声明见各模型 `asset_manifest.json` 和 [完整机器索引](multimodal_manifest.json)。官方模型卡链接见上表。

复现新增五个模型：`python data/download_multimodal_assets.py --models`；按本索引固定 revision 下载。Qwen2-VL 复用原有官方 ModelScope 下载，单独记录并重新核验每个文件版本及 SHA-256。大权重在服务器本地，GitHub 仅保存索引与核验元数据。

---

## 已有实验与加载记录

### 本地模型：能力示例与实验效果

本仓库的语言实验使用 T5-Large，含图像的连续学习实验使用 Qwen2-VL-2B-Instruct。模型根据任务说明生成文字；Qwen 还接收真实图片。

## 先看模型的回答

| 任务 | 输入节选 | 参考答案节选 | SeqLoRA 最终回答节选 |
|---|---|---|---|
| 新闻摘要（XSum） | Cross, who has admitted to the murders, said he hoped to "die a martyr" after the verdict … | White supremacist Frazier Glenn Cross has been found guilty of murdering three people at t… | A white supremacist who shot dead three people at a Jewish community center in Kansas City… |
| 自然图像问答（VQAv2） | Answer the question briefly using the image. Is it cold outside? | no | yes |
| 医学图像问答（VQA-RAD） | Answer the question briefly using the image. Are the lungs normal appearing? | No | No |

这些是 Qwen 连续学习后的真实测试回答节选；完整输入、图片和四方法的学习前后回答见 [实例报告](../summary_cv/examples.md)。加载检查用的两张图像结果仅用于确认流程，正式效果以完整实验为准。

## 大体效果与指标

Qwen 九任务最终平均 ROUGE-L：SeqLoRA 48.799、MIGU-LoRA 43.964、O-LoRA 47.239、SAPT-LoRA 42.247。T5 七任务的对应均分为 25.634、23.084、21.926、12.030。两轮任务和配额不同，应各自看方法之间的结果；完整报告分别在 [summary_cv](../summary_cv/README.md) 和 [summary](../summary/README.md)。

ROUGE-L 看与参考答案的有序文字重合；EM 看规范化后完全匹配的比例；Token F1 看词覆盖；AP 是最终各任务的 ROUGE-L 平均。分数采用 0–100，并应结合遗忘幅度与真实回答理解，定义见 [指标说明](../summary/README.md#指标定义)。

## 权重、加载流程与命令


所有权重统一存放在 `model/`：

| 子目录 | 模型 |
|---|---|
| `t5-large/` | T5-Large，语言实验 |
| `phi-2/` | Phi-2，语言模型 |
| `Llama-2-7b-hf/` | Llama-2-7B，语言模型 |
| `Qwen2-VL-2B-Instruct/` | Qwen2-VL，含图像实验 |

视觉模型：**Qwen/Qwen2-VL-2B-Instruct**（2,208,985,600参数，约2.21B，权重及配套文件共4.43GB）。

路径：`model/Qwen2-VL-2B-Instruct/`。从 Qwen 官方 ModelScope 仓库下载，文件版本和SHA256固定于 `source_api.json`，实际校验记录于 `download_manifest.json`；`download_status.json` 标记完整性。

- [官方 Hugging Face 模型卡](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [官方 ModelScope 仓库](https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct)
- Apache-2.0许可，原始LICENSE随模型保留。

包括视觉编码器、语言模型、图片processor、tokenizer及chat template。使用当前 Transformers 4.46.3，图像依赖单独放在 `.vision-env`，继承原环境的PyTorch/Transformers，沿用 `.conda-env` 的 PyTorch/Transformers。

加载与推理命令：

```bash
.vision-env/bin/python exp/vision_infer.py --image /absolute/path/image.jpg \
  --question 'What is shown in the image?' --device cpu --max-new-tokens 32
```

GPU空闲时可用 `--device cuda:0`（FP16）；CPU使用FP32。输入图像默认限制到最多128个视觉token对应的像素预算，适合小规模检查。正式实验应固定并记录分辨率预算。

重新下载/断点恢复：`.conda-env/bin/python exp/download_vision_model_ranges.py`。

已完成本地GPU FP16加载及自然/医学各一张图像的生成检查，记录见 `vision_smoke_test.json`（仅流程检查，不是准确率评测）。权重中实际存储参数数为2,208,985,600。
