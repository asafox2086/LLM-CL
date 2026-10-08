# 本地模型

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

包括视觉编码器、语言模型、图片processor、tokenizer及chat template。使用当前 Transformers 4.46.3，图像依赖单独放在 `.vision-env`，继承原环境的PyTorch/Transformers，不修改正在训练的 `.conda-env`。

示例：

```bash
.vision-env/bin/python exp/vision_infer.py --image /absolute/path/image.jpg \
  --question 'What is shown in the image?' --device cpu --max-new-tokens 32
```

GPU空闲时可用 `--device cuda:0`（FP16）；CPU使用FP32。输入图像默认限制到最多128个视觉token对应的像素预算，适合小规模检查。正式实验应固定并记录分辨率预算。

重新下载/断点恢复：`.conda-env/bin/python exp/download_vision_model_ranges.py`。

已完成本地GPU FP16加载及自然/医学各一张图像的生成检查，记录见 `vision_smoke_test.json`（仅流程检查，不是准确率评测）。权重中实际存储参数数为2,208,985,600。
