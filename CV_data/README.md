# 视觉问答数据

这里保存**真实图片+问题+参考答案**，用于下一步视觉语言模型实验。与正在运行的纯文本医学续训分开。

| 目录 | 数据 | 范围 |
|---|---|---|
| `natural/vqav2/` | VQAv2 + COCO自然场景图像 | 1700条/1700张先导子集：train1000/dev200/test500 |
| `medical/vqa_rad/` | VQA-RAD放射影像问答 | 原始公开版本2248条问答、315张图像；包含开放短回答和封闭问答 |

每个目录包含 `images/`、`train.jsonl`、`dev.jsonl`、`test.jsonl`、`manifest.json`、`image_manifest.json`、`raw/`。原始文件保留；manifest记录来源、划分、文件SHA256，image_manifest记录尺寸和逐图SHA256。顶层 `preparation_status.json` 只有在全部图片可解码、划分无图片交叉时才会标记completed。

JSONL使用统一字段：`id/domain/dataset/image/question/answer/references/split`，图片路径相对CV_data目录。VQAv2保留10个人工答案及原始question_id/image_id；医学数据保留器官、题型和病例URL。

## 划分

VQAv2：seed42按图片hash选取；每张图选原始question_id最小的一道题。train/dev来自官方train2014，test来自官方val2014，三者图片不重叠。它是本地小规模子集，不是完整VQAv2或官方test集。

VQA-RAD：按同图/同病例URL连通分组后，以seed42固定70%/10%/20%组比例划分。该协议避免关联图片或病例分散到不同划分，但不是文献常见的question-level官方划分，不直接比较其论文分数。实际每个划分条数见manifest。

## 来源

- [VQAv2官方](https://visualqa.org/download.html)；原始问答zip来自该页列出的cvmlp S3。
- [COCO](https://cocodataset.org/)：图片来自官方images.cocodataset.org存储桶，使用HTTPS S3路径，保留图片原始格式。
- [VQA-RAD作者公开库](https://osf.io/89kps/)：原始JSON与315张JPEG。此版本2248条，不混入第三方重组版本。

重新准备：`.vision-env/bin/python exp/prepare_cv_data.py`，会复用已下载且可解码的图片。下载脚本与日志可保留恢复进度。

视觉模型见 `../model/README.md`。Qwen2-VL 九任务训练已完成，含图像实验结果与实例见 [summary_cv](../summary_cv/README.md)。

当前VQA-RAD划分：train1594条/220张、dev231条/31张、test423条/63张。原始315张图片全部保留，问答实际引用314张，其余1张不进入划分。
