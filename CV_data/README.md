# 图像问答：示例与效果

自然场景与医学图像都要求模型看图后用文字回答问题。医学数据包括开放短回答和 yes/no 等封闭问答。

## 先看两种图像任务

| 任务 | 输入节选 | 参考答案节选 | SeqLoRA 最终回答节选 |
|---|---|---|---|
| 自然图像问答（VQAv2） | Answer the question briefly using the image. Is it cold outside? | no | yes |
| 医学图像问答（VQA-RAD） | Answer the question briefly using the image. Are the lungs normal appearing? | No | No |

![自然图像问答（VQAv2）](../sample/natural_vqav2_COCO_val2014_000000388829.jpg)

![医学图像问答（VQA-RAD）](../sample/medical_vqa_rad_synpic29265.jpg)

问题、参考和模型回答保持原始英文；这是固定测试顺序的真实例子。自然图像的 yes/no、number、other 和医学的 OPEN/CLOSED 各类完整示例见 [实例报告](../summary_cv/examples.md)。

## 大体效果

四方法最终在自然图像任务的 EM 约为 83.5%–86.0%，医学图像任务约为 45.5%–49.5%。在这轮子集上，医学图像问答的匹配比例较低。各方法、答案类别和医学部位的得分见 [summary_cv](../summary_cv/README.md)。

## 指标定义与范围

EM 为答案归一化后的完全匹配比例；ROUGE-L 为与参考答案的最长公共子序列 F1；Token F1 为词覆盖的调和平均，均按 0–100 报告。VQAv2 有多个人工答案，本实验每项取最大匹配分；这里的 EM 不能当作官方 VQA 共识准确率。

实际训练使用每任务 1,000 条训练、100 条验证、200 条测试，训练 1 轮，与语言阶段统一。下面的目录容量是准备的数据总量，正式实验只取固定子集。VQAv2 测试来自官方验证集；VQA-RAD 按图像/病例分组划分，因此不直接比较论文官方测试分数。

## 数据准备、来源与实现


这里保存真实图片、问题和参考答案，Qwen2-VL 九任务实验已经完成。

| 目录 | 数据 | 范围 |
|---|---|---|
| `natural/vqav2/` | VQAv2 + COCO自然场景图像 | 1700条/1700张先导子集：train1000/dev200/test500 |
| `medical/vqa_rad/` | VQA-RAD放射影像问答 | 原始公开版本2248条问答、315张图像；包含开放短回答和封闭问答 |

每个目录包含 `images/`、`train.jsonl`、`dev.jsonl`、`test.jsonl`、`manifest.json`、`image_manifest.json`、`raw/`。原始文件保留；manifest记录来源、划分、文件SHA256，image_manifest记录尺寸和逐图SHA256。顶层 `preparation_status.json` 只有在全部图片可解码、划分无图片交叉时才会标记completed。

JSONL使用统一字段：`id/domain/dataset/image/question/answer/references/split`，图片路径相对CV_data目录。VQAv2保留10个人工答案及原始question_id/image_id；医学数据保留器官、题型和病例URL。

### 划分

VQAv2：seed42按图片hash选取；每张图选原始question_id最小的一道题。train/dev来自官方train2014，test来自官方val2014，三者图片不重叠。它是本地小规模子集，不是完整VQAv2或官方test集。

VQA-RAD：按同图/同病例URL连通分组后，以seed42固定70%/10%/20%组比例划分。该协议避免关联图片或病例分散到不同划分，但不是文献常见的question-level官方划分，不直接比较其论文分数。实际每个划分条数见manifest。

### 来源

- [VQAv2官方](https://visualqa.org/download.html)；原始问答zip来自该页列出的cvmlp S3。
- [COCO](https://cocodataset.org/)：图片来自官方images.cocodataset.org存储桶，使用HTTPS S3路径，保留图片原始格式。
- [VQA-RAD作者公开库](https://osf.io/89kps/)：原始JSON与315张JPEG。此版本2248条，不混入第三方重组版本。

重新准备：`.vision-env/bin/python exp/prepare_cv_data.py`，会复用已下载且可解码的图片。下载脚本与日志可保留恢复进度。

视觉模型见 `../model/README.md`。Qwen2-VL 九任务训练已完成，含图像实验结果与实例见 [summary_cv](../summary_cv/README.md)。

当前VQA-RAD划分：train1594条/220张、dev231条/31张、test423条/63张。原始315张图片全部保留，问答实际引用314张，其余1张不进入划分。
