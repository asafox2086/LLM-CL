# ChartQA

领域：图表／数据与数量推理。任务：图表文字、数值计算与自由问答，排除 yes/no。

关系：补充领域数据，未声称在这 29 篇中使用。对应方法：无（领域扩展）。

下载来源：[HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)；原始发布来源：https://github.com/vis-nlp/ChartQA。
固定 revision：`b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`；下载字节数：964,096,451；上游许可声明：`gpl-3.0`。完整逐文件哈希见 `asset_manifest.json`。

范围：公开 train / val / test；领域扩展，不伪称在这 29 篇中直接使用。

## 使用

`raw/` 保留所选上游完整文件；`images/` 为实际图像，`open_ended/*.jsonl` 为训练用开放式问答。JSONL 图像路径相对于本数据集文件夹。原始文件中混合的封闭问题仍保留用于溯源，训练请使用 `open_ended/`。

| 划分 | 开放式问答条数 |
|---|---|
| test | 2,396 |
| train | 27,703 |
| validation | 1,864 |

导出图像 20,778 张。过滤统计：`{'boolean_yes_no': 754, 'missing_answer': 2}`。

No boolean/yes-no answers, no upstream CLOSED tags, no option-letter prompts, no GQA choose/verify questions. Short textual and numeric answers remain generation targets.

上游划分默认保留；法律合同按 CIK 公司分组确定性 80/10/10 划分，避免同公司合同跨训练／评估。这里提供的是数据资源，未执行模型训练，不能由下载成功推断模型效果。
