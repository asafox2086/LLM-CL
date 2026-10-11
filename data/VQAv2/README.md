# VQAv2

领域：自然图像／日常场景。任务：自由生成 VQA 答案，准备版排除 yes/no。

关系：已收集论文使用的数据。对应方法：DGG、ECA、Hyper-LLaVA、CoPE、SMoLoRA。

下载来源：[lmms-lab/VQAv2](https://huggingface.co/datasets/lmms-lab/VQAv2)；原始发布来源：https://visualqa.org/。
固定 revision：`32665d35052eb4a6d4414851c3c829a72754915a`；下载字节数：21,004,436,200；上游许可声明：`cc-by-4.0`。完整逐文件哈希见 `asset_manifest.json`。

范围：完整公开有标注 train / validation；未下载无公开答案的 test/testdev。

## 使用

`raw/` 保留所选上游完整文件；`images/` 为实际图像，`open_ended/*.jsonl` 为训练用开放式问答。JSONL 图像路径相对于本数据集文件夹。原始文件中混合的封闭问题仍保留用于溯源，训练请使用 `open_ended/`。

| 划分 | 开放式问答条数 |
|---|---|
| train | 266,660 |
| validation | 128,856 |

导出图像 118,740 张。过滤统计：`{'boolean_yes_no': 262588, 'missing_answer': 7}`。

No boolean/yes-no answers, no upstream CLOSED tags, no option-letter prompts, no GQA choose/verify questions. Short textual and numeric answers remain generation targets.

上游划分默认保留；法律合同按 CIK 公司分组确定性 80/10/10 划分，避免同公司合同跨训练／评估。这里提供的是数据资源，未执行模型训练，不能由下载成功推断模型效果。
