# SLAKE

领域：医学／多器官多影像模态。任务：中英双语医学图像问答；准备版仅保留 answer_type=OPEN。

关系：补充领域数据，未声称在这 29 篇中使用。对应方法：无（领域扩展）。

下载来源：[BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)；原始发布来源：https://www.med-vqa.com/slake。
固定 revision：`a9083ce6c34ac3ffb17671a605962924d8a8f9e9`；下载字节数：216,665,768；上游许可声明：`cc-by-4.0`。完整逐文件哈希见 `asset_manifest.json`。

范围：完整公开图像与 train / validation / test；补充医学域，不伪称这 29 篇直接使用。

## 使用

`raw/` 保留所选上游完整文件；`images/` 为实际图像，`open_ended/*.jsonl` 为训练用开放式问答。JSONL 图像路径相对于本数据集文件夹。原始文件中混合的封闭问题仍保留用于溯源，训练请使用 `open_ended/`。

| 划分 | 开放式问答条数 |
|---|---|
| test | 1,249 |
| train | 5,924 |
| validation | 1,249 |

导出图像 642 张。过滤统计：`{'upstream_closed_type': 5561, 'missing_answer': 44, 'missing_question_or_answer': 1}`。

No boolean/yes-no answers, no upstream CLOSED tags, no option-letter prompts, no GQA choose/verify questions. Short textual and numeric answers remain generation targets.

上游划分默认保留；法律合同按 CIK 公司分组确定性 80/10/10 划分，避免同公司合同跨训练／评估。这里提供的是数据资源，未执行模型训练，不能由下载成功推断模型效果。
