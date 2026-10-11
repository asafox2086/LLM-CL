# GQA

领域：自然图像／关系与组合推理。任务：图像关系、属性、数量问答，排除 yes/no。

关系：已收集论文使用的数据。对应方法：MaDQ / UCo-VQA、Hyper-LLaVA、CoPE、SMoLoRA。

下载来源：[lmms-lab/GQA](https://huggingface.co/datasets/lmms-lab/GQA)；原始发布来源：https://cs.stanford.edu/people/dorarad/gqa/。
固定 revision：`a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8`；下载字节数：12,385,883,700；上游许可声明：`mit`。完整逐文件哈希见 `asset_manifest.json`。

范围：官方 balanced 版本公开 train/val/test/testdev；不重复下载 all 版。

## 使用

`raw/` 保留所选上游完整文件；`images/` 为实际图像，`open_ended/*.jsonl` 为训练用开放式问答。JSONL 图像路径相对于本数据集文件夹。原始文件中混合的封闭问题仍保留用于溯源，训练请使用 `open_ended/`。

| 划分 | 开放式问答条数 |
|---|---|
| testdev | 6,924 |
| train | 494,675 |
| validation | 69,083 |

导出图像 85,758 张。过滤统计：`{'missing_question_or_answer': 95336, 'boolean_yes_no': 383173, 'gqa_choice_or_verification': 133785}`。

No boolean/yes-no answers, no upstream CLOSED tags, no option-letter prompts, no GQA choose/verify questions. Short textual and numeric answers remain generation targets.

上游划分默认保留；法律合同按 CIK 公司分组确定性 80/10/10 划分，避免同公司合同跨训练／评估。这里提供的是数据资源，未执行模型训练，不能由下载成功推断模型效果。

公开 test balanced instructions 无答案，因此只保留在 raw/，不生成带伪答案的训练 JSONL；testdev 使用公开答案。
