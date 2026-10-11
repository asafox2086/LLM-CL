# PathVQA

领域：医学／病理影像。任务：病理图像开放式医学问答；准备版过滤 yes/no。

关系：已收集论文使用的数据。对应方法：PASs-MoE。

下载来源：[flaviagiammarino/path-vqa](https://huggingface.co/datasets/flaviagiammarino/path-vqa)；原始发布来源：https://github.com/UCSD-AI4H/PathVQA。
固定 revision：`1685832883334b5bb5beaf4e4b333fdeecaa4ad9`；下载字节数：785,419,242；上游许可声明：`mit`。完整逐文件哈希见 `asset_manifest.json`。

范围：公开 train / validation / test；第三方格式化镜像，保留来源说明。

## 使用

`raw/` 保留所选上游完整文件；`images/` 为实际图像，`open_ended/*.jsonl` 为训练用开放式问答。JSONL 图像路径相对于本数据集文件夹。原始文件中混合的封闭问题仍保留用于溯源，训练请使用 `open_ended/`。

| 划分 | 开放式问答条数 |
|---|---|
| test | 3,357 |
| train | 9,899 |
| validation | 3,134 |

导出图像 4,246 张。过滤统计：`{'boolean_yes_no': 16238, 'missing_answer': 4}`。

No boolean/yes-no answers, no upstream CLOSED tags, no option-letter prompts, no GQA choose/verify questions. Short textual and numeric answers remain generation targets.

上游划分默认保留；法律合同按 CIK 公司分组确定性 80/10/10 划分，避免同公司合同跨训练／评估。这里提供的是数据资源，未执行模型训练，不能由下载成功推断模型效果。
