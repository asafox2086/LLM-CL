# TextVQA

领域：自然场景／文字读取。任务：从真实场景图像读取文字并自由回答。

关系：已收集论文使用的数据。对应方法：DGG、ECA、Hyper-LLaVA、CoPE、SMoLoRA、HPA。

下载来源：[lmms-lab/textvqa](https://huggingface.co/datasets/lmms-lab/textvqa)；原始发布来源：https://textvqa.org/。
固定 revision：`9c0699cd19768ac5ab97568f6b3cbac4c0062884`；下载字节数：7,135,034,165；上游许可声明：`None`。完整逐文件哈希见 `asset_manifest.json`。

范围：完整公开有标注 train / validation；无标注 test 不用于训练。

## 使用

`raw/` 保留所选上游完整文件；`images/` 为实际图像，`open_ended/*.jsonl` 为训练用开放式问答。JSONL 图像路径相对于本数据集文件夹。原始文件中混合的封闭问题仍保留用于溯源，训练请使用 `open_ended/`。

| 划分 | 开放式问答条数 |
|---|---|
| train | 31,802 |
| validation | 4,623 |

导出图像 24,211 张。过滤统计：`{'boolean_yes_no': 3176, 'missing_answer': 1}`。

No boolean/yes-no answers, no upstream CLOSED tags, no option-letter prompts, no GQA choose/verify questions. Short textual and numeric answers remain generation targets.

上游划分默认保留；法律合同按 CIK 公司分组确定性 80/10/10 划分，避免同公司合同跨训练／评估。这里提供的是数据资源，未执行模型训练，不能由下载成功推断模型效果。
