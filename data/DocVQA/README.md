# DocVQA

领域：文档／商业与行政材料。任务：文档图像开放式信息问答。

关系：补充领域数据，未声称在这 29 篇中使用。对应方法：无（领域扩展）。

下载来源：[lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)；原始发布来源：https://www.docvqa.org/datasets/docvqa。
固定 revision：`539088ef8a8ada01ac8e2e6d4e372586748a265e`；下载字节数：8,448,509,785；上游许可声明：`apache-2.0`。完整逐文件哈希见 `asset_manifest.json`。

范围：公开有标注 train / validation；一般文档数据，不等同法律数据。

## 使用

`raw/` 保留所选上游完整文件；`images/` 为实际图像，`open_ended/*.jsonl` 为训练用开放式问答。JSONL 图像路径相对于本数据集文件夹。原始文件中混合的封闭问题仍保留用于溯源，训练请使用 `open_ended/`。

| 划分 | 开放式问答条数 |
|---|---|
| train | 39,346 |
| validation | 5,301 |

导出图像 11,466 张。过滤统计：`{'missing_answer': 41, 'multiple_choice_prompt': 4, 'boolean_yes_no': 120}`。

No boolean/yes-no answers, no upstream CLOSED tags, no option-letter prompts, no GQA choose/verify questions. Short textual and numeric answers remain generation targets.

上游划分默认保留；法律合同按 CIK 公司分组确定性 80/10/10 划分，避免同公司合同跨训练／评估。这里提供的是数据资源，未执行模型训练，不能由下载成功推断模型效果。
