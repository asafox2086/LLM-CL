# SEC_Contracts_VQA

领域：法律／商业合同。任务：合同页图像信息抽取转开放式问答；排除布尔字段。

关系：补充领域数据，未声称在这 29 篇中使用。对应方法：无（领域扩展）。

下载来源：[chenghao/sec-material-contracts-qa](https://huggingface.co/datasets/chenghao/sec-material-contracts-qa)；原始发布来源：https://huggingface.co/datasets/chenghao/sec-material-contracts-qa。
固定 revision：`1c7df19940c150e4511d9a53bb313bc06fb56c94`；下载字节数：1,413,717,356；上游许可声明：`['apache-2.0']`。完整逐文件哈希见 `asset_manifest.json`。

范围：全部 804 份含页图像合同；基于上游 GPT-4o 提取字段生成确定性 QA 模板，非人工审校金标准；不使用混入 DUDE 的 splitted 版本。

## 使用

`raw/` 保留所选上游完整文件；`images/` 为实际图像，`open_ended/*.jsonl` 为训练用开放式问答。JSONL 图像路径相对于本数据集文件夹。原始文件中混合的封闭问题仍保留用于溯源，训练请使用 `open_ended/`。

| 划分 | 开放式问答条数 |
|---|---|
| test | 571 |
| train | 5,234 |
| validation | 602 |

导出图像 6,280 张。过滤统计：`{'missing_answer': 8065}`。

No boolean/yes-no answers, no upstream CLOSED tags, no option-letter prompts, no GQA choose/verify questions. Short textual and numeric answers remain generation targets.

上游划分默认保留；法律合同按 CIK 公司分组确定性 80/10/10 划分，避免同公司合同跨训练／评估。这里提供的是数据资源，未执行模型训练，不能由下载成功推断模型效果。

**法律数据质量说明**：上游将 SEC 合同转换为真实合同页图像，字段标签由 GPT-4o 自动抽取，未经本次人工逐条审校。这里把非布尔、非空字段转换为固定英文问题；保留整份合同的多页图像，不杜撰单页证据定位。适合领域适配候选数据，不能作为人工标注的权威法律推理金标准。
