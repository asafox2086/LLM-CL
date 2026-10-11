# 多领域图像开放式问答数据

核验日期：2026-10-11。服务器目录：`/data2/liyapeng_grp/program/LLMCL/data/`。每个数据集独立文件夹；原有数据保留。

已下载 **9 个数据集**，导出 **1,111,504 条**开放式问答。原始下载 52.39 GB，另外导出图像 49.24 GB。数字为十进制 GB。

| 数据集目录 | 领域 | 开放问答划分与条数 | 图像数 | 原始文件 GB | 与论文的关系 |
|---|---|---|---|---|---|
| [VQAv2](VQAv2/README.md) | 自然图像／日常场景 | train: 266,660；validation: 128,856 | 118,740 | 21.00 | [DGG](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Multimodal_Continual_Instruction_Tuning_with_Dynamic_Gradient_Guidance_CVPR_2026_paper.html)；[ECA](https://proceedings.mlr.press/v306/kong26h.html)；[Hyper-LLaVA](https://proceedings.mlr.press/v306/xu26cj.html)；[CoPE](https://proceedings.mlr.press/v306/wang26le.html)；[SMoLoRA](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |
| [GQA](GQA/README.md) | 自然图像／关系与组合推理 | testdev: 6,924；train: 494,675；validation: 69,083 | 85,758 | 12.39 | [MaDQ / UCo-VQA](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Re-evaluating_Continual_VQA_Toward_Fair_and_Robust_Evaluation_for_Multimodal_CVPR_2026_paper.html)；[Hyper-LLaVA](https://proceedings.mlr.press/v306/xu26cj.html)；[CoPE](https://proceedings.mlr.press/v306/wang26le.html)；[SMoLoRA](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |
| [TextVQA](TextVQA/README.md) | 自然场景／文字读取 | train: 31,802；validation: 4,623 | 24,211 | 7.14 | [DGG](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Multimodal_Continual_Instruction_Tuning_with_Dynamic_Gradient_Guidance_CVPR_2026_paper.html)；[ECA](https://proceedings.mlr.press/v306/kong26h.html)；[Hyper-LLaVA](https://proceedings.mlr.press/v306/xu26cj.html)；[CoPE](https://proceedings.mlr.press/v306/wang26le.html)；[SMoLoRA](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)；[HPA](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Harmonious_Parameter_Adaptation_in_Continual_Visual_Instruction_Tuning_for_Safety-Aligned_CVPR_2026_paper.html) |
| [ChartQA](ChartQA/README.md) | 图表／数据与数量推理 | test: 2,396；train: 27,703；validation: 1,864 | 20,778 | 0.96 | 领域扩展（未声称这 29 篇使用） |
| [DocVQA](DocVQA/README.md) | 文档／商业与行政材料 | train: 39,346；validation: 5,301 | 11,466 | 8.45 | 领域扩展（未声称这 29 篇使用） |
| [VQA_RAD](VQA_RAD/README.md) | 医学／放射影像 | test: 200；train: 852 | 275 | 0.03 | [PASs-MoE](https://aclanthology.org/2026.acl-long.1474/) |
| [PathVQA](PathVQA/README.md) | 医学／病理影像 | test: 3,357；train: 9,899；validation: 3,134 | 4,246 | 0.79 | [PASs-MoE](https://aclanthology.org/2026.acl-long.1474/) |
| [SLAKE](SLAKE/README.md) | 医学／多器官多影像模态 | test: 1,249；train: 5,924；validation: 1,249 | 642 | 0.22 | 领域扩展（未声称这 29 篇使用） |
| [SEC_Contracts_VQA](SEC_Contracts_VQA/README.md) | 法律／商业合同 | test: 571；train: 5,234；validation: 602 | 6,280 | 1.41 | 领域扩展（未声称这 29 篇使用） |

## 文件结构与任务口径

```text
data/<dataset>/
├── raw/                         # 原始文件与原始许可/卡片
├── images/                      # 可读取的真实图像／文档页图像
├── open_ended/
│   ├── train.jsonl
│   ├── validation.jsonl          # 仅当上游有此划分，或法律分组划分
│   ├── test.jsonl                # 仅当公开有答案
│   └── images_manifest.jsonl
├── asset_manifest.json          # 上游固定版本、逐文件 SHA-256
├── prepared_manifest.json       # 过滤、图像、JSONL 核验与计数
└── README.md
```

训练请使用 `open_ended/`：所有图像已实际保存，JSONL 包含 `question`、`answer`、多参考 `answers`、`images`、原始问题 ID 等字段。图像路径相对于对应数据集文件夹。答案由模型生成文字；没有类别编号或选项字母。保留数字和短文字答案，不能把短答案一概当成分类任务。

原始数据保留用于溯源，部分原始集含 yes/no 等封闭问题。准备版已排除布尔答案、上游 CLOSED 标签、选项字母提示和 GQA choose / verify 问题。SLAKE 仅使用 OPEN 标签，保留中英双语。ScienceQA、AI2D、IconQA 等主要选择题数据没有用于补足本清单。

VQAv2 / TextVQA / DocVQA 下载完整公开有答案的 train / validation；GQA 使用完整 balanced 版本，公开无答案 test 仅保存在 raw/，不编造答案。其余数据集范围和第三方格式化镜像说明见各目录 README。

## 法律与划分说明

SEC_Contracts_VQA 使用真实商业合同的页图像。上游 GPT-4o 抽取的非布尔字段被转换成确定性的开放问题，例如合同生效日期、当事方、管辖法律和付款条款。标注并非人工审校金标准，也不等于法院案例法律推理数据。整份合同的所有页图像随样本保留，未杜撰单页证据。按 CIK 公司分组 80/10/10 划分，已验证公司不跨训练、验证和测试。

其余数据保留上游划分。连续学习实验的任务顺序、预算和指标由实验协议确定。图像重复审计按原始图像字节的 SHA-256 进行，结果见机器索引的 `exact_image_overlap_audit`；它不检测重编码或近似相似图像。

重复图像核验发现：VQA-RAD 的 train/test 共用 103 张图像，SLAKE 的 train/test 共用 142 张，VQAv2 与 GQA 共用 4,265 张。这里保留上游问题划分；如实验要求评估图像未在训练中出现，应另按图像分组划分。其余重复数量见机器索引。

## 来源与复现

所有所选上游文件均通过文件大小和上游 LFS SHA-256／Git blob 校验。每个导出图像通过 Pillow 格式、尺寸和完整性验证；每个 JSONL 逐行验证问题、答案、过滤规则及图像引用。逐文件哈希与固定 revision 可在目录 manifest 中复核。模型索引见 [多模态模型](../model/README.md)。

数据许可按原始发布者条款为准；格式化镜像中的许可声明保留在各目录，不把重新排版视为赋予新的许可。

服务器使用 Python 3.10+，依赖 `requests pyarrow pillow safetensors`。复现脚本：`download_multimodal_assets.py --datasets` 下载并核验所选原始文件；`prepare_multimodal_assets.py` 创建开放式问答版本。下载按 `multimodal_dataset_specs.json` 中的 revision 固定。

GitHub 保存来源索引、固定版本、计数、哈希及整理脚本。大型图像、Parquet 和模型权重保存在上述服务器目录。
