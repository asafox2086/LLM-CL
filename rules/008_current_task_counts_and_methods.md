# 当前十任务样本量与论文源码方案

> 当前执行以 [010 等量、单 seed 协议](010_equal_budget_execution.md) 为准：十任务均为 2000/100/200，只用 seed 42。本文件旧预算/状态是历史准备记录。

核验日期：2026-10-12。本文件更新 006 中的数据量和方法选择；007 的领域与真实性要求继续有效。没有启动正式训练。训练预算和当前方法选择由 [009 小预算协议](009_small_budget_no_replay_baselines.md) 覆盖。

## 1. 当前源划分计数

单位为问答样本，不是图片、token 或唯一知识点数。下表记录服务器现有公开数据/开放式派生视图；不是宣称十任务最终清单已全部锁定。跨数据集重叠、测试集角色和输入预算仍需核查。

| 阶段 | 领域 / 数据 | 源训练集 | 源验证集 | 源测试集 | 当前状态 |
|---|---|---:|---:|---:|---|
| T1 | 自然/通用文本：SQuAD 1.1 | 87,599 | 10,570 | 无公开 test 标签 | 原始官方 train/dev 已下载；领域范围和最终开发/测试角色待锁定。旧人工 General 子集不用。 |
| T2 | 自然图像：VQAv2 + GQA + TextVQA | 793,137 | 202,562 | 仅 GQA testdev 6,924 | 现有开放式视图；另外两集无本地带标签 test，不能把 6,924 当整个 T2 test。跨集同图及各官方 validation 的角色待锁定。 |
| T3 | 医学文本：MedQuAD | 13,125 | 1,641 | 1,641 | 已按 K-COMP 的随机 80/10/10 思路建立固定 seed 42 版本；使用 16,407 条公开非空回答。不是该论文逐条精确复现。 |
| T4 | 医学图像：VQA-RAD + PathVQA + SLAKE | 16,675 | 4,383 | 4,806 | 开放式视图按来源统计；VQA-RAD 镜像只有 train/test，4,383 验证样本不含 RAD。SLAKE 保留上游双语问答。 |
| T5 | 法律文本：CUAD | 5,458 | 未单独划分 | 1,244 | 原始有答案问题的生成视图对应官方 408 个训练合同、102 个测试合同；多个必要答案 span 合在一个问题中，不按 span 重复计数。 |
| T6 | 法律图像 | 未锁定 | 未锁定 | 未锁定 | 510 份 CUAD 原始 PDF 已下载，但尚无已锁定的端到端 VLM 图像训练协议；不能用 PDF 数冒充 QA 数。 |
| T7 | 金融文本：FinQA | 6,083 | 861 | 1,111 | 官方划分中的非布尔、非空回答视图，共 8,055 条；原始文件未变。 |
| T8 | 金融图像：TAT-DQA | 13,248 | 1,644 | 1,663 | 官方划分开放回答视图，共 16,555 条；原始集 16,558，3 条封闭/空回答未进入视图。 |
| T9 | 艺术文本候选：AQUA knowledge | 候选 40,244 | 候选 3,617 | 候选 3,642 | 官方 knowledge 题数量，共 47,503；独立文本输入协议未锁定，不能算已就绪 T9。 |
| T10 | 艺术图像候选：AQUA visual | 候选 29,568 | 候选 1,507 | 候选 1,270 | 官方 visual 题数量，共 32,345；语义/开放性核验和图像下载尚未完成，不能算已就绪 T10。 |

T2 训练量分解：VQAv2 266,660 + GQA 494,675 + TextVQA 31,802。验证分别 128,856 / 69,083 / 4,623。

T4 训练量分解：VQA-RAD 852 + PathVQA 9,899 + SLAKE 5,924。验证分别 无 / 3,134 / 1,249；测试分别 200 / 3,357 / 1,249。

MedQuAD 的 seed 42 是本项目明确记录的复现选择；K-COMP 文中随机种子与逐条划分没有核实，不能声称完全相同。划分以 QA 为单位，不声称疾病页面或患者完全隔离。输出保留原 JSONL 行字节不变，原始 XML 不改。

CUAD 的 `train_separate_questions.json` 会把多个标注 span 展开为多个带后缀的条目。本表按原始问题 ID 合并计数，因此不能把 11,180 个有答案的展开条目与 5,458 个原始有答案问题混为一谈。源 train ID 的 span 后缀归一化后与 test 不交叉，6,702 个派生问题全部映射到官方 train/test。

详见服务器 `data/TenTaskCL/task_counts_audit.json` 与 `data/MedQuAD/paper_split_seed42/split_manifest.json`。来源与实际使用论文见 [数据集溯源](../data/DATASET_PROVENANCE.md)。

## 2. 方法必须来自论文二维表

本轮主要方法从 `paper/pre_finetuning/` 中选择，采用可验证作者源码作为适配依据。此前 MIGU-LoRA、O-LoRA、SAPT-LoRA 只保留为历史实验，不继续作为这轮主要比较方法。SeqLoRA 与 Frozen 作为基础对照仍保留。

第一批优先适配以下五个方法，**尚未运行十任务**：

| 方法 | 论文 / 分类 | 作者源码 | 适配重点 |
|---|---|---|---|
| DGG | CVPR 2026 / 正则化 | [lisongze/DGG](https://github.com/lisongze/DGG) | 保留历史梯度近似、Bernoulli 调度；作者修改版 Transformers/DeepSpeed 按 gitlink 版本获取。原实现单卡大显存要求不能直接套到 11GB 卡。 |
| PaRSP | ACL 2026 / 正则化 | [JinhuiBot/PaRSP](https://github.com/JinhuiBot/PaRSP) | 保留区域和零空间约束，核验 Qwen2-VL 层映射、激活统计与多模态输入。 |
| SMoLoRA | ICCV 2025 / 架构 | [Minato-Zackie/SMoLoRA](https://github.com/Minato-Zackie/SMoLoRA) | 保留视觉/语言专家和路由；核验纯文本阶段，不能用伪图像替代文本任务。 |
| PASs-MoE | ACL 2026 / 架构 | [yueluoshuangtian/PASs-MoE](https://github.com/yueluoshuangtian/PASs-MoE) | 保留通路激活、路由重加权和秩稳定约束，核验无图像输入。 |
| Hyper-LLaVA | ICML 2026 / 架构 | [ICML2026-Hyper-LLaVA](https://github.com/zhoujiahuan1991/ICML2026-Hyper-LLaVA) | 保留双曲不确定性和模态平衡路由；纯文本输入与模型替换作为明确复现扩展记录。 |

采用作者源码不等于可以直接执行原脚本：原论文多使用 LLaVA/Vicuna、BLIP-2 或纯文本 LLM；统一 Qwen2-VL-2B 的十任务实验属于适配后的公平比较。每个方法需保存原版本、适配 diff、关键机制核验和 train/dev smoke test，再进入正式比较。测试集不参与适配调参。

不能把 006 中统一“只训练 q/v LoRA、冻结全部视觉/merger”套到所有架构方法上，否则可能移除方法的核心机制。SeqLoRA 初始基线可以沿用该设置；各论文方法按核心机制配置可训练模块，统一数据、评测和当前样本曝光，并报告参数量、回放、额外 token 和算力差异。

## 3. 下载与未公开源码

20 篇持续微调方法逐篇建立 `code/<method>/` 独立目录，作者原文件在 `source/`，每目录有来源与版本 manifest。只下载源码、配置、文档；捆绑数据、权重和二进制不重复下载。源码未安装、未执行训练。

CoPE 作者仓库目前只有 README，“code will be release soon”，不算已有实现；不进入第一批主实验。MVP 作者项目页 Code 链接为占位 URL（404）；SLAO 未找到可验证的公开作者源，均如实保留状态。

MaDQ 作者源码已下载，但公开 README 与实现主要对应 BLIP；本轮不能拿该小模型实现冒充论文 BLIP-2 大模型实验。Weight Space Consolidation 作者仓库同时含传统分类代码及 `trace/wsc_llm.py` 桥接模块，本项目只考虑后者，不能把分类脚本用于本轮。FCIT 是联邦方法，不能不说明客户端协议就当普通集中式持续学习方法。

所有方法下载状态、版本、源码范围和原模型见 [code/README.md](../code/README.md) 与 `code/finetuning_source_manifest.json`。只有通过适配核验的方法才会被标成可运行。
