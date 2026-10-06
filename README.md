# LLMCL：统一基座上的语言模型持续学习实验

本仓库整理持续学习论文、官方实现和数据，目标是在同一 Llama 基座上比较不同方法：基座初测全部任务，依次学习任务，每学完一个任务重新测试全部任务，再汇总学习、遗忘和迁移表现。

## 当前状态

- 已归档六篇论文及源码，原 `code/` 已重命名为 `source_code/`。
- 完整 SuperNI 已下载并校验：1,613 个任务、5,040,134 条实例；原始数据目录已加入 `.gitignore`。
- 用户已确定七个生成任务，每任务 1000 条训练、200 条验证、500 条测试，直接从完整官方数据划分。SAPT 的现成样本划分仅作参考。
- MixLoRA 暂不开展实验，保留论文和原始源码。
- 方法整合、训练入口、汇总脚本尚未实现，没有训练结果。基座精度、方法参数等仍需在实施前锁定。

当前实验设置见 [rules/001_experiment_protocol.md](rules/001_experiment_protocol.md)，数据版本及划分指纹见 [rules/001_data_manifest.json](rules/001_data_manifest.json)。

## 目录构造

```text
LLMCL/
├── README.md                         仓库说明
├── .gitignore                        大型原始数据忽略规则
├── Paper/                            六篇方法论文 PDF
├── arxiv_2603.12658v2.pdf             额外参考论文
├── source_code/                      官方源码归档，保留许可证
│   ├── MAC-src/
│   ├── MixLoRA-src/                   暂不纳入实验
│   ├── O-LoRA-src/
│   ├── ProgressivePrompts-src/
│   ├── SAPT-src/
│   └── UNLOCK-MIGU-src/
├── code/                             待实现：每篇论文一个方法文件
├── data/
│   ├── README.md                     数据来源和版本
│   ├── SuperNI_full/                 完整官方数据，Git 忽略
│   ├── SAPT_CL_Benchmark/            原论文数据，参考用途
│   ├── SuperNI_configs/              原论文配置，参考用途
│   ├── Long_Sequence_configs/        分类基准配置，参考用途
│   └── MAC_datasets/                 MAC 原生问答数据和配置
├── rules/
│   ├── 001_experiment_protocol.md    实验设置和复现规范
│   └── 001_data_manifest.json        源数据和固定划分指纹
├── exp/
│   ├── PROTOCOL_DRAFT.md             指向当前协议的迁移说明
│   └── result/                       待生成：预测、矩阵、配置、指标
└── summary/                          待实现：汇总脚本和比较表
```

`code/` 负责论文方法；公共模型加载、数据读取、训练调度和评价由 `exp/` 提供。每个方法未来有一个启动入口，完成后写入 `exp/result/`，再由 `summary/` 读取结果生成 CSV 和 Markdown 表。当前没有可用的训练命令，目录存在不代表实现已完成。

空目录不会被 Git 保存；新克隆后可执行 `mkdir -p code exp/result summary`。

## 数据和方法范围

从 [allenai/natural-instructions](https://github.com/allenai/natural-instructions) 固定版本 `55a365637381ce7f3748fa2eac7aef1a113bbb82` 读取 Quoref、XSUM、EVALution、PersonaChat、Reddit TIFU、SciQ、GLUCOSE。

共 7000 条训练、1400 条验证、3500 条测试。所有方法共享固定样本，切换顺序或训练种子不重新划分。确定性划分算法、指纹核验命令和下载命令见实验协议。这是基于官方原始数据的自定义持续学习划分，不是官方跨任务泛化测试划分。

计划接入 O-LoRA、Progressive Prompts、SAPT-LoRA、UNLOCK/MIGU、MAC，建议增加 SeqLoRA 对照。保留各论文核心机制，记录统一基座适配，尤其需落实 MAC 从文档流问答到本实验的迁移。

主指标为 ROUGE-L 上的 AP、F.Rate、FWT、BWT；问答另报 EM/token-F1。FWT 需要真实单任务训练对照。结果携带协议、数据、模型、代码、环境指纹，仅合并可比较的运行；失败或缺失显示 N/A，不填零。实施训练流程前仍需按用户要求确认尚未确定的训练设置。
