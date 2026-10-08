# LLMCL：统一基座上的语言模型持续学习实验

**Qwen2-VL-2B 含图像的九任务实验已完成**：结果、领域和任务分数见 [summary_cv/README.md](summary_cv/README.md)，每个任务及图像答案类别的真实示例见 [summary_cv/examples.md](summary_cv/examples.md)。T5-Large 的语言及医学续训结果保存在 [summary/](summary/README.md)。

本仓库整理持续学习论文、官方实现和数据，目标是在同一 Llama 基座上比较不同方法：基座初测全部任务，依次学习任务，每学完一个任务重新测试全部任务，再汇总学习、遗忘和迁移表现。

## 当前状态

- 已归档六篇论文及源码，原 `code/` 已重命名为 `source_code/`。
- 完整 SuperNI 已下载并校验：1,613 个任务、5,040,134 条实例；原始数据目录已加入 `.gitignore`。
- 用户已确定七个生成任务，每任务 1000 条训练、200 条验证、500 条测试，直接从完整官方数据划分。SAPT 的现成样本划分仅作参考。
- MixLoRA 暂不开展实验，保留论文和原始源码。
- 四种 LoRA 方法的 T5-Large 语言实验、医学续训以及 Qwen2-VL 九任务实验均已完成；包含参数、完整预测和恢复 checkpoint。

当前实验设置见 [rules/001_experiment_protocol.md](rules/001_experiment_protocol.md)，数据版本及划分指纹见 [rules/001_data_manifest.json](rules/001_data_manifest.json)。

指标定义、得分计算示意和七个任务的真实测试示例见 [summary/README.md](summary/README.md)。

## 目录构造

```text
LLMCL/
├── README.md                         仓库说明
├── .gitignore                        大型原始数据忽略规则
├── paper/                            方法及医学持续学习论文 PDF
├── model/                            T5、Phi、Llama 与 Qwen2-VL 权重
├── summary_cv/                       含图像实验的汇总、评分和实例
├── arxiv_2603.12658v2.pdf             额外参考论文
├── source_code/                      官方源码归档，保留许可证
│   ├── MAC-src/
│   ├── MixLoRA-src/                   暂不纳入实验
│   ├── O-LoRA-src/
│   ├── O-LoRA-language-src/           正确的语言模型 O-LoRA；上行实际为视觉 Online-LoRA
│   ├── ProgressivePrompts-src/
│   ├── SAPT-src/
│   └── UNLOCK-MIGU-src/
├── code/                             olora.py、migu_lora.py、sapt_lora.py、seq_lora.py
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
│   ├── README.md                      执行入口、参数、技术适配和验证记录
│   ├── PROTOCOL_DRAFT.md             指向当前协议的迁移说明
│   ├── configs/olora_first.json       首轮完整数据配置
│   ├── common.py                     数据校验、划分、编码、计分
│   ├── run_olora.py                   单个训练/评价作业
│   ├── launch_olora.py                持续学习主线调度及结果汇总
│   ├── run_olora.sh                   一条命令启动 O-LoRA
│   ├── run_migu_lora.sh               一条命令启动 MIGU-LoRA
│   ├── run_sapt_lora.sh               一条命令启动 SAPT-LoRA
│   ├── run_seq_lora.sh                一条命令启动 SeqLoRA 基础对照
│   ├── requirements.txt              运行依赖
│   └── result/                       待生成：预测、矩阵、配置、指标
└── summary/                          collect.py、results.csv、results.md
```

`code/` 负责论文方法；公共模型加载、数据读取、训练调度和评价由 `exp/` 提供。当前 O-LoRA 已有启动入口，完成后写入 `exp/result/`，由 `summary/` 生成 CSV 和 Markdown 表。其他论文尚未整合。

空目录不会被 Git 保存；新克隆后可执行 `mkdir -p code exp/result summary`。

## 数据和方法范围

从 [allenai/natural-instructions](https://github.com/allenai/natural-instructions) 固定版本 `55a365637381ce7f3748fa2eac7aef1a113bbb82` 读取 Quoref、XSUM、EVALution、PersonaChat、Reddit TIFU、SciQ、GLUCOSE。

共 7000 条训练、1400 条验证、3500 条测试。所有方法共享固定样本，切换顺序或训练种子不重新划分。确定性划分算法、指纹核验命令和下载命令见实验协议。这是基于官方原始数据的自定义持续学习划分，不是官方跨任务泛化测试划分。

已接入 O-LoRA、MIGU-LoRA、SAPT-LoRA 和 SeqLoRA 基础对照，使用统一基座与七任务协议。技术细节、原论文差异和辅助预算见 [exp/README.md](exp/README.md)，运行汇报见 [summary/run_status.md](summary/run_status.md)。Progressive Prompts、MAC 后续接入，MAC 从文档流问答到本实验的迁移仍需明确；MixLoRA 暂不实施。

主指标为 ROUGE-L 上的 AP、F.Rate、FWT、BWT；问答另报 EM/token-F1。FWT 比较学习前序任务后与初始基座在尚未训练任务上的成绩，仅需一条持续学习主线。结果携带协议、数据、模型、代码、环境指纹，仅合并可比较的运行；失败或缺失显示 N/A，不填零。

## O-LoRA 首轮运行

用户已授权先跑一个论文。首轮固定 order_1、seed 42、1 epoch、NF4 冻结基座/FP16 计算、rank 8、alpha 32、dropout 0.1、正交系数 0.5、学习率 1e-4、有效 batch 16。完整七任务各 1000/200/500，仅运行一条七阶段持续学习主线。它是单顺序单种子首轮结果，不是最终多次重复实验统计。详细设置和对原论文的适配见协议第 13 节。

数据协议已升级为 v2：Reddit TIFU 的 45 条空参考答案在排序前明确排除，更新其划分指纹，其余任务不变。配额仍为 1000/200/500。

使用项目专用的独立 Conda 环境 `.conda-env/`，不继承 `MM` 或用户 site-packages。首次执行 `bash exp/setup_env.sh` 创建 Python 3.10.21 环境，安装 CUDA 11.8 对应 PyTorch 2.6.0 及 `exp/requirements.lock` 中锁定的全部依赖。PyTorch 从官方源安装，其余默认使用清华镜像，可用 `LLMCL_PIP_INDEX_URL` 改源。已有环境不重复创建。启动脚本固定使用 `.conda-env/bin/python`，设置 `PYTHONNOUSERSITE=1` 并清除 `PYTHONPATH`。默认基座为原版 T5-Large，权重与 tokenizer 放在 `model/t5-large/`；源码不能代替权重。

```bash
bash exp/run_olora.sh --prepare-only
bash exp/run_olora.sh --model model/t5-large --gpus 0
.conda-env/bin/python summary/collect.py
```

已在本机配置 Hugging Face 授权登录时，可省略 `--model` 自动下载并锁定 revision。token 不写进仓库或命令行参数。模型不可访问时会记录 blocked_model 并退出，不会伪装成已训练。

需要登录时在服务器执行 `.conda-env/bin/huggingface-cli login`，交互输入已获得该模型访问资格的 read token。不要把 token 发到聊天中。也可以直接提供已有权重目录。

一条主线使用 `--gpus` 指定的第一张 GPU（默认 0），不启动单任务对照。运行日志在该次结果目录的 `logs/`，状态为 `status.json`。默认通过 nohup + setsid 后台运行，关闭终端不影响实验。每次新建 run_id；传入 `--resume 原运行目录` 可恢复训练和评价，详见 [恢复说明](exp/README.md#后台运行与断点恢复)。汇总排除 smoke 功能测试，不把小模型成绩作为论文结果。

医学续训（2026-10-07）：从四方法已完成的 T5-Large 七任务 checkpoint 继续学习 MTS-Dialog 与 IU X-ray 两个生成任务。参见 [实时结果表](summary/medical_continuation.md)、[续训协议](rules/003_medical_continuation.md)及 [已下载论文](paper/medical_papers.md)。
