# 持续微调论文作者源码

核验：2026-10-12。对应论文二维表的 **20 篇持续微调方法**，各方法分目录存放。源码在服务器，GitHub 保存索引、版本、哈希及获取脚本；没有安装包或启动正式训练。

已下载 17 个方法的源码快照；2 个未找到作者实现；1 个仓库只有 README。源码下载失败剩余 0 个。

原源码位置为 `code/<method>/source/`。源码、配置和文档按固定 Git commit 保存，捆绑数据、模型权重和二进制不重复下载；逐文件 SHA-256 与 Git blob SHA-1 已核验。作者修改版 DGG 两个依赖按父仓库 gitlink 下载。

第一批按源码核验和适配优先级选择：**DGG、PaRSP、SMoLoRA、PASs-MoE、Hyper-LLaVA**（去掉回放类比较方法；其他方法保留作者原设置）。SeqLoRA/Frozen 作为基础对照；旧 MIGU/O-LoRA/SAPT 不作这轮主要比较方法。适配仍待完成，详见 [方法选择](method_selection.json) 与 [任务和方法协议](../rules/008_current_task_counts_and_methods.md)。

| 方法 | 分类 / 发表 | 作者仓库 | 下载与实现状态 | 复现说明 |
|---|---|---|---|---|
| [CoPE](CoPE/README.md) | architecture / ICML 2026 | [wz7in/COPE](https://github.com/wz7in/COPE) | 仅 README，尚无实现 | 官方仓库目前只有 README；不计入可实现的方法。 |
| [ECA](ECA/README.md) | architecture / ICML 2026 | [Snowball0823/ECA](https://github.com/Snowball0823/ECA) | 作者源码已下载；待适配 | 以视觉对齐/Q-Former 为核心，十任务纯文本阶段需核验。 |
| [Hyper-LLaVA](Hyper_LLaVA/README.md) | architecture / ICML 2026 | [zhoujiahuan1991/ICML2026-Hyper-LLaVA](https://github.com/zhoujiahuan1991/ICML2026-Hyper-LLaVA) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [PASs-MoE](PASs_MoE/README.md) | architecture / ACL 2026 | [yueluoshuangtian/PASs-MoE](https://github.com/yueluoshuangtian/PASs-MoE) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [SLAO / Merge before Forget](SLAO/README.md) | architecture / ICLR 2026 | 未找到作者源码 | 未找到可验证作者源码 | 未找到公开作者源码；不以第三方实现替代。 |
| [Any-SSR](Any_SSR/README.md) | architecture / ICCV 2025 | [ZHUANGHP/Any-SSR](https://github.com/ZHUANGHP/Any-SSR) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [CL-MoE](CL_MoE/README.md) | architecture / CVPR 2025 | [ECNU-ICALK/CL-MoE](https://github.com/ECNU-ICALK/CL-MoE) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [FCIT](FCIT/README.md) | architecture / ICCV 2025 | [Ghy0501/FCIT](https://github.com/Ghy0501/FCIT) | 作者源码已下载；待适配 | 原设定是联邦持续学习，不能直接当集中式方法。 |
| [MVP / Instruction-Grounded Visual Projectors](MVP/README.md) | architecture / ICCV 2025 | 未找到作者源码 | 未找到可验证作者源码 | 作者项目页 Code 为占位 URL /YOUR REPO HERE，404。 |
| [SMoLoRA](SMoLoRA/README.md) | architecture / ICCV 2025 | [Minato-Zackie/SMoLoRA](https://github.com/Minato-Zackie/SMoLoRA) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [ALER](ALER/README.md) | regularization / ICML 2026 | [dakfjalka/aler-distill](https://github.com/dakfjalka/aler-distill) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [DGG](DGG/README.md) | regularization / CVPR 2026 | [lisongze/DGG](https://github.com/lisongze/DGG) | 作者源码已下载；待适配 | 含作者定制 Transformers/DeepSpeed；官方原实现提示单卡 >=80GB，尚未做本机预检。 |
| [OSFT](OSFT/README.md) | regularization / ICLR 2026 | [Red-Hat-AI-Innovation-Team/mini_trainer](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [PaRSP](PaRSP/README.md) | regularization / ACL 2026 | [JinhuiBot/PaRSP](https://github.com/JinhuiBot/PaRSP) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [SpaRTA](SpaRTA/README.md) | regularization / ACL 2026 | [Xnhyacinth/SpaRTA](https://github.com/Xnhyacinth/SpaRTA) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [FV-guided Training](FV_guided_Training/README.md) | regularization / ICLR 2025 | [GangweiJiang/FvForgetting](https://github.com/GangweiJiang/FvForgetting) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [FOREVER](FOREVER/README.md) | replay / ACL 2026 | [WoodScene/FOREVER](https://github.com/WoodScene/FOREVER) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |
| [MaDQ / UCo-VQA](MaDQ/README.md) | replay / CVPR 2026 | [Zi-Jian-Gao/MaDQ](https://github.com/Zi-Jian-Gao/MaDQ) | 作者源码已下载；待适配 | 公开代码主要为 BLIP；不能冒充论文 BLIP-2 大模型扩展。 |
| [Weight Space Consolidation](Weight_Space_Consolidation/README.md) | replay / ICLR 2026 | [umamicode/weight-space-consolidation](https://github.com/umamicode/weight-space-consolidation) | 作者源码已下载；待适配 | 只考虑 trace/wsc_llm.py 等 LLM 桥接代码；不是完整 TRACE 训练器。 |
| [SERS](SERS/README.md) | replay / NeurIPS 2025 | [JerryWangJun/LLM_CL_SERS](https://github.com/JerryWangJun/LLM_CL_SERS) | 作者源码已下载；待适配 | 原作者模型和任务需适配；未运行新实验。 |

主索引：[finetuning_source_manifest.json](finetuning_source_manifest.json)。每个方法目录记录完整 commit、源文件数、许可证、获取依据和 SHA-256。尚未公开源码的方法如实记录，不把 README 或第三方重写当作者实现。

复现对同一 2B 基座和十任务的适配需要保留各方法机制。不能统一删掉视觉专家、路由或重要性约束，也不能把回放法替换成普通 ER。新模型上的结果属于适配实验，不能写成原论文原模型的精确复现。
