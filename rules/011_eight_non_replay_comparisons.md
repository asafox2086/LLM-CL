# 八种非回放论文方法的执行设置

2026-10-14 用户要求：从原二维表的持续微调正则化、架构方法中选择八种；Frozen/SeqLoRA 为额外对照，不计入八种。
本轮不训练历史 QA 回放。DGG 原作者流程使用 replay_data/task*_with_replay.json，移出本轮主比较，文献表和源码保留。

| 方法 | 分类 | 发表 | 固定作者源码版本 | 当前适配状态 |
|---|---|---|---|---|
| SMoLoRA | architecture | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) | `ba50e074f7468e18f0e6d512360acebe8e213424` | 正式实验运行中 |
| PASs-MoE | architecture | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) | `eb0d571173fcaa21db5f7231d63a64f0db7529c8` | 作者机制/恢复 CPU 检查通过，等待完整 2B GPU 预检 |
| Hyper-LLaVA | architecture | [ICML 2026](https://proceedings.mlr.press/v306/xu26cj.html) | `ea4fb8a7871e524607d87e684b2b6ac58d7cf688` | 作者源码已获取，Qwen2-VL 十任务适配未完成 |
| PaRSP | regularization | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) | `00685fe4f9086f03ac1d1a725dcb280714e6125a` | 作者源码已获取，Qwen2-VL 十任务适配未完成 |
| OSFT | regularization | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) | `fd5b552177bde9a551d30e73660d6a96205c97c0` | 作者源码已获取，Qwen2-VL 十任务适配未完成 |
| SpaRTA | regularization | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) | `933a8f6ff590aedbba8f49475a3a3f2595e389cf` | 作者源码已获取，Qwen2-VL 十任务适配未完成 |
| FV-guided Training | regularization | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html) | `f9f219450c6b7fdadc7b62d89feeeab9aa93968a` | 作者源码已获取，Qwen2-VL 十任务适配未完成 |
| ALER | regularization | [ICML 2026](https://proceedings.mlr.press/v306/xia26j.html) | `ac278e35bb24008fa03604577861765666002970` | 作者源码已获取，Qwen2-VL 十任务适配未完成 |

全部沿用 [010](010_equal_budget_execution.md) 的五领域、十任务、2000/100/200、seed 42、阶段保存及 AP/ForgetRate/FWT/BWT。
均是新基座适配实验；只有保持作者完整方法机制并通过恢复/数值/显存预检后才启动，不用普通 LoRA 换名补足数量。

PASs-MoE 保留作者完整专家容器与 forward、隐式 A 能量路由、rank 能量 EMA、重要性 top-p=0.4、clip=10、稳定项 1e-4、B 稳定项缩放 0.1、30% warmup 和逐任务参考聚合；专家数 2、各 rank=128、alpha=256、gate_tau=1.5→0.7 cosine。
训练前向开启能量记录，反向重算关闭记录，按作者 post_backward 修改梯度。其 BF16 路由的 -1e9 mask 在本机采用 FP32 路由/专家算术；冻结基座仍 FP16。
AdamW 动量张量存 CPU，逐参数执行原生 AdamW 更新；CUDA 上输出逐位一致及恢复检查通过。原始 QA、图像和采样 manifest 不变。
所有重要性、参考权重、EMA、warmup 计数和 gate_tau 保存进完整阶段 training_state.pt；不重复持久化作者的 latest/task 副本。
PASs-MoE checkpoint 实体存放在用户拥有的 `/var/tmp/liyapeng_grp_llmcl_20261012/PASs_MoE/`，仓库标准 runs 路径用软链接访问，避免 /data2 的 700 GiB 用户配额耗尽。队列检查至少 60 GiB 空间。
额外方法队列：`exp/ten_task_cl_20261012/comparison_queue.py`；PID/状态、各方法具体阶段见 `summary/ten_task_cl_20261012/comparison_status.json`。其余六种尚未排队训练。
