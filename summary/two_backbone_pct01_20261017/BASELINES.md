# 当前基线及发表信息

2026-10-18用户最新调整：标准ER-LoRA不再参与本轮比较，正在运行的Qwen ER按授权取消；旧OneVision ER完整结果及所有未完成记录保留历史。移出未进入正式训练的FV-guided Training（ICLR 2025）与ALER（ICML 2026），保留CVPR方法CL-MoE。保留原文献、源码与失败预检记录。当前已准入7种方法；新增FOREVER作为第8种比较，适配与GPU准入待完成，不计作正在运行或已经完成。

| 主要类别 | 方法 | 正式会议及年份 | 本轮状态 |
|---|---|---|---|
| 朴素 | SeqFT-LoRA | 标准顺序微调对照；无独立新方法发表年份 | 保留 |
| 回放 | [FOREVER](https://aclanthology.org/2026.acl-long.1144/) | ACL 2026主会 | 新增比较，待多模态适配及GPU准入 |
| 正则化 | [PaRSP](https://aclanthology.org/2026.acl-long.1244/) | ACL 2026主会 | 保留 |
| 正则化 | [OSFT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) | ICLR 2026主会 | 保留 |
| 正则化 | [SpaRTA](https://aclanthology.org/2026.acl-long.334/) | ACL 2026主会 | 保留 |
| 架构 | [CL-MoE](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) | CVPR 2025主会 | 保留 |
| 架构 | [SMoLoRA](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) | ICCV 2025主会 | 保留 |
| 架构 | [PASs-MoE](https://aclanthology.org/2026.acl-long.1474/) | ACL 2026主会 | 保留；兼有正则机制 |

计划为1回放＋3正则化＋3架构＋1朴素，共8种。当前正式运行范围为7种×2骨干；FOREVER完成准入后才扩展正式范围，不用空分数冒充结果。

## FOREVER的比较定位与准入条件

原论文是LLM持续学习，按实际优化器更新幅度定义模型时间，结合遗忘曲线回放调度及强度感知正则化；包含0.6B模型实验。论文并非原生图文方法，本实验必须标明“FOREVER多模态适配”，不能改名ER或仅实现其部分机制后冒充原方法。

作者仓库：[WoodScene/FOREVER](https://github.com/WoodScene/FOREVER)，已有固定源码版本`bbeb549a21e59d9490f20a2b02099ee766a67f48`，见[源码记录](../../code/FOREVER/README.md)。原作者Qwen脚本包含较多当前任务轮次及额外回放；不能直接运行其脚本并宣称与本轮1%一轮公平。

接入需使用独立、可追溯的扩展入口，不热改活动V2 Python源码；两骨干共享冻结1313 TRAIN/21454 TEST、19阶段、seed42、Answer F1、原图文输入和主任务预算。回放仅来自历史TRAIN，作者额外回放/正则/参数快照与优化器成本单独记录；不得将这些成本隐藏为相同总计算。保留实际调度触发、回放ID/次数、正则激活、loss、真实LR/梯度、参数变化、CPU/GPU前后及峰值与最终checkpoint。两骨干及重载有限梯度门禁通过后，再进入正式排队与完整主表核验。

## 移出的两项

- [FV-guided Training](https://proceedings.iclr.cc/paper_files/paper/2025/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html)：ICLR 2025。Qwen仅有失败预检，未进入正式训练；OneVision未启动。
- [ALER](https://proceedings.mlr.press/v306/xia26j.html)：ICML 2026。Qwen仅有失败预检，未进入正式训练；OneVision未启动。

移除范围包括两机1%及未来全量队列；不删历史预检、论文或源码，不改变其余方法配置、数据划分与已取得分数。
