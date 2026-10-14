> 已完成核验：最终 T5/T6/T8 为 GeoQuestions1089、GeoChat、MME-Finance。下文保留候选研究过程；最终结论见 [核验报告](DATASET_VERIFICATION_20261014.md)。

# 真实视觉任务修订记录（2026-10-14）

用户指出：CUAD 合同页及 TAT-DQA 文档页主要依赖文字读取，不能满足本轮真正视觉理解的意图。法学领域允许整体替换，优先考察地理／遥感；艺术文化保留。预算仍为每任务训练 100／测试 50，seed 42，无验证集，训练尚不恢复。

| 任务 | 状态 | 候选及提出者 | 提出／实际使用依据 | 未完成的核验 |
|---|---|---|---|---|
| T5 地理文本 | 候选 | GeoQuestions1089；Kefalidis 等（ISWC 2023） | [官方问答、查询与原始答案](https://github.com/AI-team-UoA/GeoQuestions1089)；[GeoQA2 后续正式评测（2024）](https://doi.org/10.1016/j.jag.2024.104203)；[GeoBenchLLM 数据卡明确使用](https://huggingface.co/datasets/rfr2003/GeoBenchLLM)（该后续论文发表状态尚未核实） | 本来是地理知识图谱问答，包含 WKT 几何及实体集合答案；不能把它未经说明冒称普通段落阅读问答。核验数值／实体答案、论文的直接问答用法、可用样本数及划分方法。不能改写或生成地理答案 |
| T6 遥感图像 | 候选 | GeoChat_Instruct；Kuckreja、Danish、Naseer、Das、Khan、Fahad Shahbaz Khan（CVPR 2024） | [官方论文与源码](https://github.com/mbzuai-oryx/GeoChat)；[TEOChat（Irvin 等，ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ac3af725ae398b6184faae0828bdbd6c-Abstract-Conference.html)，训练使用 GeoChat 指令，正文 3.1.1 与训练数据附录 | 混合数据含分类与定位，不能整包当开放 QA；原始混合 JSON、候选行和图像 ZIP 必须逐项对齐。之前 117218 个候选行并非已核验的最终任务；不能据此宣称准备完毕 |
| T8 金融图表 | 候选 | FinChart-Bench；Dong Shu、Haoyang Yuan、Yuchen Wang、Yanguang Liu、Huopu Zhang、Mengnan Du（ACL 2026） | [正式论文](https://aclanthology.org/2026.acl-long.615/)；[官方数据](https://huggingface.co/datasets/Tizzzzy/FinChart-Bench)；[MoCA-Agent（2026）实际评测](https://arxiv.org/abs/2606.11537)，目前仅核实预印本，不写成正式会议发表 | 仅考虑作者 QA 任务，排除 MC、TF；HF 自动 imagefolder 标签不是答案。核验原始问答文件、图像 ID、分组隔离及作者／后续论文的划分；更多正式论文复用依据仍需核实 |

资源迁移独立继续。新服务器 `/data1/liyapeng/program/LLMCL`，Tailscale `100.87.141.22:8666` 已验证。95 GiB GeoChat 原始分卷 ZIP 暂不整包迁移，旧端保持可用；若采用遥感任务，优先仅准备实验实际需要的图像与原始问答，不重复搬运整个压缩包。

原 CUAD/TAT-DQA 配置与样本仅保留历史依据，README 已标为停用／待修订。`execution_policy.json` 的 `training_enabled=false`，启动队列和正式训练均检查此标志，避免意外重跑旧任务。

## 下载更新

三个候选数据的原始资源已准备至新服务器，详见 [实际下载进度](DOWNLOAD_PROGRESS_20261014.md)；物理资源完成与正式任务通过核验分别记录。文本子集、图像任务语义及论文采样依据未全部完成，训练仍关闭。
