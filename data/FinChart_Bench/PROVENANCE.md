# FinChart-Bench：已下载实验预算所需候选资源

提出者：Dong Shu、Haoyang Yuan、Yuchen Wang、Yanguang Liu、Huopu Zhang、Mengnan Du。[ACL 2026 正式论文](https://aclanthology.org/2026.acl-long.615/)，[官方源码](https://github.com/Tizzzzy/FinChart-Bench)，[官方数据](https://huggingface.co/datasets/Tizzzzy/FinChart-Bench)。实际复用：[MoCA-Agent 作者评测代码](https://github.com/UBC-NLP/MoCA-Agent/blob/main/docs/datasets.md)，[2026 论文](https://arxiv.org/abs/2606.11537)，目前只核实预印本，不能写为正式会议发表。

只下载 QA 数据，未采用 MC／TF 分类任务。固定 HF revision d3cea12aa1c5ccc5b5a891d28ce2a165a7ee45ff，官方 QA_data.json 原样下载，实际 2285 条，不把论文摘要的 2284 硬写成下载数量。官方代码将图像 ID 的末尾 _q编号 删除后寻找真实图像，我们沿用这一映射。reasoning 是金标准标注，不作为模型上下文。

seed 42 抽取 150 个不同图表文件，每图随机选择一条原始 QA，暂记训练 100／测试 50。150 张图像全部下载，通过 PIL 解码和 SHA-256 记录检查，跨组没有相同图像哈希；约 9.4 MiB。该 100／50 是本研究候选预算划分，不是作者官方 train/test；作者数据原用于评测，训练用途和复用依据还需继续核验。官方 HF 直连不通，从 hf-mirror 固定版本通过 IPv4 下载，显式 --noproxy，不走 Clash。
