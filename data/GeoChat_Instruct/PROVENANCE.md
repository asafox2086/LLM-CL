# GeoChat_Instruct：已下载实验预算所需候选资源

提出者：Kuckreja、Danish、Naseer、Das、Salman Khan、Fahad Shahbaz Khan；GeoChat，CVPR 2024。[官方论文与源码](https://github.com/mbzuai-oryx/GeoChat)。实际使用：Irvin 等 TEOChat，ICLR 2025，训练使用 GeoChat 指令（正文 3.1.1、训练数据附录）。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ac3af725ae398b6184faae0828bdbd6c-Abstract-Conference.html)。

150 张原始图像已从官方 ZIP 分卷按成员提取，通过 CRC、SHA-256 和 PIL 解码检查。所选原始对话记录、source_index、retained_turn 一起保存；问题和答案与原始 JSON 对齐。候选清单按图像分组、seed 42 选 150 个图像，每图随机一条已保留问答，暂记训练 100／测试 50，图像哈希跨组无重合。原始混合数据含分类和定位；沿用先前开放问答候选过滤，抽样最终任务语义仍待检查，不宣称整包都是开放 QA，也不宣称该划分为作者官方划分。22 个旧候选问答不能严格对齐原始记录，已排除，未改写。

选中图像约 136.4 MiB，旧端约 95 GiB 分卷及完整原始 JSON 保留。资源直接服务器到服务器复制，不走 Clash。下载清单见 asset_manifest.json；原始选中记录 selected_original_records.json 仅保存在服务器。

## 本轮核验结论

已通过并进入新任务清单。此前的候选待核验状态由 [逐项核验报告](../DATASET_VERIFICATION_20261014.md) 覆盖；划分为本研究适配，原始问答不改。
