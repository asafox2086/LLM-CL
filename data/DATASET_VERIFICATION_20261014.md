# 新任务数据核验结论（2026-10-14）

核验通过并写入正式小数据清单：T5 GeoQuestions1089、T6 GeoChat_Instruct、T8 MME-Finance。三项均训练 100／测试 50；无验证集、seed 42。其余七项源文件 SHA-256 保持不变。原始标注不改；数据准备与方法适配状态分开记录。

| 任务 | 提出论文 | 实际复用依据 | 本轮任务口径 |
|---|---|---|---|
| T5 地理文本 | Kefalidis 等，ISWC 2023，[官方数据](https://github.com/AI-team-UoA/GeoQuestions1089) | [GeoQA2，2024 期刊](https://doi.org/10.1016/j.jag.2024.104203)；Abayomi-Alli、Moreno、Radouane、Tamine，[GeoBenchmark，LREC 2026](https://aclanthology.org/2026.lrec-1.417/)，附录 B、表 10 实际重评计数／人口问答，B.1 为零样本及少样本问答 | 从原始正常问题部分（ID ≤1017）选单值数值答案、how many 或 population 问题；不取 Yes/No、WKT、实体集合；不给模型金标准 SPARQL 或答案上下文 |
| T6 遥感图像 | Kuckreja、Danish、Naseer、Das、Salman Khan、Fahad Shahbaz Khan，[GeoChat，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.html) | Irvin 等，[TEOChat，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ac3af725ae398b6184faae0828bdbd6c-Abstract-Conference.html)，正文 3.1.1、训练数据附录实际训练使用 GeoChat 指令 | 计数、空间关系、图像描述等自然问答；不使用场景类别、[identify]/[refer] 定位任务；原回答包含作者生成数据，不冒称全人工标签 |
| T8 金融图像 | Gan、Dong Zhang、Haohan Li、Yang Wu、Xueyuan Lin、Ji Liu、Haipang Wu、Chaoyou Fu、Zenglin Xu、Rongjunchen Zhang、Yong Dai，[MME-Finance，ACM MM 2025](https://doi.org/10.1145/3746027.3758230) | Huang、Shi、Peng、Lai、Xie、Jin，[OCR-Reasoning，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f97eefb5e158cc6f64c7883cf4afc6d3-Abstract-Conference.html)，3.1 从 MME-Finance 等既有数据实际获取图像；另有 [BICR 原始评测代码](https://github.com/Ledengary/BICR)，2026 预印本 | 官方英文 MMfin，作者 image_type=candlestick/indicator/statistic、image_style=PC；保留图像描述、空间理解、数值计算、建议解释等原始开放任务，排除 OCR、实体识别、金融知识常识题、Not Applicable；不采用 Binary 文件 |

## 复用依据边界

OCR-Reasoning 的“使用”是取既有图像、重新标注其自身问答，不能写成复现 MME-Finance 全套原问答评测。本轮直接使用 MME-Finance 原图与原 QA，不使用 OCR-Reasoning 的改图／新 QA。BICR 是原问答评测的补充代码依据，其论文目前仅核实预印本，不冒称正式会议论文。

FinChart-Bench（ACL 2026）只找到 MoCA-Agent 预印本明确实际评测；UniFinEval 等目前属于引用。不能用原论文、其 arXiv 版本和同一数据卡凑多篇正式复用。故本轮不采用 FinChart-Bench，下载文件保留候选追溯。

## 子集和小数据划分

- T5：LREC 论文表 10 报告计数 143、人口 28。当前官方发布按上述显式条件得到 how-many 146、population 32、并集 175；与作者数字并不完全一致，作者具体 ID 未提供，不能声称复现其 171 条。采用同一问题类型，保留原始问题、原 CSV 的单值答案字符，seed 42 洗牌后固定 100 训练／50 测试；剩余 25 条不进入实验。属于本研究小数据划分。
- T6：从已对齐原始记录及 ZIP 图像的开放 QA 中，seed 42 选择 150 张不同图像，每图选择一条原始对话问答，固定 100／50。150 张图像哈希均不同，跨组无重合。<image> 是原数据结构标记，仅在模型输入序列化时移除，原 human 字段完整保留；没有改写语言内容。旧候选中 22 个无法与原始问题严格对齐的记录已排除。
- T8：官方英文 1171 QA 中，PC 样式、上述作者类别得到 153 QA。以原 image_path 的父目录标识底层图表组，并合并字节相同的图像组。seed 42 打乱组，确定性整组分配 100 训练 QA；从其余组取 50 测试 QA，剩余 3 条不进入实验。训练 25 个图表组，测试 22 个；同图多问题允许在同一组内出现，不能据此称 150 张独立图表。跨组图像 SHA-256 不重合。

三项原始数据用于论文评测或指令训练，均未发布本轮所需的官方 100／50 训练测试划分。我们的划分是持续微调研究适配，不等于复现原论文协议，文章需明确说明。现有医学任务仍沿用官方 QA 划分，可能同图多问题跨官方划分，已在既有协议记录；没有为了宣称全域图像隔离而改动它们。

## 原始标注与资产核验

原始问题、答案逐条与官方下载记录比较；图像校验 CRC／SHA-256、PIL 解码；单任务训练／测试 QA ID 不重合。除 CSV 字段解析和移除结构性 <image> 标记外，不生成答案、转换数值单位或翻译问题。原始标注和完整字段均保存服务器，选中 ID、路径和 SHA-256 由 [当前清单](TenTaskCL/current_small_manifest.json) 固定。

[机器核验结果](TenTaskCL/dataset_revision_verification_20261014.json) · [准备脚本](TenTaskCL/prepare_geo_fin_revision.py) · [十任务真实样例](TenTaskCL/samples.md)。正式训练策略仍关闭，当前原因是方法适配及新任务 GPU 预检尚未完成，数据核验已完成。
