# GeoQuestions1089：官方原始文本资源已下载

提出者：Kefalidis、Punjani、Tsalapati、Plas、Pollali、Mitsios、Tsokanaridou、Koubarakis、Maret，ISWC 2023。[官方数据与引用](https://github.com/AI-team-UoA/GeoQuestions1089)。实际使用：[GeoQA2，2024 正式期刊论文](https://doi.org/10.1016/j.jag.2024.104203)；[GeoBenchLLM 作者评测代码](https://github.com/Rfr2003/GeoBenchLLM)，[2026 预印本](https://arxiv.org/abs/2608.07411)，后者尚未核实正式会议发表。

官方 Git commit b23ded0dd321c45b421be21f907d05929dbc20d8。GeoQuestions1089.json 与 GeoQuestions1089_answers.json 已完整下载，分别 1089 条。问题 SHA-256 58645517fbb50723d02501bb24d41c663e020616a3f6cb4a9c6ea3fcfb9cb862；答案 SHA-256 75b5a3abfab344139633345b03de24c4efe86244dde50c379570c4d8eb0d44e0。原始文件约 79 MiB。

原始任务基于 YAGO2geo/YAGO2，包含查询结果 CSV、JSON、WKT 和实体集合。初步格式审计检出 238 条单值数值答案，但其中包含非开放任务，不能把数量直接称为已确定的最终子集。大于 100 万字符的几何答案在格式审计中单独统计，原始内容保留。正式文本子集与 100／50 划分仍待按论文用法核验，不输入原始金标准查询或答案作为上下文。
