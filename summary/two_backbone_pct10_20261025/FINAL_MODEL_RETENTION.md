> 2026-10-30更新：10%现有双骨干SeqFT及LLaVA PaRSP、OSFT共4组完整质量已核验。新PaRSP完整S19保留在OLD原持久checkpoint目录（5928092498B／SHA74c58b82…49f3），其19阶段／74格已核，未下载Windows；[新增留存记录](PARSP_FINAL_MODEL_RETAINED_20261030.json)与[完整结果](PARSP_COMPLETE_FINDINGS_20261030.md)为准。下文旧自动归档索引是历史机制/观察，不把其中旧1/16或旧PID当实时状态；未伪造NEW归档副本。 OSFT完整S19为3028599989B／SHAa041efb3…69bd，同样保留OLD持久盘；[OSFT留存](OSFT_FINAL_MODEL_RETAINED_20261030.json)。

# 10%完整主表与最终模型留存

人类2026-10-26明确：两个backbone、全部八类基线，与1%报告同样主表指标，并保留最终参数。两处10%主表现完整保留1%的18列，另列完整参数量、CP大小、新习得补充与状态，共23列。新习得和金融错答不能替代主表。

最终参数是每个正式组合S19的完整checkpoint：原模型图权重及其必要方法状态、processor、config／来源绑定一并保留，不能仅保存LoRA或参数量标量。16组合固定为onevision／qwen各自SeqFT-LoRA、FOREVER、CL-MoE、SMoLoRA、PASs-MoE、PaRSP、OSFT、SpaRTA；Ours仍为额外空行。

NEW耐久目录为`/data1/liyapeng/program/LLMCL/exp/two_backbone_pct10_20261025/private_old_archive/`。原归档只保各run latest／final，原训练checkpoint保留策略不变；新增CPU留存包装复用原传输SHA／fsync并将S19以hardlink固定在`final_models/<backbone>/<method>/<run>/`，不重复权重载荷，也不删除最终pin。未来NEW本地运行的S19先核文件SHA再pin；OLD动态FOREVER目录也纳入归档。

每组`FINAL_MODEL_RETAINED.json`绑定真实文件、bytes、SHA、config和来源；`FINAL_MODEL_INDEX.json`列出全部16组的已存／待存。模型留存与质量核验分开：归档文件不能替代完整19阶段／74格／原Answer F1证明。没有最终模型的组合标待存，缺测不填0。权重文件与QA不入Git。

此为归档与报告补齐，0模型／新评分；不调整任何训练、生成、数据或评分设置。实际启动及保存数量以本目录`FINAL_RETENTION_ACTIVATION.json`和`FINAL_RETENTION_OBSERVATION.json`为准，文档及源码准备不等于已启动或已有16组最终模型。

2026-10-26历史观察（不代表当前unit/PID）：NEW unit `llmcl-pct10-final-retention-20261026.service` active/running，PID1538724／startticks32618866／UID1004与源码匹配；已保存1/16组，即LLaVA SeqFT。其完整CP为1926008036 bytes，SHA `801ce3d059deb0e9e786147fcb03a983784194c4b13ce98c3767e68b93b8b618`，实际hardlink pin及配置记录已核，归档无错误。原CPU归档unit已停止，训练worker不受影响。启动后收据观察断言未通过，随后只核原同unit／PID并补存实际观察，没有重新启动服务；更正细节保存在ACTIVATION记录。
