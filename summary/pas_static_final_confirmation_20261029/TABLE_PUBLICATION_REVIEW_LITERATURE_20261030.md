# PAS-static入表独立复核

接受当前两主表及FINAL_FINDINGS_20261030.md口径。此审只读完整RESULT/VERIFIED与待发布文档，没有重verify/score或模型操作。

- README首个HTML新行23个数值格（19小类+医学/遥感/交通域均值+五域宏）逐格与RESULT三位小数完全一致；宏46.332不是AP19。
- 首个资源表AP19=46.002、FWT/BWT缺测、893.676M总部署/0新增、1704.551MiB权重/0增量、2166.101MiB固定五窗allocated最大值、TTFT/tok-s缺测、适配器数0均正确。
- summary pct01 MAIN五域与AP19行一致。两处均明确末态部署表示而非新训练基线，Ours占位保持。
- README从首HTML仅移除PAS-static新行后与HEAD逐字节相同，其余3个HTML块逐字节相同；原资源表旧行和summary MAIN全部原表行原字节内容保持。
- FINAL_FINDINGS保留宏下降0.288pp、AP下降0.404pp、遥感/Presence下降、80历史TRAIN答案回访、FP32分支至FP16表示、固定五窗非全TEST峰、0B故障保留及一次批准核验恢复，不称无损、普适或零训练费用。

未修改README或主表；接受root白名单发布。
