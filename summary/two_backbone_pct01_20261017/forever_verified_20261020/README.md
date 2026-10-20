# FOREVER／OneVision：正式1%基线完整核验

2026-10-20新增已完成结果，当前正式目标16组中完整6组。原七方法发布器只覆盖14组，其索引仍为5个完成结果；本FOREVER额外结果独立核验、独立发布，不伪造其范围或修改活动发布器。

| 方法 | 自然 | 医学 | 遥感 | 金融 | 交通 | AP | FWT | BWT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 顺序LoRA | 75.632 | 40.758 | 46.711 | 15.891 | 31.938 | 40.286 | 14.783 | 1.775 |
| FOREVER | 64.486 | 46.103 | 50.570 | 9.623 | 33.370 | 45.245 | 17.748 | 5.642 |
| OSFT | 73.907 | 43.894 | 51.709 | 23.982 | 36.196 | 46.357 | 18.932 | -1.616 |

同OneVision正式公共图像、原1313 TRAIN／21454 TEST、19阶段、seed42、唯一原始Answer F1。AP按19原生任务等权，五域列按各域QA加权，不能把两种平均混用。

FOREVER相对顺序LoRA AP高4.959pp，低于OSFT 1.111pp；本轮最强已完成OneVision方法仍是OSFT。FOREVER本次BWT为正，不能把它的作用概括成没有保持收益，也不能据单次同池比较将全部差异唯一归因于回放。没有新自研方法收益或端侧部署结论。

**权重必须分清：** FOREVER五领域等权均分40.830，低于顺序LoRA42.186；OSFT为45.937。FOREVER医学等域提高但自然／金融降低，19任务AP提高不等于五域均分也提高。保留原AP定义，并列说明域均分，不挑更有利的聚合口径来宣称全面改善。

## 额外回放必须计入

- 原主训练：1313条QA，一轮、93次正LR更新。
- 作者历史回放：**13,952次QA曝光、872次优化器更新**，其中854次正LR，18次作者warmup零LR。
- 合计主训练与回放965次优化器调用，其中947次正LR；辅助校准／额外NLL前向另有成本，965不是完整计算量。
- 13,952是从同一个冻结TRAIN池中反复访问历史QA的次数，没有增加独立训练样本。不同方法虽然使用同一1%唯一数据池，计算预算并不相同。
- replay资源按真实进程窗口记录；不把训练checkpoint大小或训练内存填入部署列。部署大小、RAM／VRAM和延迟仍待同条件实测。

## 核验及恢复边界

原正式作业已自然结束，最终checkpoint SHA `0a773e6ab10ea30cf40d714a3b09984ec592d6ce48fcbd6e839997aeae32c37a`。独立原v3验证器重新核对19有效阶段、74必要评测格、全部原ID／参考／原Answer F1、21454最终QA与最终参数文件SHA，并检查18阶段作者回放、更新／曝光、AMP清理和rolling checkpoint事务。

第一次核验发现旧机缺少已冻结的preflight证据文件，尚未产生完成证明；用新机同源原文件按预期SHA `99a01a7cb53b88deb4e1099e5f7618a5d9bbefcde06b66fdef24e87f4bd82d05`逐字节恢复后，在独立输出目录使用**同一个未改动验证器**通过。第一次日志和恢复证据保留；没有重训／重新生成预测、改评分、放宽门槛或更改活动基线源。

[完整v3证明](../forever_multimodal_20261019/v3/formal_verification/dependency_restore_v1/onevision_COMPLETE.json) · [原缺依赖日志](../forever_multimodal_20261019/v3/formal_verification/onevision_verifier.log) · [依赖恢复](../forever_multimodal_20261019/v3/formal_verification/dependency_restore_v1/RESTORATION.json) · [新旧机证据传输SHA](SOURCE_TRANSFER.json) · [联合发布凭据](PUBLICATION.json)。

联合文档入口为`analysis/v2_forever_main_table_20261020.py --write`。它在独立CPU进程中组合两个已核验范围，更新根README和内部主表；原14组索引与原活动发布器源保持不变。不再用仅七方法的旧文档渲染入口覆盖FOREVER新行。Qwen FOREVER仍无可发布结果，保持空白。
