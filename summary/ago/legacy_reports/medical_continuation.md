# 医学续训进度与结果

更新时间：2026-10-07 21:50:24 CST。

从各自七任务 checkpoint 继续：任务8 MTS-Dialog → 任务9 IU X-ray。
每新增任务 train/dev/test = 1000/100/200；旧任务保持原500条测试。分数为 ROUGE-L，变化单位为分。
运行中仅展示已完成整阶段的汇总，不把未完成结果当最终结果。

| 方法 | 状态/阶段 | 医学前旧任务均分 | 当前旧任务均分 | 旧任务变化 | 医学前均分 | 当前医学均分 | 医学变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| seq_lora | completed / 9  | 25.634 | 26.272 | 0.638 | 9.544 | 19.198 | 9.654 |
| migu_lora | completed / 9  | 23.084 | 24.842 | 1.758 | 10.084 | 12.742 | 2.658 |
| olora | completed / 9  | 21.926 | 21.930 | 0.004 | 7.562 | 6.211 | -1.351 |
| sapt_lora | completed / 9  | 12.030 | 7.926 | -4.104 | 9.135 | 8.661 | -0.474 |

## 运行目录

- [seq_lora](../exp/result/medical_generation2_after_superni7_v1/seq_lora_t5large_after7_medical2_seed42_epoch1/20261007T085237Z/)：completed；[阶段对比表](../exp/result/medical_generation2_after_superni7_v1/seq_lora_t5large_after7_medical2_seed42_epoch1/20261007T085237Z/comparison.md)。
- [migu_lora](../exp/result/medical_generation2_after_superni7_v1/migu_lora_t5large_after7_medical2_seed42_epoch1/20261007T085237Z/)：completed；[阶段对比表](../exp/result/medical_generation2_after_superni7_v1/migu_lora_t5large_after7_medical2_seed42_epoch1/20261007T085237Z/comparison.md)。
- [olora](../exp/result/medical_generation2_after_superni7_v1/olora_t5large_after7_medical2_seed42_epoch1/20261007T085237Z/)：completed；[阶段对比表](../exp/result/medical_generation2_after_superni7_v1/olora_t5large_after7_medical2_seed42_epoch1/20261007T085237Z/comparison.md)。
- [sapt_lora](../exp/result/medical_generation2_after_superni7_v1/sapt_lora_t5large_after7_medical2_seed42_epoch1/20261007T085237Z/)：completed；[阶段对比表](../exp/result/medical_generation2_after_superni7_v1/sapt_lora_t5large_after7_medical2_seed42_epoch1/20261007T085237Z/comparison.md)。

详细设置：[协议](../rules/003_medical_continuation.md)；[论文目录](../paper/medical_papers.md)。
