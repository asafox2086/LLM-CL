# 当前实验状态

2026-10-07：四个 T5-Large 七任务实验全部完成。从各自第七阶段 checkpoint 延续的两个医学新任务（MTS-Dialog → IU X-ray）也已全部完成。

医学续训已在 GPU 0/1/2/3 完成，分别为 SeqLoRA、MIGU-LoRA、O-LoRA、SAPT-LoRA。每个医学任务 train/dev/test 为1000/100/200；每学完一个医学任务，完整复测原七任务，测试集不变。

- [医学续训实时进度及结果表](medical_continuation.md)
- [原七任务对比表](t5_large_comparison.md)
- [医学续训协议和恢复方法](../rules/003_medical_continuation.md)
- [已下载论文](../paper/medical_papers.md)

所有实验保持 nohup + setsid，参数、逐题回答、分数、日志及checkpoint按方法独立保存于 exp/result/medical_generation2_after_superni7_v1/。四方法已通过7→9任务集成测试、父checkpoint保持不变测试及优化步骤中断后精确恢复测试。
