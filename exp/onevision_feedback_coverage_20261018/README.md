# 运行目录

真实输出位于服务器`exp/onevision_feedback_coverage_20261018/v1/{build,full_coverage_AnswerF1,full_coverage_AnswerF1_metric}/时间戳`；每次新进程，失败留证、不自动拼接或盲重试。

保存原控制500／155精确门槛、20锚输入特征、60条原始TRAIN输出、40步梯度／参数／loss／LR、CPU/GPU前后与峰值、逐题诊断／全量输出和checkpoint。没有新骨干训练，原93更新／池保留／额外反馈和验证计算仍计费。TRAIN反馈窗口与整次评测峰值分别报告。

[协议／实际ADMISSION和ACTIVATION](../../summary/onevision_diagnostics_20261016/feedback_coverage_20261018/v1/PROTOCOL.md) · [只收完整证明的四格表](../../summary/onevision_diagnostics_20261016/feedback_coverage_20261018/v1/VERIFIED_RESULTS.md)
