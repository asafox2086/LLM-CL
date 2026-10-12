# 实验协议已迁移

当前唯一设置文档为 [rules/001_experiment_protocol.md](../rules/001_experiment_protocol.md)。

用户已确定直接使用完整官方 SuperNI 的七个生成任务，每任务 1000 条训练、200 条验证、500 条测试。SAPT 仅供方法与指标参考，不使用其样本划分。

原 Long Sequence 分类主实验、MixLoRA 实验及 900/100/100 建议均不适用。当前数据协议为 v2，排除 Reddit TIFU 的空答案后仍保持配额。用户已授权 O-LoRA 首跑，入口为 `exp/run_olora.sh`；首跑参数和模型前置条件见新文档第 13 节。
