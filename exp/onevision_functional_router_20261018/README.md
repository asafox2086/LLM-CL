# 固定专家池：质量校准与软残差

协议/源SHA/完整证明位于[summary协议](../../summary/onevision_diagnostics_20261016/functional_router/PROTOCOL.md)，源码索引为[code/OneVision_FunctionalRouter](../../code/OneVision_FunctionalRouter/README.md)。服务器本目录保存各独立pilot/full运行的日志、校准参数、原NLL/梯度、逐题输出和资源，不提交Git。

保持原1313TRAIN/21454TEST/19原生任务/seed42/nativeAnyResBF16；继承原学习相似度3LoRA池，本轮无LLM或LoRA训练。实际原43TRAIN输入对应172次NLL，16参数头；四视图全报告，quality_hard与quality_soft均完整测试，不按诊断选赢家。额外池/向量/缓存/计算与总内存均计入，不称64B总模型，不以推理峰值代替TRAIN峰值。

新GPU1唯一研究卡；原活动归因worker先自然结束并验完整证明，所有基线持续。CPU通过不是GPU收益。实时入口[functional_router_queue](../../summary/onevision_diagnostics_20261016/functional_router_queue/README.md)，新修复另建版本，不热改冻结源码。
