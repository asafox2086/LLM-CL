# 首批更新前NLL差异：先解决起点功能承接

## 本次回答的问题

原r200两流最终宏F1只有约.91/.88，不能把全部失败预设为阶段末归并或后续遗忘。本次一次标准库CPU消费者读取三个已封SHA事件日志中的首记录，检查差异是否已经在第一次optimizer更新之前出现；不复算旧F1、配对、梯度能量或谱。

| 条件 | S1首批QA | prompt tokens | target tokens | 更新前批均回答NLL |
| --- | ---: | ---: | ---: | ---: |
| 既有LoRA_qv | 16/327 | 65871 | 72 | 5.087518595159054 |
| Spectral | 16/327 | 65871 | 72 | 17.27503740787506 |
| Answer | 16/327 | 65871 | 72 | 17.27503740787506 |

压缩两流比qv首条记录均高12.187518812716007。这里只是一个批的数值差，不称统计显著、不求区间或PPL倍数，不以NLL替代原Answer F1，不外推全部五域。两压缩流首批NLL相同，也不证明后续轨迹相同。

原TRAIN源码先累计各QA内token平均loss，再以16QA均值保存loss_sum；optimizer.step发生在这些前向/反向之后。日志虽然step后落盘，保存的loss来自step前；S1尚未有任何阶段末归并。这只定位日志差异的时间，不隔离最终质量的全部原因。

## 哪些可比，哪些没有被证明

- 已有封链：LoRA_qv旧VERIFIED绑定事件SHA b91dabc812f1c180fab6bb53874277f03ecb91e3c5989ac475fe234a6b89ed7e；两新COMPLETE绑定各自events_sha256。本次三个小日志SHA均匹配，各87条，但只消费首条数值。
- manifest/SPLIT/原TRAIN AST相同，共有基座权重SHA相同，训练/answer_loss/OneVision桥/冻结视觉输入四份当前源码与两READY共同冻结SHA匹配；首条stage/step/QA、prompt与target token计数相同。
- **旧READY只封权重，新READY另封14份processor/config/tokenizer文件；旧历史预处理哈希缺测。** 这是缺测，不是发现冲突，不能补当前hash冒充旧封存。原逐批input digest也不存在。
- 因此historical_preprocessing_hash_match、strict_input_and_preprocess_match、controlled_causality均为false。本轮只称“同声明FIT配方/顺序源码、相同首批计数下的已记录NLL观察”，不称严格受控的纯压缩效应。

这个范围在读loss之前、只看到READY字段覆盖差异时已明确收窄，并由review_next_probe再次接受；旧实验科学源/READY/完整性门槛全部不改，没有按质量结果放宽任何实验判据。若哈希冲突/聚合不同，本消费者原定停止，而不是换数据或补模型探针。

## 会改变什么设计判断

可以记录：**相同声明配方下，首批日志差异在首次更新、任何归并之前就已存在。** 下一设计应优先解释压缩后起点如何承接预训练回答功能、输入如何可比，而不是直接把全部问题归给后续遗忘/归并，再去换mask或加保护。

不能记录：SVD已被唯一证明致损、NLL差能解释最终F1降幅、恢复一定有效、全部低秩不可用或gate/up私有方向一定必要。实现/数值路径、输入与起点差异和后续归并仍有未隔离因素。旧r200配方仍关闭，不能据此补恢复、换rank/LR或重开旧模型。

原NativeGate核对的是**同一压缩模型**manual/native两条前向的NLL接近、4token生成相等；它不是对未压缩父模型的功能保真比较。通过兼容性门槛与tiny反向不保证原回答质量保留。此区别来自原冻结源码，未新增前向/验证。

未来若有具体新结构/功能替代假说，协议应区分起点功能、合法恢复与持续适应的证据，并给出最少对照/完整成本与正负出口；本报告不是新算法、准入或自动训练授权，也不靠再次审计本文维持活跃。

## 费用、证据与附录分支

消费者analysis/initial_compression_loss_evidence_20261023.py已唯一执行0.14164666099532042秒（SSH另计），0模型/张量/QA/生成/optimizer/GPU/原score/bootstrap；EVIDENCE SHA0d678580f77860a506e6f942ba942c3257a9b61c7020a5bcedc41e2a7a401ca3。Question、源、首记录、各封链SHA、缺测文件名均在EVIDENCE保留，不复算旧质量。

同期独立视觉可达性核验见../../vision_reachability_20261023/v1/FINDINGS.md：实际vision_feature_layer=-1，末block必须保留；pool head未构造；只有post_layernorm的2304系数/4608bytes在当前eval回答路径不可达。微小公共冗余只记附录、不启动模型清理/重载/导出或宣称方法收益；它不能解决小模型保质问题。

两项分析均未改正式基线或恢复worker，当前没有研究模型训练，完整科研目标未实现。
