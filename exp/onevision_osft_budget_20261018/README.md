# 固定参数预算SVD研究运行

唯一CPU调度为analysis/onevision_osft_budget_queue_v2.py，同llmcl-onevision-causal-priority.service；自行等待真正空闲GPU0/1，单研究流，我们最多3张新机卡。不要手工再起竞争worker。

[协议和证据](../../summary/onevision_diagnostics_20261016/osft_budget_20261018/v2/PROTOCOL.md)。v2训练运行保存20份q/v SVD状态、每步raw／projected梯度／更新前因子、loss／LR／有效位移与逐阶段内存。每组约3.1GiB状态，加梯度／运行证据；模型／checkpoint不进Git。两组155诊断之后各自完整21454，未测出结果不填正式Ours。

v1只做过CPU检查，没有GPU或QA运行；因混合精度重建问题保留停用。v2需要真实GPU前向／共同S0／首批原始梯度门槛，当前等待卡，不称已训练。
