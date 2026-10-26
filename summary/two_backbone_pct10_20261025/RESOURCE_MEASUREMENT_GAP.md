# 10% SeqFT资源空列与现有复用入口

本次只核本地源码、协议及已封小元数据，0模型、0生成、0评分；未实现新测量入口，未新增协议、门、队列或服务。两条10% SeqFT的完整S19已保留，但 `FIRST_COMPLETE_ROW.json`、`SECOND_COMPLETE_ROW.json` 的GPU五窗、TTFT、tok/s、常驻适配器字段仍为null。不能从1%结果、CP大小或训练资源推算填入。

| 指标 | 已有定义／代码 | 当前可复用范围 |
| --- | --- | --- |
| 固定五窗GPU峰 | [paired_readouts.py](../../exp/native_response_selection_20261025/v1/paired_readouts.py) 的 `resource_windows`，63–100行 | 五域各一条hash-min原始QA，batch1／greedy／固定32新token；独立fresh reload后、任何DEV读出前，重置并记录allocated／reserved峰，包含输入准备。原始ID／图像／上下文及实际卡UUID绑定；跨骨干不要求prepared张量相同。须在10%各自S19原图上GPU实测。 |
| TTFT | [原主表指标协议](../edge_deployment_20261019/PROTOCOL.md) 第12行 | 模型已加载后，从原始图文请求开始到首个输出token，包含图像处理、视觉编码、prefill及首步解码；加载耗时单列。定义已有，不需另造指标。 |
| 生成速度 | 同协议第13行 | 首token之后的实际解码token数除以对应时间，batch1、固定生成预算，另保留完整请求时间；必须GPU实测。 |
| 常驻适配器数量 | 同协议第14行 | 为完整推理常驻的独立适配／专家数；明确“同一个LoRA的多个层不算多个专家”。这是已有主表计数定义。 |

GPU streamer的精确实现位于 [analysis/edge_visual_gpu_probe_20261019.py](../../analysis/edge_visual_gpu_probe_20261019.py)：外层 `run()` 第90行，内层 `call(row, task, view, tokens, speed)` 第143行，计时起点第147行，`Clock(BaseStreamer)` 第186行，`Clock.put` 第188–190行。它跳过首次prompt回调，每个生成token回调先 `torch.cuda.synchronize()` 再记 `time.perf_counter()`；第200–201行计算 `TTFT_seconds=times[0]-start`、`decode_tokens_per_second=(len(times)-1)/(times[-1]-times[0])`。这些是可借用的计时机制及公式。

该旧GPU探针整体服务于另一视觉实验，含视觉重排／裁剪及直接 `language_model.generate`，不是10%完整原图的可直接运行入口；只复用计时机制，保留原完整模型的生成路径。旧8-token调用也不冒充本次固定32-token测量。现有五窗 `resource_windows` 没有streamer／时间字段，[原Qwen资源worker](../../exp/strong_qwen_resource_20261025/v1/worker.py) 第221行明确 `latency_measured=False`，因此已有五窗内存收据无法回填TTFT或tok/s。

薄计时helper现已由root准备：[analysis/fixed_window_token_clock_20261026.py](../../analysis/fixed_window_token_clock_20261026.py) 的 `FixedWindowTokenClock`。它不导入Torch／模型；caller须在输入prepare之前调用 `start()`，跳过prompt回调，每个生成token同步记时，完整32-token后按原公式汇总。静态审阅未见接口阻断；root的fake时钟检查仅核回调及拒绝不完整窗口，不是GPU／时延实测。该helper尚无测量consumer、队列或GPU验证，未连接任何在跑／冻结源；所有10%GPU峰和TTFT／tok/s仍缺测。以后复用时须披露逐token同步，结果只称此instrumented固定窗口墙钟时间，不冒称生产吞吐。

原SeqFT只有一套跨任务连续更新的q/v LoRA bank；LLaVA的48个、Qwen的56个投影模块是这一套bank在各层的A/B，不计为48／56套。源码依据为 [methods.py](../../exp/two_backbone_native_cl_20261017_v2/methods.py) 的 `SeqLinear`（44–56行）和 `Method.__init__`（131–159行）：Seq分支逐q/v目标安装一对A/B，`begin_stage`（166行起）没有为Seq新增task bank。故按现有定义，保留原未fold Seq图应为1套；这是结构计数，不是GPU性能估计。可据绑定S19的已存graph metadata核定，并在实际重载时顺带核结构；本说明不直接改主表null。若以后另导出fold图，其0套属于另一实际表示，不能暗替原保存态。

完整S19恢复可借 [checkpoints.py](../../exp/two_backbone_native_cl_20261017_v2/checkpoints.py) 的 `load(run, expected=None)`（51行起）：验证原CP SHA、保留原CPU／CUDA placement、method／processor／config，再由原方法执行。两个1%资源worker提供原图恢复和五窗绑定范例，但其CP／manifest／条件均硬绑定1%，不能原样指向10%执行。尤其不能调用 [checkpoint_probe.py](../../exp/two_backbone_native_cl_20261017_v2/checkpoint_probe.py) 测最终资源：它要求preflight的 `probe`、做loss／backward，且第35行会删除 `checkpoints/*.pt`。

本地现有调度源只安排正式preflight／reload／training，以及已定位置／conditional研究slot；未发现两条10% S19资源补测的READY或排队条目。最小真实交付缺口是：用已保存原完整图逐个fresh恢复，绑定同五个raw请求，测五窗并核常驻bank；若同时填时延，补上述薄计时调用即可。无须重训、再测F1、改骨干／精度／后端或重建benchmark框架。该结论仅说明已有本地计划缺口，不宣称已做服务器实时队列实查。
