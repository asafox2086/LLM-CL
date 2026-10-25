# 最强已完成Qwen基线：同卡五窗GPU实测

原正式Qwen2-VL-2B-Instruct SeqFT-LoRA stage19完整checkpoint直接重载后，五个固定32-token窗GPU峰 **4311.996–4954.604 MiB**。相同物理GPU1、相同五条raw QA的已完成LLaVA朴素r8为1856.979–2166.101 MiB，逐窗少 **55.147769%–58.803849%**。这是资源参照的新实测，不是新门效果或正式F1提升。

| 域 | 原Qwen峰 MiB | LLaVA朴素r8峰 MiB | 下降 % | Qwen／LLaVA视觉token |
| --- | ---: | ---: | ---: | --- |
| natural | 4890.847168 | 2166.100586 | 55.711137 | 504／5157 |
| medical | 4954.603516 | 2041.105957 | 58.803849 | 529／3699 |
| geography | 4311.996094 | 1856.979004 | 56.934585 | 81／1485 |
| finance | 4550.723633 | 2041.101074 | 55.147769 | 340／3369 |
| traffic | 4905.764648 | 2166.091797 | 55.845990 | 510／4501 |

窗口均含原输入准备、GPU视觉提取与greedy32新token生成；每窗清理/同步/reset peak，原输入ID、图像SHA/尺寸、检索、问题/指令与401408面积预算相同。保各自原processor和模板，native prepared/token数不同合法，不强行匹配图像token或以更少token证明算法收益。五窗最大值用于主表时须加注，不代表全TEST最坏峰、NVML、CPU总内存、latency或最优部署。

原Qwen完整2,210,075,136系数／4,422,329,344权重bytes（FP16基座＋1,089,536个FP32 A/B坐标），另7,504注册buffers bytes；checkpoint容器4,562,999,772bytes。LLaVA已fold完整893,675,552系数／1,787,351,104bytes，系数少59.563567%、权重字节少59.583492%。跨骨干减量主要是基座/表示差异，不称新结构创新；Qwen保原未fold混合dtype，不是公平最优fold的上限。

没有重构/合并/量化/改变模型dtype或后端；原公开generation配置、全部权重/注册buffers、56个q/v-r8结构前后不变。旧视觉CPU缓存清空并禁用，原CPU checkpoint元数据133183696bytes不是RSS。GPU视觉准备按末维区分Qwen二维raw patch与特征，不采用OneVision的ndim-only快捷判据。

实际5生成／160强制新token，0评分／NLL／训练／更新。模型68.625645367秒，controller 69.931951760秒重叠不加；唯一metadata核验0.012348259秒。运输40.240818748秒、准入0.228828288秒及实现/审阅/发布另计。完整原CP来源未删、OLD私有副本保留；不重复预训练权重、不以模型加载作为CPU神经门。

实施记录：准入客户端首次误用审阅JSON的protocol_sha256扁平字段，实际为protocol.sha256，发生于远程调用/模型之前，修客户端后首次成功准入。首次push遇同一提交并发更新ref，远端已是本次4a921c0；核远端/本地/8白名单内容一致并fetch承接，没有新提交/强推/丢改。发布proof确认后才首次启动。它们不是科研阴性或模型重跑。

unit实际inactive/MainPID0/exit0、worker身份已无、COMPLETE/EXIT/VERIFIED/ALL_EXIT完整才提取结果。独立单流在GPU1运行，ARC固定三臂继续GPU2训练，不更改其科学源/预算/轨迹。只补全资源空白，主科研目标仍未实现。

[完整核验](VERIFIED.json) · [标量与全部窗口](OUTCOME.json) · [原始测量](memory.json) · [事前协议](PROTOCOL.md) · [独立源码审阅](SOURCE_REVIEW.json)
