# 10%第二个完整正式结果：Qwen／SeqFT-LoRA

原CPU观察器已接受19阶段、74格、80316条原Answer F1核验，最终完整TEST为21454题。训练13105题，828次优化器调用中822次正LR、6次为原warmup合法零LR。本次仅从NEW耐久归档取回已封证明和保存图元数据，未连接OLD、读取QA或原预测正文、重评分、生成或运行模型；不修改主表或旧retention activation。

| 指标 | 原封存数值 |
| --- | ---: |
| 自然／医学／遥感／金融／交通最终F1 | 83.824996／45.391625／63.844462／33.087028／38.120223 |
| 最终五域宏F1／AP19 | 52.853667／48.634249 |
| FWT／BWT（继承描述） | 14.864222／−5.218223 |
| 刚习得五域宏／AP19（不同任务时点） | 56.481117／53.577828 |
| 完整保存系数 | 2210075136 |
| 可训练LoRA系数 | 1089536 |
| 原保存态完整权重MiB | 4217.461914 |
| 相对原基座增加MiB | 4.156250 |
| 最终完整CP MiB | 4351.614285 |

上述质量和刚习得读数均直接复制原proof；刚习得是各任务对应时点的结果，不能当作同一个终态模型。当前没有这张10%最终图的独立GPU五窗、首词延迟、生成速度或常驻adapter数，本行保留未测。旧1% Qwen的GPU实测不移填本10%行，保存字节也不能推出GPU峰。

保存图沿既有纯`pickletools` reader读取ZIP的`data.pkl`和目录元数据，未执行pickle全局、构建模型、加载tensor或读取权重载荷。原metadata尚不存在，因此本次唯一生成该元数据，耗时0.868973秒。按原完整保存态口径对实际storage去重：FP16基座4417971200 bytes＋FP32 LoRA4358144 bytes＝4422329344 bytes；不是全FP16，也未fold或转换。另排除30个可重建rotary buffer共7504 bytes；模型内注册外29个`original_inv_freq`副本及85个视觉特征缓存按既有口径单列排除。未发现未归类的模型内存储或模型外必需权重；CP中其余训练历史、缓存、RNG及容器元数据不伪装成部署权重。

耐久归档的`FINAL_MODEL_INDEX`、实际`FINAL_MODEL_RETAINED`和被接受proof之间，最终stage19的checkpoint SHA声明、文件大小、config SHA及manifest一致；retained完整模型图已存在，非仅adapter或参数计数。本次只核CP stat和既有归档SHA链，没有重读大CP payload再hash；原pin已用archive transfer SHA校验并以硬链接留存。此证明范围不等于本次重新检查所有权重字节。

证据：[原proof](second_complete_sources/dca76f5e9c3b31a8fc93.json)，SHA `23e5d338cdc1579b6f8ca4f72d511f32e23896e248bfaf91e0ae775beb257081`；[原COMPLETE](second_complete_sources/COMPLETE.json)、[原核验索引快照](second_complete_sources/VERIFIED_INDEX.json)、[耐久模型索引快照](second_complete_sources/FINAL_MODEL_INDEX.json)、[最终pin](second_complete_sources/FINAL_MODEL_RETAINED.json)及[保存图元数据](second_complete_sources/SAVED_GRAPH_METADATA.json)。[SECOND_COMPLETE_ROW.json](SECOND_COMPLETE_ROW.json)保留精确数值和完整source SHA，[SECOND_COMPLETE_FETCH.json](SECOND_COMPLETE_FETCH.json)记录NEW-only取回来源及费用。

原唯一CPU核验18.980119秒是历史费用，本次未重复。此次取回端到端2.797000秒包含NEW脚本1.184388秒及其中元数据reader0.868973秒，彼此重叠不相加；本地单次标量/storage消费者0.063000秒。原正式训练与生成费用保留在原运行证据，本次不推算或抹除。0新模型、0新生成、0新评分；主表发布与独立审阅由root完成。本结果是固定10%基线结果，不是自研方法收益或所有基线已经完成的结论。
