# 10%首个完整正式结果：LLaVA／SeqFT-LoRA

已完成19阶段、74格、80316条原Answer F1核验，最终完整TEST21454题；训练13105题、828次优化器调用中822次正LR及6次原warmup合法零LR。训练原worker自然exit0后，原唯一CPU观察器已评分并封存，本次只读取已完成证明，没有重评分、生成或模型调用。

| 最终指标 | 数值 |
| --- | ---: |
| 自然／医学／遥感／金融／交通 F1 | 73.069／45.294／57.157／24.602／37.189 |
| 五域宏F1／AP19 | 47.462／45.874 |
| FWT／BWT（继承描述） | 21.156／−2.933 |
| 新习得五域宏／AP19（不同任务时点） | 49.996／48.652 |
| 完整保存系数 | 894216224 |
| 原保存态完整权重／相对基座增加 MiB | 1706.613／2.062 |
| 最终完整CP MiB | 1836.784 |

同骨干同方法的1%最终五域宏为42.186，本10%为47.462，约增加5.276pp；医学／遥感／金融／交通提高，自然下降。支持本固定单轮配方增加数据和更新预算后总体得分提高，不是纯样本量因果、全部基线结论或自研方法收益。尚未重读原预测细分集中错答，不能据F1增加声称该现象已消失；正式主表保持完整五域，不只报金融。

完整CP已保存于NEW耐久目录，SHA与原已接受proof相同；私有保存态图使用既有纯`pickletools`工具读取ZIP对象元数据，未加载权重／执行pickle全局。参数storage去重后1789513792 bytes（FP16基座＋FP32 LoRA）；排除26个配置可重建buffer共9032 bytes，model内注册外25个original_inv_freq副本及16个视觉特征缓存沿1%原口径排除。没有未归类的模型内状态或模型外必需权重。GPU五窗、首词延迟、速度、常驻adapter数尚未独立实测，表中保留未测，不用训练峰或CP大小替代。

证据：[原完整proof](completion_observer_native/completed/5d039de56ea15c2f32fc.json)，SHA `73fdbbd139a6910225088e78cc44033824456e09dec93411ed93582ac4d67c99`；[原COMPLETE](completion_observer_native/completed/5d039de56ea15c2f32fc.COMPLETE.json)及[原索引](completion_observer_native/VERIFIED_INDEX.json)；[主表完整标量与存储分类](FIRST_COMPLETE_ROW.json)；[真实最终模型归档](FINAL_RETENTION_OBSERVATION.json)。baseline_review独立只读核对全部计数、来源链、指标、参数布局及排除分类后接受首行发布；0新模型／评分／权重载荷读取。原唯一评分10.738419秒，元数据读取费用见ROW，连接／运输／审阅另计，历史正式训练与生成费用保留。
