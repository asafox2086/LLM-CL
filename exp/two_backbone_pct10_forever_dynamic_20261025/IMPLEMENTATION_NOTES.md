# FOREVER 10%动态调度：独立实施记录

状态：**WIP源码实现，只有AST／compile检查，0模型、0GPU／CPU模型测试、0评分、0运输、0启动。** 这里只拥有新`core.py`与`forever_adapter.py`；其余runner、policy和核验尚待root独立构建/审查。现有12个native基线和两项旧FOREVER defer状态不因本文件自动改变；旧1%与现有10% clone、冻结三源均未修改。

## 来源与文件身份

- 作者固定revision：`bbeb549a21e59d9490f20a2b02099ee766a67f48`；[原训练文件349–678行](https://github.com/WoodScene/FOREVER/blob/bbeb549a21e59d9490f20a2b02099ee766a67f48/src/finetune_cl_our_qwen.py#L349)。
- 原`analysis/forever_multimodal_core_20261019.py` SHA：`2d9d5be87d81df80eab6ddf0095ce399c9a73d9c8c29909a5022d2364fdeeed8`。新core启动时核其字节，继承其已固定作者selector、DriftRecorderCallback、ReplayTrainer loss和强度公式，不修改旧文件。
- 原`analysis/forever_multimodal_replay_v2_20261019.py` SHA：`8024931496a032ea666e44376d79614d7c223b17c73ff11a8074494f87cd3683`。新adapter继续调用这一实际HF回放与AMP清理，不改变两epoch／5warmup／正则损失。
- 原v3`forever_adapter.py` SHA：`a9fd1fc2750117b94dfc712392ef616dca9cdb586b738e0b28ec5408097f710d`。
- 本次新`core.py` SHA：`7dcbadf1a7793cf32b13cbf08e2697a4d5588d5062841c5055eb82ed0d70202b`；新`forever_adapter.py` SHA：`68a911e46c1e6db9d8da8125e9243f6480032e98bbbdc1f8cd9e49db9c5327cb`。如后续审查修源，须更新此身份记录，不称本版本已冻结。

## 实际状态机

`ForeverStage`由真实当前数据计算`N=ceil(n/16)`，在第`min(24,N)`次真实current更新后校准，保留作者参数时间、强度与既定阈值。校准后允许继续current；`main_update_complete`只在真实optimizer更新后调用一次。漂移callback继续只观察current参数位移，回放不推进其step／tau／mu。

`dynamic_due`同时要求已有历史、未耗完当前N、已经校准且已到下一阈值。adapter在每次`after_update`中循环执行所有此时已达的阈值，每个阈值仅一次；包含第24步结束后的首阈值即时检查。若第N步正好越阈值，不再补一个动态回放，转阶段末的最终回放，保持作者循环的预算终止语义。S1无历史，217个current更新正常继续而不伪造回放。

`begin_replay(kind)`生成当时beta及固定task起点theta_star的原ReplayTrainer；`replay_complete`只更新drift位移参考，严格检查current步数／tau／mu／mu0不变，再推进阈值索引或完成final。没有随触发重建theta_star，也不将回放位移加入模型时间。阶段结束必须当前N全部完成，且有历史时final确已完成；无历史则记录跳过。

## 与作者执行方式的明确差别

主runner仍采用共同的一task一optimizer／cosine、原hash数据顺序及一遍current数据，插入回放不重建外部主optimizer／scheduler，不重抽当前样本。作者HF脚本按current片段创建新Trainer，因此本实现是**作者触发数学在共同多模态主训练协议下的适配**，不是逐字节复现作者分段HF训练。此差别不能隐藏为一般I/O修补。

主训练器原`after_update(step,updates)`调用时序保持；回放结束恢复当前task instruction、train模式及清理梯度，避免历史QA instruction泄漏到随后current步骤。原回放的AMP unwrap检查继续保留，是否与随后主训练实际衔接正确仍需GPU证据，源码通过不等于已经验证。

### 活跃主训练状态与随机流的实际边界

只读项目回放源及本地Transformers4.46.3／Accelerate1.2.1实现后，正常单卡路径未见HF Trainer取得外层run中的主optimizer／scheduler／GradScaler对象：它为同一Parameter对象建立自己的AdamW、scheduler和scaler。因而没有发现直接重置主m/v／step或主scaler的代码；但**回放会真实改变权重，外层主m/v仍是回放前的参考**。从旧矩状态在回放后的权重上继续current，是共同协议的明确选择，不能称主执行数学完全没有变化，也不同于作者重建current Trainer。此处是源码边界判断，尚未实际证明整个回放前后主moment／scaler、Parameter引用、requires_grad和gradient-checkpoint标志逐值不变。

HF Trainer初始化会调用`set_seed(42)`，重置Python／NumPy／Torch／CUDA全局随机流，随后回放又消耗dropout等随机数。**本实现没有隔离整个replay的RNG，也不声称主训练随机流保持不变。** 这是继承原HF回放执行的行为；如改变它，必须另行明确说明，不能暗加保存／恢复整个replay随机状态的措施。

原`execute_replay`仅在回放结束的两次AMP清理NLL probe前保存Torch CPU／CUDA RNG，并在probe后恢复；它不撤销Trainer初始化或实际回放的seed／随机消耗，也未保存Python／NumPy状态。probe使用eval并还原原mode，但输入处理仍可能更新视觉cache、bridge输入引用和运行计数；这些不是权重不变性证明的一部分。新adapter恢复current instruction／train并清梯度，后续current输入会重建其批次，不能把所有运行缓存说成原封不动。

`tensor_digest`本身只detach、CPU复制并读取字节，没有模型参数写入或随机数调用。原unwrap检查保存的是**回放训练结束后、unwrap前**的参数ID／storage／版本与trainable digest，并比较unwrap前后两次NLL；它证明的范围是清理操作，不覆盖整个回放对主训练状态的影响。未来真实插入点验证仍须核对上述外层状态并继续真实current更新；本次未执行该验证。本地库版本与已封ARC环境版本一致，但未据此声称新服务器安装文件已逐字节复核。

## 输出与费用hooks

每task使用`method_state/stage_XX/`：

- `main_update_NNN.json`：真实current步数、tau、校准状态、mu／mu0及触发前索引。
- 每次`dynamic_II/`或`final/`独立保存`REPLAY_REQUEST.json`、`actual_author_replay/`完整原HF日志/梯度证据及`REPLAY_CLOSED.json`，不覆盖不同触发的文件。
- 各次回放的`resources/`独立ResourceMonitor包含实际回放及原AMP清理/两次TRAIN NLL探针；原外层current更新计时亦包含插入回放，重叠区间不能重复相加，reset的局部峰不称整次训练全程峰。
- `memory_selection.json`保存实际原selector结果，`calibration.json`保存current结束状态，`SCHEDULER_COMPLETE.json`保存回放后完整调度摘要；method_state中保留各stage摘要，checkpoint前core已释放。

828仅为10%冻结主更新数；每个动态及最终回放的实际QA、epoch、optimizer调用、零LR warmup、额外NLL、保存/监测费用必须另外记录。ratio=2仍是作者按答案类选择参数，不保证全局缓存正好2%。本实现没有声称同总计算、同训练内存或已取得质量/端侧收益。

## 预检合同与尚未完成的证据

集成源码审查发现并在GPU运行前修正一个确定入口错误：原run先`begin_stage(1)`，再做19个各task首题的公共诊断，而该循环不调用`after_update`；原stage1 core尚在，新预检原来的`core is None`断言会恒失败。现在只允许丢弃严格满足stage1、step/tau均0、尚未校准、无历史／活跃或已完成回放／终态的这个disposable调度器，记录`FOREVER_COMMON_PREFLIGHT_DISCARD.json`。公共诊断真实改变的权重保留，明确它不是已经训练完成的S1前缀；19题实际发生的证明由原外层preflight记录绑定，不凭此处的来源合同冒充实际完成记录。

root接受将原S2七步预检直接替换为**唯一T13派生25步检查**，不先做7步再叠一个门。保留原完整`self.train_tasks`、manifest及`task_by_id`不变，只向新core传临时19task view：原T1–T12分别按`SHA256(42:task:ID)`取首1，T13取前385，未来task为空；原selector实际选择过去12题中的memory，不按答案人为选择。view ID及摘要单列保存；公共loss仍经原全量ID→task映射读取原问题、图像、答案和指令。

预检以原optimizer／cosine／GradScaler／batch16、micro1执行真实25个current更新，首次24步自然校准后须实际完成至少一个dynamic replay，再完成正LR的第25步，最后`end_stage(13)`执行实际final replay并释放core。没有修改tau／threshold／beta；若自然覆盖不成立，门不通过，不通过改输入补跑。25个梯度／LR、385个current ID、动态→current25→final次序、真实memory与各replay曝光／额外unwrap NLL全记录。

仅在此预检里临时观察每次dynamic前后的外层optimizer／scheduler／scaler对象和state、Parameter对象／storage／requires_grad、gradient-checkpoint标志，要求不变；允许权重被回放更新，也不保存恢复整个replay RNG。这是检查合同而非已经获得GPU证据，正式方法没有新增该观察hook。当前更新完成后像原协议一样释放外部optimizer，再执行final回放。

原run保持字节不变，因此其随后固定的**T19首TRAIN probe**、完整checkpoint保存、独立reload／有限backward仍由原框架执行，不能称T13 probe。它们与原19个公共QA、385 current、实际每次2M replay训练和每次2个unwrap NLL均计费；门receipt只是先记录后续尾部费用仍须外层proof闭合，不宣称尚未运行的尾部完成。没有伪造前12个正式stage events或提交rolling checkpoint。本门不是科学训练轨迹或效果，0TEST／生成／F1；上述全部GPU及完整CP／reload验证当前均未运行，不沿用旧1%零触发proof准入。
