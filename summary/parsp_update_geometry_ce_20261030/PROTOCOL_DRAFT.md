# 原PaRSP一次真实更新的约束几何观测（草稿，未准入）

问题是原始稀疏选择M、历史协方差投影P与真实Adam更新是否保持同一几何约束，当前不检验学习损害，不设计新方法。正式基线原图、源码、原RUN/latest/S19及所有原checkpoint不改。

使用OLD真实10% PaRSP完整S18：SHA0b704968b1314595f5382464a1acc6cdaabc600da6a9e560692583040d992d06，5928091794B。已做符号metadata核查：stage/method.stage均18，96组历史协方差、原runtime/source与完整RNG存在，25源匹配。独立load_RUN只hardlink同persistent S18 inode，复制原stage18 metadata为自己的latest；零新增payload。原loader的resolve-containment/fullSHA/expected-config检查与RNG restore不改。研究锚link在完成前需受旧checkpoint清理保护。

这是相同完整S18起点和原begin_stage19的独立条件回放，不声称原stage19逐位重演。将原NativeIO重绑到诊断输出及同T19 TRAIN，避免原RUN写入；原io.cache必须为空。恢复原saved next-stage RNG后调用一次原begin_stage19。严格原localizer预算是hash序前256QA（NativeIO本来的预算），不是把492QA全量替换进去。按原manifest/hash顺序取原T19首16QA，batch16/micro1，一个完整原生step。

执行原模型gradient checkpoint、原activation offload、FP16 autocast、scale初值1、原HostAdamW/eps/betas/weight_decay0、新stage optimizer、LR3e-5/warmup0、全134195328原selected parameters、原unscale/全局clip1与scheduler顺序。只训练一次，模型不end_stage、不评估、不生成、不保存新checkpoint；在原step结束后仅添加同16原TRAIN的配对CE前向读出，仍不宣称学习损害。

原begin_stage19结束后，以当次原生选中的最小MLP层为target，记录其down_proj列mask与同层gate/up行mask对照。不依据结果改层号或M/P。S19历史元数据的最小层只是来源参考，不强迫独立回放的选择逐位一致。

对三个原weight各原两hook（mask hook_fn，然后project）只包裹输入输出：每micro原函数恰调用一次、原顺序与算术返回不变。采集rawG、M后、P后，16micro FP64累计用于只读统计；native原grad仍原FP32累计。记录其与实际preclip grad的逐元素FP32 gamma15舍入闭合。记录原scale、pre/postclip梯度及真实FP32参数before/after的dense deltaW；不是用gradient norm猜update。

GPU FP64统计：P对称/幂等残差；列mask的MP-PM交换子范数；各梯度/真实deltaW的M外能量、非零数、最大泄漏坐标与该坐标真实before/after；x-xP残差（P近似投影，不能冒作正交能量分割）。rowmask的结果作为负对照记录，不把理论预期预设为观测成功。不上SVD阈值求交集，也不由40/20比例认定实际交集0。

记录raw当前TRAIN CE梯度与实际deltaW的target分量内积、单位步幅内积与余弦。三target仅为整次原生更新的一部分，不能称全模型loss变化。几何观测不等于有害/有益或普适性，OSFT独立负对照未启动。

执行入口默认只source/metadata inspect，不导入Torch或加载模型。真实GPU槽/UUID及root+lit源审后才可正式admit。模型硬期限7200秒，独占输出/no-auto-retry；失败保现场，不改超参、数据或捕获器后悄悄重跑。仅小JSON证据与原auxiliary记录，原checkpoint不复制到Windows。


## 首GPU前新增CE-only观测（新独立指纹；原草稿原样保留）

原a1b/db6f/c38/774/910 source、协议、binding与metadata不覆盖。该版本只给已审原单step添加配对CE，不加关保护/新方法/全134M rawG或参数clone。原256localizer、16backward、全selected参数的原optimizer/scale/clip/LR/RNG顺序不改。

原每micro在input_batch/prepare前保存完整原RNG，并在CE入口保存prepared batch张量字节SHA、RNG指纹、model buffer小指纹与各module train flags；原CE和target长度取自既有forward。原step后保存真实post-step RNG及全registered参数的版本号（不是权重clone）。

保持原model.train与grad-enabled，按每micro原prepare前RNG重放相同input_batch/prepare/activation_storage/autocast；逐QA forward图立即释放，没有backward、optimizer或scheduler第二更新。readout计16，原one_step计16。finally恢复真实poststep RNG并实核。参数版本应无第二次变化。

NativeIO.cache属于localizer缓存，本step/readout直接走原input_batch；视觉LRU可改变命中或成员，不声称其原态恢复。是否strict pair必须根据实际prepared张量、CE入口RNG、buffer、mode、target长度全部匹配判定，不能只以同QA宣称。若不匹配照实保存原16before与mismatch项，不修复输入、不改eval/no_grad、不据此声称同随机目标严格配对。

输出16个idSHA/numeric CE前后和均值差，保留strict标记与版本/RNG证书。它衡量全原生step对同TRAIN batch的有限变化；三target只是全更新的一部分，不把CE变化因果归于目标泄漏，不外推收敛、泛化或F1。

原真实S19 auxiliary记录begin_stage19为777.356771226041秒（256localizer），原首16QA step为21.711140132509172秒。7200秒硬父期限另含本次load/SHA/统计/16额外前向，不把历史时间当新运行保证；无自动重试。
