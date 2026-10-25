# 固定容量宽QV：单次正式终态确认

本协议为可审阅实施版本，不是启动或模型成功记录。原草稿、成对调制实验和QV控制均不覆盖。本轮仅一个fresh训练和一次终态TEST，不改任何基线scheduler。

## 问题、候选与事前比较集合

原匹配QV-only VERIFlED（1b0a8f8df3e4f1a4f3338a439398cd9c3197930002302dd8e8a72015fb9fa99c）宏45.085962854378714，高于dynamic44.75760269784193和static44.68120094961681。依已声明宏高优先、精确平局简图规则，正式候选固定为bare wide-QV。100DEV已探索且单seed；调制包未产生正总体点增益，不再保留其有效性/创新主张，不按结果追beta尺度、rank、LR或步数。

本次只问：保留完整OneVision预训练功能、跨19阶段共享固定容量QV学习，能否避免永久专家银行而保留/提高正式总体回答质量，并实际降低完整权重和GPU内存？普通LoRA不是新算法；正式阳性最多先支持此条件下的经验性设计选择，不将被控制否定的模块重新包装。

人类允许各域得失，主量固定为五个完整域各自QA均值F1再等权宏；AP19为19任务等权的伴随量。不得按结果改聚合或要求各域分别胜出。比较集合在TEST之前由FORMAL_ANCHORS.json冻结全部8个已sealed正式行和原小proof SHA：同骨干OneVision最强PAS46.619976395158304；全体最强Qwen SeqFT-LoRA48.972351176657504（AP19 47.812159352619155）。同时输出same_backbone_gate与all_sealed_global_gate，用户“最强”总体质量门采用全体max。只赢PAS未赢Qwen只能报告同骨干结果/质量资源Pareto，不标总目标完成。后续新sealed参照出现，发布比较只能收紧、不改变候选或据TEST追加训练。

## 唯一全TRAIN学习与完整图

fresh原OneVision base，全部冻结1313 TRAIN、19阶段、seed42、单epoch、micro1累积batch16，共93次正LR更新；不续FIT CP。48个q/v位置沿原BudgetQV ranks65/66，4399104个FP32 A/B坐标、scale4、dropout0.1、同初始化，0beta/response_sign/额外Down/c校准。加载完整base后先导出base-only初态并自包含strict reload，无新初态DEV/TEST生成。初base state/buffers/layout/config/generation/alias绑定封存QV控制及原校准receipt；reset42→48AB初值与封存QV精确相同，再reset42→原训练AST。

原stage-reset schedule/warmup、GradScaler、clip1、answer512+EOS保持。GPU AdamW lr1e-4、betas(.9,.999)、eps1e-8、wd0、foreach/fused false、capturable true，阶段重建。FP16 base与外层CUDA FP16autocast、原SeqLinear FP32存储语义和SDPA公开；不声称适配器全程FP32计算。模型前后向/梯度/optimizer及GPU FP32禁autocast/TF32折叠均GPU；继承初始化器的CPU随机参数生成与checkpoint I/O不算CPU模型前向，无activation offload或CPU模型门。

训练只读TRAIN，不读TEST QA；先前100DEV属于原正式TRAIN，本次合法加入训练后不再是独立验证。保存93次实际梯度/事件、19阶段A/B与有效更新，96梯度对象17596416bytes、optimizer35193216bytes，原冻结参数/buffer不变。48QV合并回FP16后完整图893675552系数/1787351104权重字节，普通buffers9032bytes与完整包字节另报；最终不保A/B、optimizer、教师或外部base。

## 数据和图像绑定

沿原pct01 manifest（36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777）、split_lock和frozen_ids，逐TRAIN/TEST jsonl校验原SHA与冻结ID，1313TRAIN/21454TEST无QA ID交集。原split lock不直接锁图像字节，不能作相反宣称。TRAIN沿原SPLIT.image_sha256；TEST由原ROOT完整数据的images列生成一次path→SHA ledger，5958唯一TEST图像全部可见，计该I/O成本而不读答案选样/评分。每进程对图像首次使用核SHA后cache该验证，未将图像改写。

旧PAIR asset_view缺正式TEST文件/lock/影像，不能原地补旧冻结view。准入建立新独立asset_view，链接原完整数据和必要base/source，核冻结原字节；不重复旧CP hash、不把283文件历史proof map全部当runtime map。源码/环境/manifest/lock/IDs/训练图像表/TEST图像表及prior小proof在READY绑定。实际TEST QA只在train_COMPLETE后由固定分片读出进程加载。

## 独立资源窗与四分片盲TEST

train结束后先独立resource进程，在原PAS测量同一OLD物理GPU1/UUID，自包含fresh重载同一完整末态CP。使用原五个已探索TRAIN资源ID、同顺序、raw/transport/prepared/shape完全匹配、无视觉cache、每窗32新token，窗口含准备与生成，记录baseline/peak allocated/reserved；0资源F1。PAS锚为strong_pas_resource VERIFlED c987ec2ac68980944a1ebdc515fab08a792667f923d142f6f8489d3725753d08，其COMPLETE/memory链一并绑定。正式末态实际测量不由FIT的“约25%”代替。

随后OLD四张真正空卡各加载同一个末态完整CP，eval模式、加载后seed42，batch1 greedy（do_sample=false,num_beams1,use_cache=true）、原EOS/pad/任务cap、401408面积/AnyRes/检索/模板不变。四shard按SHA256(`formal-qv-test-shard-v1:{task}:{id}`)完整digest的大端整数mod4唯一分配；各shard内部保持原任务和QA相对顺序。每题独立generate，无跨题KV/学习/可变状态。保存实际token IDs、完整输入摘要/参考与CP/身份证明，不打分、不消费partial输出。

Greedy关闭采样和dropout，使分片无须模拟抽样RNG流；seed42不能保证跨GPU/CUDA kernel逐token bitwise一致，当前不做额外全TEST重复来声称一致。固定实际硬件/软件/topology，不能按结果换卡/换shard。四份驻留只为离线评价吞吐，不称端侧部署形式，资源主证据仍是独立GPU1单图五窗。

只评价末态21454 TEST/19cells：natural5000、medical2094、geography10004、finance2233、traffic2123。四shard全部成功、ID恰覆盖且无重复/遗漏、原task/参考/输入/cap正确后，按原canonical顺序唯一调用原Answer F1 scorer21454次。先按域完整QA数合并，不平均shard均值；AP19另算。无初态/阶段TEST、TEST NLL、bootstrap或FWT/BWT；最终19cells不冒充原74cells，不拿TEST选备选/超参或救配方。继承FiniteScores只验证原logits有限，不改写分数或改变greedy选择规则。

## 固定预算、失败与成本

train首次STARTED起3600秒；resource首次STARTED起600秒；四个test_i各首次STARTED起14400秒并发，总上限61800 GPU模型秒。逐进程实测秒求和与controller墙钟分列，不能把并行重叠计成实际墙钟。所有加载、摘要、训练、保存重载、5资源生成及21454盲生成计入对应phase，不延长或重置。旧探索、诊断/调制、QV控制与历史故障费用另列承接，不消失、不当本轮新计算。

controller先train→resource，再同时启动4个隔离PGID TEST worker，metadata轮询所有子进程；任一实际失败/缺COMPLETE即停止本批其它已绑定UID/starttick/PGID的存活worker，保各EXIT/partial与成本，绝不向他人/基线发信号。Restart=no、不enable、不resume/静默重启；实现故障是质量缺测，不作科学阴性。必要透明纯实施修订另行明确审判，不能按质量改变数值或配方。

## 结果出口及边界

正式五域宏严格超过OneVision最强只过同骨干门；超过全部冻结sealed最强才过用户总体质量门。各域可下降，全部得失保留；精确平局不称提高。单seed点估计不证明显著/稳定/非劣。负结果同样完整公开，不回退dynamic或追加另一TEST。

完整系数/实际dtype权重字节、mask/buffers、包、GPU窗峰分别报告。PAS原混合dtype/必要动态专家/原CPU保护历史保留，其metadata字节不是RSS；低于该原实现不能全归架构或称同dtype最优部署。OSFT常规fold可回原base，不能拿其USV保存态宣称bareQV结构更小。

当前实际显存锚只有PAS，最强全体质量锚却是Qwen。即使正式过global质量且低PAS峰，也只能支持“global质量阳性+原PAS资源优势”；不能直接证明比最强Qwen实际GPU峰低，更不能自动标用户联合目标完成。若该全体质量条件实际成立，Qwen原完整图的公平资源证据才是另一个具体缺口；本协议不擅加该模型。完整权重少不替代GPU峰实测，五窗不代表全分布最坏峰，也不报告未测延迟/功耗。

若只是普通QV在固定条件下保质省容量，论文贡献需诚实定位为经验性共享容量/有效适应发现及成本证据；不能包装新算法、恢复调制有效性或声称专家普遍无用。执行/准入/源码审阅不算科研收益，科学总目标由实际正式质量和相应公平完整成本共同判断。
