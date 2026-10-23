# S1起点功能承接诊断：原视觉尾13块能否由1块承接

本协议是一次可推翻的必要性诊断，不是新框架、完整F1实验或原创性主张。不是优化OSFT/PAS；不恢复任何旧条件。完整权重和原回答功能是主问题，训练状态只是费用。

## 问题、依据与限制

强参照共同冻结原视觉接口，OSFT既有优势更偏刚习得而非更好BWT。这提示保留语言原表示与有限步适应空间，但不能据此证明视觉尾冗余、语言参数全部必要或任一机制已被隔离。r200语言压缩先导在首次更新前已有大NLL差距，故本次先检查另一载体的起点功能承接，不直接投入19阶段训练。

假说：同输入下，前13块保持原特征接口，用一个独立的原第14块副本学习原后13块的投影后空间token函数，能在有限合法S1数据下缩小未参与蒸馏组的回答NLL缺口。原前13块/连接器/完整语言冻结；学生学习后也冻结。复制块不得与教师尾参数别名。该13→1是单个事前成本点，不是已知可删深度。不搜索深度/宽度/LR，不以“冻结”证明无用。

普通继承与视觉蒸馏已有EfficientVLM、TinyCLIP先例，非新方法：
[EfficientVLM](https://aclanthology.org/2023.findings-acl.873/)、[TinyCLIP作者训练说明](https://github.com/wkcn/TinyCLIP/blob/main/docs/PRETRAINING.md)。其大规模训练不保证本1%/单域恢复成功。本候选只检验具体功能替代，不接受新颖性。

## 数据与最少对照

只打开原manifest的T1 TRAIN文件，使用原SPLIT中的327个S1 FIT QA/325已知组。T1原TRAIN容器含历史DEV行，读取容器仅按既有FIT ID过滤；蒸馏、诊断和选择均不使用任何原DEV行或分数。不打开未来TRAIN文件、DEV预测文件、官方TEST或正式teacher。325已知组不是325统计独立图像，未知近重复/关联未保证。

按SHA256('42:visual-function-s1:'+group_id)排序，前65组只诊断，后260组只蒸馏。每组选SHA256('42:visual-function-representative:'+QA_id)最小的一个原QA。只按ID/已冻结组选择，不看答案/损失，不改原FIT/DEV归属。诊断回答不参与蒸馏。只读teacher原预训练视觉函数，非见过DEV训练的正式1313 teacher。

三个功能状态：完整冻结父、未蒸馏浅学生、蒸馏后实际FP16浅学生。同65代表、同原processor/面积上限401408/AnyRes/空间token顺序、同原answer_loss的原answer与EOS。不是原AnswerF1，0生成。父NLL只运行一次，后观察复用父标量并核对原父全参数digest及同输入原tensor SHA；原始token/pixels/size SHA绑定一致。共195次回答NLL前向，130次学生特征诊断前向。教师空间特征前向另按真实cache命中记录，不将缓存payload说成完整RAM。

## 唯一蒸馏配方和预算

260组代表按分组hash顺序，仅一遍，batch16/micro1，17个正LR更新。只学习独立学生块，FP32 master、外层CUDA FP16 autocast；初/末诊断均使用FP16学生，最后固定转回FP16，不做精度搜索。CPU准入只FP32微型图，不证明实际GPU数值或质量。

目标是**实际projected+packed空间token**的每QA均方误差，除以同QA教师token平方均值clamp_min(1e-8)；再按QA等权。包含原newline，无CLIP pooled embedding替代。QA文字/回答不参与KD目标，patch token不能算独立训练图像。原prefix、projector、语言与教师尾全冻结。

HostAdamW lr1e-4/eps1e-8/betas(.9,.999)/decay0、seed42、原floor(.03*updates) warmup/cosine、clip1、GradScaler(init_scale=1.，与原formal一致)与CPU saved-tensor offload。教师参数必须全程不变；每步loss/LR/gradient、真实optimizer IDs/grad/moment bytes立即原子JSON+fsync持久化。原公共视觉LRU上限128MiB是训练缓存，不是整机RAM上限。READY记录实际解释器与实际导入HF模块文件路径/hash，worker在模型加载前匹配，不能凭另一个环境的安装路径封链。

KD时student独立挂在父model之外：ResourceMonitor的model_parameter_bytes字段只数完整冻结父，另以TRAIN_STATE_SCOPE列出学生FP16初态/FP32训练/FP16终态权重与可训练字节，逐步optimizer证据另列grad/CPU moments。进程RAM/CUDA观测实际包含两者，不能把父字段当训练总权重/总内存，参数字节和不代替实际峰值。

首次STARTED至COMPLETE上限1800秒，含加载/260KD/195回答NLL/特征诊断/参数digest/保存完整图与重载/资源观察；必要CPU准入、独立核验另列。不是用户要求或1%定义，不称同计算预算。一次STARTED后失败/中断/超时即关闭，不补恢复/续CP/延长/自动v2。只有STARTED/CUDA前资源变化75是未尝试健康wait。研究仅NEW物理GPU1一流，不抢占别人，不用GPU0/OLD闲卡，最多占NEW三卡；正式基线不动。

worker内部alarm之外，独立controller按STARTED同PID/startticks绑定的monotonic_start设deadline。到界只对自身新会话worker经UID/startticks/PGID核验后发TERM，最多5秒回收后仍活才同身份KILL；终止回收延迟不是追加模型预算，不能封存超时COMPLETE。若身份不可确认则不发信号并needs_attention，不重启或接续。任何正式/他人GPU均不得信号。

## 冻结判据与行动

65QA内原answer token平均NLL再QA等权。G0=mean(浅前NLL−父NLL)，G1=mean(浅后NLL−父NLL)。均使用最终实际数值边界，不以master训练loss替代。

只有实际完整权重≤原893675552的80%、G0>0、G1≤0.5G0，且同65组归一化feature MSE下降，才进入后续必要性评审。50%是作者事前投入筛选阈值，不是保住50%能力/用户允许掉分或F1保真阈值。G0≤0时记“蒸馏必要性门不适用”，不改比例或换before/after救门；微小G0比值不包装为大恢复。

所有原始逐QA NLL/差距/逐组MSE/成本/实际参数量公开（只ID/标量，不Git存QA正文/图像/权重）。失败只关闭本固定点，不否定所有视觉替代；成功不证明跨域、持续学习、非劣/稳定、完整质量或新颖性。不自动开1213/1313/21454，不填Ours/FWT/BWT。

最终紧凑图物理移除原尾12块；语言/视觉/连接器/newline/冻结/必需buffers全部计数、共享存储去重。teacher/optimizer/历史cache不得残留序列化对象；完整图保存与重载核对14视觉块/完整参数digest。仅训练期教师与全部额外费用分列，不能藏到模型外逃计数。
