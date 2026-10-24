# 小视觉先验的S1空间接口能否承接已学语言功能

## 研究问题与收益来源

固定一个已有小视觉先验，只用S1图像拟合一个空间/通道接口，不改变已学语言参数，能否保留未校准域的完整回答？这是一次跨域功能承接验证，不是已完成CL框架，也不解释OSFT优势。原来的小塔/半宽/语言删层各固定点继续关闭。

OneVision原视觉397747744系数；现有SmolVLM500中独立视觉86433024。候选删除原整个视觉塔和1836800系数projector，仅保留学生塔、768→896仿射桥和原语言/newline。完整设计893675552→581213056系数（约减少34.96%）；需以真实完整包及资源验证，不能按占比认定视觉冗余。

已有DIVE-Doc用蒸馏小视觉塔并复用decoder，包含较多视觉训练和后续微调；不能把它视作几百图/只学小桥的证据。既有技术可借用，本次贡献边界是该少数据跨域功能问题的实际答案，不是换encoder本身。

## 固定模型与接口

- 父是合法1213FIT/19阶段/87更新的S19 wide-q/v静态FP16完整模型，原宏43.791987仅为已有点。保留的语言已经见过全部FIT课程，整个候选不能称仅用S1训练。父源绑定原COMPLETE/VERIFIED/静态CP。
- 只抽现有Smol500资产的vision_model，预训练权重BF16转FP16，两者均16位存储；不含Smol语言、connector或processor，不修复旧Smol整模型分支。学生12层/768/patch16/512，视觉固定eager，原语言sdpa。学生输出取post-layernorm的last_hidden_state。
- 学生来源实际为SmolVLM-500M-Instruct，已有其原始指令训练历史；不是新收集外部训练数据，但也不声称两塔预训练历史相同或全部系统只用S1。这是固定已有先验的替代实验，不能将差异纯归为参数删减。
- 继续原OneVision的图片面积限制、AnyRes切图和384×384归一化crop；同crop内容用FP32 bicubic/align_corners=False/antialias=False插值到512，再转FP16。学生32×32×768用FP32 bilinear/align_corners=False/antialias=False缩回27×27再转FP16。该重采样是候选的一部分，不声称纯参数差异。
- 仿射桥把768维映射到原projector之后896维；原projector被Identity替代。保留原27×27空间顺序、pack/unpadding/image_newline、文字token/位置/生成规则。空间尺寸相同不代表语义已对齐。
- 同时更新core及PackedVision注册引用；最终不得暗留旧塔/桥/教师/校准统计。完整包独立进程按新结构严格重载，核所有权重digest/别名/布局。冻结学生和全部语言/qv，0backward/optimizer更新；岭回归仍是学习，不称零训练数据。

## 数据与唯一拟合

原SPLIT中S1的325个已知FIT图像组，以sha256('vision-interface-20261025:'+group_id)排序，前293组拟合、后32组诊断；每组取字典序首个S1 FIT QA，保持原图/预处理。若实际组数不同，模型启动前关闭提议，不换集合追数。组只由既有分组定义，不称独立分布样本。无DEV/TEST/未来域图像统计、外部新训练数据或正式1313 teacher。

325图文预处理各触发原视觉/学生各一次，0decoder前向；只图像特征用于拟合，答案/文字不进回归。每QA对全部crop×729个位置等权，其总贡献归一为1；保留包含padding crop位置的全部token，与最终有效pack范围并不等同，公开此限制。

以CPU FP64流式积累sum(X)/n、sum(Y)/n、XᵀX/n、XᵀY/n，每QA权重1；X是学生固定网格768维，Y是原实际CUDA FP16 projector输出896维。中心化后一次求解 `(Cxx+λI) B=Cxy`，λ=1e-3*trace(Cxx)/768；b=meanY−meanX B，截距不罚。桥W=Bᵀ及b转FP16。无调λ/轮数/型号/桥容量，无学生或语言梯度。不缓存全部特征，只当前一QA与小统计矩阵。

32组只在拟合完成后报告FP16实际桥特征误差/目标能量，不能据此改候选或停止掉分案例的全图评价。局部特征误差不是F1证明。

## 完整评价、资源与出口

父和候选各fresh进程加载各自完整图并核digest，原100已探索DEV各生成一次、各一次gold NLL；0正式TEST、0bootstrap。各域20题，DEV已探索/单seed，不称留出保证或非劣。保留所有原参考，原token_f1唯一评分。prepared视觉tensor应不同；文字/位置、图像SHA、原裁剪像素及顺序、pack长度和生成cap必须绑定。

每图另对五域各一个固定hash最小ID做恰32token生成，batch1/无视觉cache，含图片预处理/视觉/prefill；记录torch峰allocated/reserved与请求前值。两个全新进程同卡，同workload定义；不能从权重推峰，不承诺速度/CPU/NVML收益。

父每域点F1均保留/提高、完整权重及文件都更小、五窗口实际GPU峰均降低且完整性通过，才保留该接口进入独立CL必要性评估。任一质量不满足则关闭本固定小塔/重采样/仿射桥/S1预算，不更换桥/增数据/解冻塔/救分；不否定所有视觉替代。通过也不自动填Ours、全1313/21454、宣称语言CL独立或新颖性。

只OLD一张真正空卡固定同卡串行fit/parent/candidate，每phase首次STARTED后1200秒、总3600模型秒；含加载/提取/求解/校准、完整保存重载、评价与观测。CPU必要门、资产提取/运输、核验及历史成本另列。首次实际失败/中断/超预算关闭固定点，不续CP/延时/自动版本。保全NEW两个正式基线，至少留一张NEW空卡。

来源：[DIVE-Doc官方实现](https://github.com/JayRay5/DIVE-Doc)、[作者模型说明](https://huggingface.co/JayRay5/DIVE-Doc-FRD)。尚未激活，无新质量或资源结果。
