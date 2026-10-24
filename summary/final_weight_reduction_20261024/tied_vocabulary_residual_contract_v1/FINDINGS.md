# 缩窄残差载体时，不能漏掉词表共享和末端归一化

## 问题与原有边界

已关闭的r200、FFN半宽、视觉尾和普通共享草案不重开。上一轮同问题配对可构造性已结案，本轮不扩成四图对象小测。当前只核查一个此前未闭合的结构问题：若以后研究更窄的语言残差载体，能否保留原模型已经共享的词表，而不隐性保存第二份大权重？这是功能接口和完整成本问题，不是数值修跑。

该问题不解释OSFT为什么学得好。已有强参照要求设计同时承接可用起点与适应能力，但不能据此证明残差缩宽可保质；本轮只补构造约束，不将SliceGPT或线性连接命名为新框架。

## 指定源码与本骨干事实

只读取Microsoft官方仓库固定revision `183cd92d8469bf3f1e979585a08dc382692bfbed`的三个小源码文件，不执行、不安装作者代码。原文件仅留私有目录，Git中只保存URL、SHA、定位和解释。

- [layernorm_fusion.py](https://github.com/microsoft/TransformerCompression/blob/183cd92d8469bf3f1e979585a08dc382692bfbed/src/slicegpt/layernorm_fusion.py#L83)的`fuse_modules`实际复制lm_head权重；随后将最终norm的增益融合进head。
- [rotate.py](https://github.com/microsoft/TransformerCompression/blob/183cd92d8469bf3f1e979585a08dc382692bfbed/src/slicegpt/rotate.py#L91)分别处理输入embedding和最终head的旋转/切片；各block的attention与MLP残差变换矩阵也实际另存。
- [modules.py](https://github.com/microsoft/TransformerCompression/blob/183cd92d8469bf3f1e979585a08dc382692bfbed/src/slicegpt/modules.py#L19)的RMSN按保存的`mean_dim`除平方和，不能仅凭新张量宽度自行替换分母。

这是本轮所取当前官方revision的行为，不补称论文v1的历史源码逐字相同，也不声称作者漏算或论文有错。论文的旋转不变性和删除后的质量保证仍按原边界区分。

本骨干当前小config的SHA仍匹配既有component_footprint证明：语言宽度896、词表配置152000、24层、14个query头和2个KV头，text_config要求共享词表，外层tie标志为false。既有头部记录只有一份embed_tokens权重键；本轮不重读权重头、不重新加载模型验证别名。config未显式给出`rms_norm_eps`，不能把作者RMSN的默认值直接当成当前模型实际值。

## 一个闭合端点的构造，而非完整压缩算法

采用行向量。令原共享词表为`E`（V×d），最终RMSNorm增益为`gamma`，保留宽度为r。令输入与输出投影`P0, PL`均为d×r列正交矩阵。保留原E一份：输入token先查E再乘`P0`；压缩decoder输出为z时，使用

`s(z) = sqrt(||z||²/d + epsilon)`，

`logits = (z/s(z)) C Eᵀ`，其中`C = PLᵀ diag(gamma)`。

因为列正交，`||z PLᵀ||²=||z||²`，所以该输出在精确实数中等于“先把z lift为z PLᵀ，再应用原最终norm/head”。这只闭合端点函数，**不证明压缩decoder的z等于原模型隐藏状态**。满宽正交变换时，端点可兼容不同首末基；整个decoder及视觉接口仍需对应一致的变换，不能从端点证明跳到整模型等价。有限精度累计、cast、GEMM路径也不保证逐值相同。

直接用除以r的普通RMSNorm替代上述除以d的标量，会改变归一化。本构造必须承接真实原epsilon，不能为跑通或追分改变精度/epsilon。

共同首末投影也不自动解决问题：若只在r维里处理gamma，再调用投影词表，一般会漏掉`PLᵀ diag(gamma) (I−P0 P0ᵀ) Eᵀ`。只有额外子空间条件成立才能消去，不从形状假设成立。

## 完整成本与去留

若最终只保留E、P0和已融合gamma的C，端点共`Vd+2dr`个权重；不能再重复保存可由这些矩阵替代的PL/gamma或第二份词表。训练若需独立参数化，临时和必需训练状态另计；所有内部残差桥、视觉连接器、newline及其它权重仍须计全。

公平参照是原模型已经只有一份E，因此**端点本身增加2dr，没有压缩原词表**，完整词表输出乘法也仍存在。与两份独立的投影词表`2Vr`比较时，它也不总更少：只有`Vd+2dr < 2Vr`才有该局部存储优势。即使成立，也不能把避免新复制的开销当作相对原模型的减重成果。真正整体减重仍必须来自decoder其它部分，并覆盖端点与残差桥的新增成本。

因此否决“首末独立旋转后仍按单份投影词表记账”和“共同小桥无条件保留原最终norm/head”两种不闭合方案；保留一份完整词表加两个连接矩阵作为可构造的接口工具。它没有承诺删除多少decoder权重，也没有解决有限步可学性，尚不能作为论文方法或模型准入。

本轮不新建模型协议/READY/service，不修复旧稀疏入口，不开权重/QA/图像/评分/bootstrap。没有新AnswerF1、实际压缩包、内存峰或速度成果。指定源码取得2.160096秒仅下载区间，config读取/SSH和审阅另计。这里补的是一个会改变结构成本判断的设计约束，完成后不反复计数、推导或检索维持活跃。
