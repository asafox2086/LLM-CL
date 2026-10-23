# 读取首个更新前loss，约束压缩失败的时间位置

问题：已完成r200两流的严重质量失败，是否只能在阶段末归并/后续遗忘之后出现损伤？检查第一次optimizer.step之前、首16个micro样本的已记录回答NLL，与既有合法FIT-only LoRA_qv的首条同定义标量比较。

这是新且有用途的既有artifact问题，不是重评分或重做先导。此前未隔离初压/适应/归并根因；本分析只定位最早可见的损伤证据，不检验最终F1因果。

执行前固定：只消费三条首记录及其既有封SHA日志、READY/COMPLETE/旧VERIFIED、必要静态源码。检查原manifest/SPLIT/训练AST、共有基座文件哈希、公共输入源码、首条stage/step/QA/prompt与target token计数和loss定义一致；若发现冲突或封链失败就停止跨条件解释，不改条件或新增模型探针。样本顺序由同SPLIT与原确定性TRAIN源码支持，原运行没有逐批input digest，不能补称逐张量输入已验证相同。

在读取任何loss前的本地元数据预检发现：旧READY仅封safetensors，新READY另封14个preprocess/config/tokenizer文件；共有权重和源码无冲突，但旧历史预处理哈希缺测，不能补封冒充历史。review_next_probe再次接受一次**描述性**消费：strict_input_and_preprocess_match=false、controlled_causality=false，只报告同声明配方/QA与token计数下首批日志差异的时间位置，不能认定压缩致损。该解释范围在读取结果前收窄，原实验及其门槛不改。

loss_sum是每个QA内回答token平均NLL再按本批16QA平均，记录虽在step之后写入，值来自该step之前的前向。两方法无额外loss项，stage1还没有阶段末归并；新残差初始A=0、旧LoRA零输出初始化不代表二者整函数相同，压缩基本表示本就不同。核静态定义，不加载/执行模型。

正向用途：若r200在首次optimizer更新前首批NLL已更高，说明日志差异在归并/后续遗忘之前就存在，提示新设计应先关注起点功能承接和输入可比性，而非直接把全部问题归给后续保护/归并。无差/较低/无法匹配则不能用此支持起点损伤，保留其他解释。只报告数值，不设显著性/等效阈值，不外推首批到全部五域，也不将NLL当Answer F1；由于历史预处理缺测，不能写成已证压缩损伤。

一次标准库CPU消费，预计小于10秒（SSH另计），0模型/张量/QA/生成/optimizer/GPU/原score或bootstrap。只读已完成小JSON/log，不加载CP/复算旧质量/重复旧梯度能量分析，不改科学源/READY/门槛，不恢复任何worker，不据已见DEV/TEST调rank/LR。

review_next_probe已在执行前接受此一次有限比较的用途及边界，明确要求不称显著、无逐批input digest时不称完全受控、不能唯一归因SVD或最终F1。结果和原运行额外费用保留；若本问题已回答不再重复消费。
