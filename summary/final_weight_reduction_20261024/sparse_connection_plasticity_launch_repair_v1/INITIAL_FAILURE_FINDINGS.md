# 首个剪后权重含非有限值：固定连接对照关闭

问题仍是：更少连接的共同起点能否承接原回答功能，以及允许保留连接学习是否值得。此次未获得这两个科学问题的质量答案；发生的是实际初始化数值故障。

## 实际结果

首次初始化worker3525229/start751210642于2026-10-24T03:48:06.326562UTC STARTED。实际trace在 `initialize.py` 的 `Packed2of4Linear(base, keep)`，由 `core.py` 的 `torch.isfinite(weight).all()` 拒绝。2026-10-24T03:54:20.343689UTC独立终态检查记录服务failed/MainPID0/exit1，controller3525203和worker均已退出；canonical queue为closed_failure。原STARTED预算未延长，没有人为终止其它进程。

按固定block和投影遍历顺序、唯一作者日志对 `time 0.58` / `error nan`，这是第0语言块 `gate_proj` 的第一次剪后打包。`prune_fixed`此前已通过每4保留2及全部删除位置为0的检查，所以打包前至少一个保留槽为非有限值。没有保存其数量、坐标、NaN/Inf细分、H、迭代权重或转换前后张量，不能补猜。

无完整block元数据、无CP0或任何checkpoint、无INITIAL_FUNCTION/COMPLETE/INITIAL_VERIFIED/最终VERIFIED。Frozen和Plastic均未STARTED；没有训练更新、DEV预测或Answer F1。737323040值及模式码仍是设计规模，不能报成实际最终减重。也没有通过或未通过log(2)质量投入门的结果——压缩后诊断尚未执行。

原十二源SHA与新READY5a1ca3...完全一致，原v1亦未热改。原启动前空白行解析已正常越过；本次故障位于实际模型权重数值检查，不能再借“启动前修复”裁决恢复。

## 新查明的实施覆盖缺口与根因边界

真实校准在外层CUDA FP16 autocast中执行 `formal.answer_loss`，forward hook同步调用作者 `add_batch`。作者先把输入转float，再以 `inp.matmul(inp.t())` 累积H；该钩子没有禁用外层autocast。H以Float32保存，并不证明这次矩阵乘法实际使用Float32输出。

[PyTorch v2.6.0官方AMP参考](https://github.com/pytorch/pytorch/blob/v2.6.0/docs/source/amp.rst)将matmul列为CUDA autocast至Float16的运算，说明函数、Tensor方法都遵循相应规则；显式dtype参数与先对输入调用float不是同一件事。[同版本分派源码](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/autocast_mode.cpp)按lower_precision_fp策略注册该类运算。因此存在具体的统计计算上下文未对齐，而不是仅凭“模型可能不稳定”的泛猜。

这是有源码和版本规则依据的执行路径缺口。它可以解释为什么FP32小图观察者/打包/反向一致性检查不足以覆盖真实初始化，但**不是已证NaN根因**：当时H、matmul实际输出dtype/有限性、条件数和每次补偿W均未持久化。现有原始权重SHA匹配和先前稠密前向loss有限，也不能替代逐权重/逐统计矩阵有限性证明。

作者先打印 `error nan`，随后才把补偿W回写原权重dtype；所以不能把这条内部NaN日志只解释为最后一次FP16回写。具体非有限权重首次出现于统计、分解、补偿还是回写，现有证据不能唯一识别。没有证据归因空白解析、2:4表达能力、数据标签、初始知识不足、显存不足或硬件故障，也不称作者算法普遍失败。

## 已发生的工作和费用

- 真实加载、原基座核验、上设备、视觉cache及初态前向已发生。由固定控制流到达位置可推得：稠密兼容检查2次NLL及2次短生成、16个稠密诊断回答NLL、第0块16个校准回答NLL已执行；合计34次teacher-forced NLL和2次短生成。这是代码到达位置推断，**不是新增逐次计数文件**。其中没有独立新增QA或最终评分，不能将这些前向写成0工作/0费用。
- 一个投影的作者剪枝调用返回并打印内部NaN，打包拒绝发生在物理新层构造完成前；0完整block/0checkpoint不代表原dense权重从未被临时修改。进程退出后不加载或恢复任何内存轨迹。
- STARTED到FAILURE文件mtime差30.415655124秒，仅为故障区间，不是完整模型RUN。作者0.58秒为内部子段，不能与该区间相加或当完整初始化时间。6条原同PID资源行是失败前采样，不报训练/推理峰或压缩内存收益。
- 一次小字节扫描写seal前0.006742234秒另列，运输、SSH、解析修复0.000761522秒子段、原小门4.147746754秒及所有历史费用保留。没有重新导入模型/重采QA/重门/重评分/bootstrap或大权重hash。

## 结论和行动

这次实施没有建立有限、可用的压缩起点，故Frozen/Plastic必要性比较与完整确认均不放行。不能据此说稀疏质量阴性、保留连接学习无用、某参数可删或方法成功。

按冻结规则关闭整个固定点：不重启新旧service，不续CP，不放宽finite检查，不更换精度/阻尼/校准集、mask、LR或层，不自动新版本或另一模型。原科学源码、READY、全部原始故障与费用保留；同一统计/精度覆盖缺口已完成静态核对，不再为活跃重门、张量重演或重复审计。

以后任何新的研究干预仍需有功能承接假说、最少对照、完整成本和正负出口，并与原停止规则一同独立评审；本报告不授权新的尝试。完整权重与原Answer F1主目标仍未实现。

[终态及小字节封存](INITIAL_FAILURE_SEAL.json) · [实际失败trace](phases/init/FAILURE.json) · [实际退出](phases/init/EXIT.json) · [原协议](PROTOCOL.md)
