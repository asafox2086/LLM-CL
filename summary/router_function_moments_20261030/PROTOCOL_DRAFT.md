# DRAFT — 原动态模型的路由与专家输出联合性质观测（未准入）

2026-10-30。本文件不是模型运行许可。没有运输OLD、绑定、排队、GPU模型或commit。CL四臂暂停不变；本观察不拟合、不替换模型、不QP、不生成、不CE/F1、不更新参数。基线ENOSPC处理及冻结PAS完整确认优先，不能抢冻结槽或以观察为由中止。

## 可证伪问题

在同一历史TRAIN样本上，输入相关路由变化是否主要落在专家输出的低敏感方向？不同机制PAS（A能量）和CL（learned token + question anchor）是否共享该性质？这是待观测问题，不是已知结论或新方法。高F1常答案只支持检查条件功能，不能证明参数冗余。

## 冻结范围及尚缺的运行合同

拟复用原PAS探针事前固定的80条历史TRAIN QA（各域配额及对象/顺序/来源必须在运行绑定明确列出并核原digest），PAS与CL的S19原完整CP及原实际推理图，分别严格一次加载、每QA一次原teacher-forced forward。CL原question feature所需额外原decoder前向保留且单独计数；不把它混成一次总成本。原80条曾参与训练且含答案回访，不能称未见验证集或泛化实验。这里只规定纯数学接口；worker、CP/源SHA、80对象、GPU槽、loss-mask对齐、实际hook闭合及数值容差均尚未绑定，必须独立审查后才可能运行。

答案位置来自原causal loss移位后的预测位置，含原loss计分EOS；不是选择答案已作为输入之后的位置。每QA先对其选中位置平均，再对QA等权，各层同样的QA集合。不得读TEST或根据TEST选择层/样本。原eval/no_grad、dtype、scale、mask、真实有效复合gate保持。记录layer及domain，全部层完整报告，不看数后挑层。此范围仅teacher-forced历史TRAIN答案预测位置，不包含自由生成decode行为。

## 原始充分统计

U(x)为按列组织的实际专家输出，包含全部推理scale与mask；g(x)为原实际有效复合gate，t(x)为原路由分支实际增量，b(x)为原base投影输出。逐QA记录选中位置、token数、g一阶均值、完整E[gg^T]（CL为8×8）、H=E[U^TU]、h=E[U^Tt]、a=E||t||²及E||b||²。保留原矩、计数和有限精度闭合误差。所有矩必须是同一位置集合和权重下的真实联合统计，不能将E[g]与E[H(x)]相乘冒充h/a。g的有效性及t=sum(Ug)必须与原分支输出实际闭合，不能仅检查新公式内部自洽。

纯统计接口接收已经detach的CPU数组，内部FP64累加；不持有模型/GPU激活或checkpoint。接口无法证明调用方loss-mask与原分支取值正确，这些是独立worker审查义务。每QA原始记录先耐久保存，闭合失败保留现场且拒绝解释，不自动重试。

## 事前固定描述（不优化c）

每层使用全80QA的c=E[g]，不是优化解。Rjoint=a−2c^Th+c^THc，正是均值gate参考与原实际增量的均方差（以实际数值闭合为前提）。Rindep=tr(E[H(x)] E[(g−c)(g−c)^T])是将专家几何与路由变化解耦的独立参考；全体时第二项等于Cov(g)。分域依旧同一全局c，因此分域项是偏差二阶矩，不错称域内协方差。另只以同一已记矩计算预定uniform参考，不做任何新模型forward或数值求解。

报告Rjoint/Rindep、相对routed-update及base能量、gate变化和H，原值不clamp、零/负分母NA。能量是平方范数，不能称相对范数。FP64消减误差尺度单列，运行前必须按原dtype声明原分支数值闭合容差；无科研“成功”阈值。H/h/a重构R需在一个事前固定的小窗以直接t−Uc的平方范数独立交叉核，纯统计函数不实现该模型hook审查。

## 判读边界

若Rjoint明显小于Rindep且高于数值噪声，支持此样本上“路由变化与专家输出几何负耦合”的局部性质。gate近常量、专家原本相近、动态选择落在低敏感方向须分别辨认。Rjoint大仅说明均值gate参考不准确，不否定存在别的常数c；不得推出所有静态化失败。小Rjoint不证明删参数后全模型功能/F1不变，局部扰动可累积，历史TRAIN可能不泛化。PAS和CL不一致就否定共同性质，不用新臂/改样本救解释。两种router仍不等于全部持续学习；Seq等静态结构的R=0只是定义控制，没有加载即不得计为第三法实测。

## 准入仍需

独立source review、真实hook合同、原CP/80样本/源码绑定、数值小窗交叉核方案、真实空闲合法槽和时限/no_retry/不干扰基线保证。禁止直接执行草稿来绕过CL暂停或恢复新方法路线。当前文件无正式绑定，任何模型运行均未获本草稿准入。

## 源审补充：中心化与真实分支闭合

原始矩另存μ=mean_e U_e、V_e=U_e−μ、s=t−μ的增广特征[V,μ]矩及h/a。评估c时使用增广系数[c,sum(c)−1]，严格保留原gate sum有限精度误差，不重新归一gate；同一R的原/中心化表示差值单列。存contrast energy=E sum_e ||U_e−μ||²及uniform direct residual。它们区分专家近同、gate近常量、变化与几何耦合，仍不证明删参质量。

worker_draft.py仅hook/helper，无资产加载和可运行CLI。TorchDispatchMode钩原模块中首个真实softmax、B模块实际输出，CL复合gate按原运算合成，PAS原gate直接截取。t并非可直接hook的内部张量：它按原分支操作重构，独立重算同形base并以原模块最终输出闭合，明确记重构来源/误差；PAS和CL不同加法/cast顺序保留。额外base GEMM、统计/CPU复制/dispatch费用单记，不称原生部署费用。只有原最终输出闭合通过的QA才允许解释；尚无实际模型验证，dispatch算子名、原源/桥接口和严格数值容差必须审阅。

## 执行前数值合同修正

原layer输出采用完全相同运算顺序的重构，要求bitwise闭合（atol=rtol=0）；这不用于FP32分支与FP64数学矩的比较。主矩目标改为t_ideal=FP64ΣUg，原FP32重构delta能量、两者差和逐元素算术证书另存。证书无自由阈值：γ_(2E+2)Σ|Ug|+(2E+2)×2^-126，其中FP32 unit roundoff=2^-24，最小normal项保守覆盖FTZ。任何超界为失败，不抹原误差。Rjoint/Rindep为理想有限数值U/g的函数性质，不能冒称原层直接delta；实际输出闭合和roundoff证据用于约束解释。额外保存QA内直接中心化gate协方差及增广[g,sum(g)−1]协方差，以均值差的全方差法组合，避免gg−meanmean在近常量gate时假造方差。

实际collector在GPU FP64做矩乘，仅小矩/标量移到CPU；numpy模块仅数学参考和离线描述。run.py拟两进程串行、每进程一原CP加载、80QA一次hidden forward；CL原feature额外前向单计。模型hidden输入逐项复用原answer_loss，省略lm_head/CE。每QA168行统一flush/fsync后写独占QA完成收据；失败若已有当前QA行，单存FAILED_QA_PARTIAL保留，不可用于完整聚合。原CP/来源源SHA、80TRAIN选择、原桥、loss-mask、实际GPU UUID和独立ADMISSION均需绑定；BINDING_DRAFT.admitted=false阻止执行，无自动retry/resume。

## 最终执行前明确合同

PAS原fan_in_fan_out必须False，helper初始化硬断言；否则不以F.linear原weight重算base。真实collector硬断言expert输出、有效gate及原操作重构delta均为torch.float32，才适用固定γFP32证书；其他dtype直接拒绝。当前只准NEW→OLD独立源运输及不加载模型的两法metadata检查，admitted=false，不准--execute。最终独立source/admission/controller由root负责，不新增allocator。
