# 小条件差异来自什么：既存完整功能矩的精确描述性分解

## 直接结论

同LLaVA、1%正式S19、同80历史TRAIN、168投影，CL相对PAS的局部差值平方比高282.3345倍。这可以严格拆成 **43.6507倍的专家能量加权路由二阶变差尺度 × 6.4680倍的剩余实际归一敏感度**。因此，小路由变差是较大的描述性尺度因素，但“仅因router几乎不动”不足以说明全部差别。两模型实际input联合关系都在放大、而非总体抵消差异。

这不是因果贡献百分比。两模型专家数2/8、隐藏轨迹及混合不同；四因子是恒等式，不是四次干预。参数数量不同也不是本次差异的随机处理。

| 描述量 | PAS | CL |
|---|---:|---:|
| 新增神经权重（已核原账） | 140,771,328 | 37,158,912 |
| R/U，局部差值平方／原动态增量平方 | 0.0001045687294 | 0.02952336273 |
| V=I/T，expert能量加权gate二阶变差 | 0.0003658322603 | 0.01596882584 |
| R/(UV)，实际残差除去上述尺度 | 0.2858379118 | 1.8488123691 |
| R/J，实际联合关系／独立矩参照 | 1.571525593 | 1.065197440 |
| T/U，expert总输出能量／动态增量 | 2.254310251 | 18.52334795 |
| J/I，expert几何trace收缩 | 0.08068349205 | 0.09370078714 |
| J/(UV)，独立参照除去上述尺度 | 0.1818856232 | 1.735652283 |

V是二阶变差而非标准差。T/U受expert数量、共同分量和混合尺度影响，不是“冗余量”；J/I同时含expert contrast与方向，不是纯方向对齐。R/J>1只否定总体及原分域记录中的负input联合耦合解释，不能排除低敏感方向或个别层的相反现象。PAS双专家精确simplex变化仅一维；不能把CL的多维方向问题直接套给它。

## 算法与证据

每QA先对答案预测位置含EOS平均，再对80QA等权。每层C是gate相对全80均值的偏差二阶矩，H是expert输出Gram。逐层乘后相加：I=Σ tr(C_l)tr(H_l)，T=Σ tr(H_l)，J=Σ tr(C_lH_l)，R=Σ原Rjoint_raw，U=Σ原routed增量能量。

R/U = (R/J) × (I/T) × (T/U) × (J/I)。两模型相应因子CL/PAS为0.6778111、43.6506770、8.2168583、1.1613378。不是层中位数相乘。

只读原READOUT SHA55e95fd9dde9a76b34aa02bdd12b507e03506feb2137a706ba31067c37da6ef1；没有重跑原消费者、模型、QA、生成或评分。新只读源49d7b47b...先独审，原草稿归一敏感度漏联合项在首执行前修正，草稿与修改原因保留。唯一result_v1已EXIT0/COMPLETE，矩阵trace与stored J逐层闭合，0非正gate trace层、0超消减尺度负R层，乘积闭合误差最多1.36e-20。raw与原centered R总量差PAS8.39e-16、CL2.22e-16，未归一化gate、投影C或clamp负值。

保存每层原值和总和：[完整结果](result_v1/RESULT.json)、[完成证明](result_v1/COMPLETE.json)、[独立结果审阅](RESULT_REVIEW_OBSERVATION_20261030.json)。主源analysis/conditional_function_factorization_20261030.py；原C/H统计的输入/模型/权重和QA来源不变。

## 从性质走向成因的边界

PAS源码code/PASs_MoE/source/CoIN/peft/tuners/coinmoelora.py（SHA75d46bcd...）的router a使用S_e=mean_j[(A_e x)_j²]，rank128，原adapter每stage tau重置1.5，无x/A范数归一。两个expert有效时一般公式为g1=sigmoid((||A1x||²−||A2x||²)/(128·tau))。stage起点tau=1.5不能冒作S19实际值；原观测直接重载完整method/model对象并保存Python属性，未见重建默认tau的路径，现存READOUT没有记录实际tau数值。这是明确源码事实，**尚无实际能量差及其尺度证据，不能由rank均值或温度直接断言低变化的训练原因**。权重/专家重叠、路由幅度和全网敏感度仍需分开。

[MC-SMoE](https://proceedings.iclr.cc/paper_files/paper/2024/file/3d09a88c3372cdb79401fde592ca4db8-Paper-Conference.pdf)已分开研究expert使用频率、相似度和合并损失；[PASs-MoE](https://arxiv.org/html/2601.13020v2)的跨checkpoint路由稳定不是本末态跨input低变差；[SMoLoRA](https://arxiv.org/html/2411.13949)的expert收益饱和也不直接证明本功能残差小。已有先例不否定本场景观测价值，但这些性质不能混称。

## 适用范围与当前主线

目前只支持1%两个原末态模型；10%同类功能数据仍缺，不能由10%主表质量代替。非扩展方法用于原质量/参数/内存对照，不能把其router变化定义零计作实测普适证据。先查性质、成因和反例，不设计新方法。PaRSP保护预算/geometry/CE线已取消，未启动任何模型，基线pause已撤回继续；100%仅保留optional准备。

PAS-static另有自己的完整21454TEST，F1比原PAS低0.288pp、原固定五窗GPU低24.84–26.81%，但使用另一能量加权常数c55，不是本平均gate参照；本分解不继承或新增F1/质量结论。
