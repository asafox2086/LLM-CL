# 既存功能矩的描述性分解，首消费前冻结

只读已核验READOUT55e95fd9...，不重新跑原消费者、模型、QA、生成、评分或TEST。仅all组：每QA答案位置平均、80历史TRAIN等权、168投影直接相加。原C/H/raw与centered差保留，不归一化gate、不投影C、不clamp、不设epsilon、不把层中位数相乘。

逐层C为gate偏差二阶矩、H为实际expert输出Gram；I=sum_l tr(C_l)tr(H_l)，T=sum_l tr(H_l)，J=sum_l tr(C_l H_l)，R=sum_l Rjoint_raw，U=sum_l routed增量能量。精确恒等式R/U=(R/J)(I/T)(T/U)(J/I)。核stored J与矩阵trace闭合，记录raw/centered差与原消减尺度；分母非正缺测。保存每层原数值及总和，负值不删、不替换。

要回答的是：移除gate变差尺度后，两模型局部差异对比是否仍大，不能把四因子称独立原因。T/U混有expert数量、共同分量与混合；J/I混有contrast大小和方向，不是纯冗余/纯方向。联合耦合R/J也不是方向对齐。PAS/CL隐藏轨迹与专家数不同，比例描述不是预算匹配或因果贡献率。1%同两模型，10%仍缺观测；非扩展方法的定义零不计普适样本。无新方法设计。

独立源审通过后仅执行一次，独占result_v1，不覆盖失败标记，不自动更换目录重试。当前尚未执行。
