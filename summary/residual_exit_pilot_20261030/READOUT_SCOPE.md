# 固定残差出口pilot只读核查范围

只在原CONTROLLER_V1 COMPLETE、SLOT released且绑定/terminal SHA闭合、原controller及两teacher worker真实gone、两phase EXIT0后读取完整方法结果。否则WAITING；原controller失败则FAILED。不开新模型、不读partial rows、QA正文或CP载荷，不重生成/CE/F1，不改三个科学源、数据、README或模型设置。

核binding/source/protocol及既有RESULT/READOUT/FIT/temporary-resource哈希；20个事前固定TRAIN读出ID各有A/B/C/D共80行，五域各4QA，forward/zero-update等counts符合原run合同。复核已有RESULT的等QA宏/域均值与原标量行闭合。输出B→C、C→D、A→D同20题raw block MSE和完整词表KL的右减左配对均值、胜负平、五域。lower is better；数值精确相等才记tie，不代表统计等价。

只使用事前范围中的比较：D同时在block均值和最终KL均值优于A/C才支持这个固定点出口对齐；只局部改善却不传到KL不支持扩展；C胜B/A而D无进一步证据只支持prefix配对。没有新effect-size成功阈值，也没有多seed显著性、F1/语义或全24块结论；不自动启动下一模型。两法保留各自完整正负结果，不用一个成功覆盖另一个失败。C/D的拟合目标不同，fitloss只各自诊断，不能跨臂排名。

仅复用input hashes和consumer hash均一致的缓存。单一READOUT_STARTED防止中断后自动再消费；不创建新许可、daemon或流程层。当前只准备源及scope，未执行实际消费。临时teacher+student+solve GPU峰不属于部署推理峰。
