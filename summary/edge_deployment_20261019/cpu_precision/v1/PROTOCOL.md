# CPU精度与量化：避免把半精度内核成本误当方法开销

GPU1他人作业等待期间的补充部署控制，不占GPU、不训练或改正式基线。原始质量起点仍为冻结1% OSFT最终模型。基于旧机已实际导出并独立重载的静态CPU包；CPU重建与原GPU算术等价未证，不沿用原完整F1。

三个预定条件全部执行：FP32稠密；仅语言解码器线性层动态INT8；全部原nn.Linear动态INT8（包括视觉、投影和lm_head）。使用现有PyTorch2.6.0的fbgemm后端，不升级依赖、不引入GPU库。卷积、嵌入、归一化保留FP32。Embedding与lm_head是否原有共享需按真实序列化、常驻和打包缓冲计入，不以量化层数量估算总容量。

这是[PyTorch标准动态量化](https://docs.pytorch.org/docs/2.14/generated/torch.ao.quantization.quantize_dynamic.html)，不是新持续学习算法。文档当前版本与实际2.6不同，执行依据冻结环境中实际API与真实小线性层CPU运行证明。所有基线未来应获得同等常规量化机会，不能只量化我们的方法制造优势。

五个固定TRAIN图文请求与原CPU／GPU探针相同、原文字／图片／指令／公共处理不变，无token剪枝。每条件独立新进程，2CPU线程。每请求显式8token预热一次、三次强制8token速度、一次最多64token自然停止TRAIN生成；唯一既有Answer F1按原runtime函数计算，不翻译／抽答案。无TEST输入／答案／分数、无校准选择、无训练／优化器、无结果驱动删条件。

记录真实TTFT（预处理／编码／prefill全含）、后7token速度、加载时间、同PID CPU RSS/PSS/USS前后与10ms采样推理峰值。部署统一使用PyTorch state_dict容器（包含真实packed INT8权重）、配置／processor／运行规范，实际文件清单／字节／SHA全部记录。它不是HF通用可直接加载包，也不保证手机后端支持；运行时依赖大小另列。文件精确反序列化先比对每个张量／packed tuple；独立新进程构建并生成复现仍需后续验证，不能把同进程state检查写成完整部署重载。

仅五TRAIN可行性，不代表完整21454测试质量，不填正式主表或Ours。新FP32控制用于解释FP16旧CPU内核差异，不直接以先前单轮FP16与本次预热三轮的比值宣称算法加速。任何异常保存，READY冻结后修复新版本，不盲重试。
