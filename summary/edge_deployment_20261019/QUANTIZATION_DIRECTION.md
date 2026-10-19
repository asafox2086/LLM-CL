> 既有CPU三精度对照及独立重载已全部自然结束：[工程封存](cpu_precision/v2/closeout/VERIFIED_RESULTS.md)。没有稳定恢复五个TRAIN请求的质量，未形成可采用方法；不再沿局部量化追加实验。

> 最新用户纠正优先：[研究问题更正](RESEARCH_RESET_20261019.md)。下文是历史工程对照及局部证据，不能作为整个领域缺陷或当前已定方法路线；停止扩展未准入的激活粒度试验，视觉骨干替换亦未准入。

# 当前方向：保住小模型已有功能，再压缩真实推理负担

## 我们实际发现了什么

| 问题 | 实验与结果 | 结论 |
|---|---|---|
| 压缩语言模型就足够轻吗？ | 实际读取两骨干权重header。OneVision视觉编码器占tensor字节44.51%，Qwen2-VL占28.57%。只将语言权重16→8bit，理想总tensor字节仅少27.64%／34.94%，尚未计scale等。 | 小语言基座仍背着较大的视觉部分。只减少LoRA／语言参数，不能直接等同整个图文模型轻量。比例是存储记账，不是实测速率。 |
| 全部INT8就能安全部署吗？ | 同五个固定TRAIN请求已完成FP32先行参考与全INT8。自然／遥感的单题F1从100降0，医学仍100；交通23.229降0。已按全部原参考独立复算。 | 不能直接全部压低精度并沿用原F1。尚未定位是视觉、投影、输出头或其组合，也不是完整测试退化结论。 |
| 是否存在可以直接删的尾层／头？ | 实际OneVision配置选视觉最后一层（-1），不是常见CLIP倒数第二层；池化头已禁用。 | 不能照搬其他架构，声称最后一层和池化头是“无用结构”。全hidden_states收集只取最后一层的开销可另测，目前没有数值／内存收益证明。 |

[两骨干结构证据](component_footprint/v1/RESEARCH_FINDINGS.md) · [五个TRAIN先行质量对照](cpu_precision/v2/initial_quality_observation/VERIFIED_RESULTS.md)

## 可借鉴的已发表方法

[MBQ，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Li_MBQ_Modality-Balanced_Quantization_for_Large_Vision-Language_Models_CVPR_2025_paper.html)指出，不同模态token的量化敏感度不同，用SFT梯度加权校准重建目标，而非统一对待。论文覆盖7B–70B模型，并结合实际低比特内核；没有证明我们的0.5B／1%场景有效。[作者正文](https://arxiv.org/html/2412.19509v1) · [作者源码](https://github.com/thu-nics/MBQ)

作者参考已固定revision `a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1`，只存放服务器`code/MBQ/source`作参考，主Git仅收metadata／MIT许可信息。不安装其依赖，不替换活动环境；这不是方法复现或GPU试验完成。

需要区分：**模态token敏感度不等于模块敏感度**。当前全INT8同时改变解码器、视觉、投影和lm_head；不能凭自然题出错就认定视觉模块该保留高精度。先完成已定decoder INT8与fresh FP32，对比原功能损失和实际资源，再决定是否借鉴敏感度校准或保护特定部分。

## 下一实际动作与边界

1. CPU三条件全部完成；新独立进程只从真正部署包恢复，严格核对state和五题8token。不能把同进程保存检查写成部署成功。
2. 新GPU1空闲即按既有四条件执行，测试视觉冗余／token精简；基线和旧研究任务保留。
3. 若实施MBQ式校准，只允许冻结1313TRAIN，不引入作者COCO／Pile等外部校准数据；新增前后向和搜索成本计入。选择不用TEST。
4. 任何最终压缩版本都须同设置21454QA确认和实际端侧测量。所有方法获得相同常规导出／无损存储优化机会，不能把FP32计算镜像当作不可压缩的存储基线。

**改进效果尚未成立。** 现在有的是完整五个TRAIN请求上的负例和两骨干资源结构依据，目标仍是最终质量略高于最强基线、端侧成本显著改善。


## 已启动的最小改动

新增固定per-output-channel INT8控制：位宽仍8、同327个线性层与同五TRAIN，仅权重scale粒度不同；不是新LoRA／新训练或MBQ复现。现有CPU内核和严格state恢复的小实际Linear门槛通过，完整模型试验已真实启动，完整五题与重载结果待。原全部三条件保留，不能预设逐通道必恢复质量。

[独立协议](cpu_precision/per_channel/v1/PROTOCOL.md) · [实际启动记录](cpu_precision/per_channel/v1/LAUNCH.json)


## 逐通道控制已完整，下一步换检验对象

[逐通道完整五TRAIN结果](cpu_precision/per_channel/v1/VERIFIED_RESULTS.md)没有恢复原高分。仅改权重粒度尚不足够。[实际CPU激活范围控制](activation_granularity_20261019/v1/RESEARCH_FINDINGS.md)表明，两种权重方案仍共享整输入的激活范围；一个大行可使不变的小行输出归零。

这是可检验的下一方向，不是已证明的QA唯一原因。接下来应在相同整数核下比较激活粒度，或借鉴已发表的敏感度校准；不要继续盲目增LoRA、删视觉尾层或仅以更小权重MSE宣称质量改善。完整模型对照尚未实施，保留所有负结果和完整评测要求。
