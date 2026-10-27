# 两个10% SeqFT最终模型：资源实测

两组原S19完整未fold图均已正常退出；同物理GPU1（UUID `GPU-3c72e4ce-0896-0b40-d610-be585d2bb51e`）串行测量，原verifier各调用一次，均通过。共10个固定生成窗、320新token，0训练、0F1评分。原正式质量分数不变。

| 骨干 | 五窗最大allocated峰值 MiB | 最大reserved MiB | 中位TTFT 秒 | 中位解码速度 token/s | 常驻adapter bank |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-OneVision | 2176.659 | 2482.000 | 0.315051 | 31.542991 | 1 |
| Qwen2-VL | 4954.604 | 5312.000 | 0.515078 | 24.872793 | 1 |

完整参数/权重字节分别为894216224 / 1789513792、2210075136 / 4422329344；均保原FP16基座和FP32 r8适配器。48/56个q/v投影模块分别属于一个共享bank，不能计成48/56个bank。原CP与配置均按已sealed正式proof绑定，loader核完整CP SHA，测前后参数、buffer及公开generation配置保持。

每组独立fresh reload后顺序测五个原hash-min raw QA，每题固定32新token；原图/输入摘要与旧窗口逐项匹配。峰值含输入准备、视觉和decoder，属于PyTorch allocated/reserved口径。TTFT从prepare前至首token；解码速度为31/(末token时间−首token时间)，含有限logit检查和逐token同步。因此是受控instrumented时延，不代表生产吞吐、全TEST峰或每题独立冷启动。跨骨干各用原processor，raw QA相同不意味着prepared tensor相同；五个资源输入不是新heldout质量评价。

两worker耗时34.815/69.021秒，其中原CP校验及加载8.693/19.137秒；两phase含启动共36.192/70.373秒，supervisor总106.603秒。环境修正只改用`.vision-env`；四个绑定包版本及15个实际库文件SHA与原绑定一致，Pillow12.3.0可用，原worker/CP/五窗/期限不改。

旧`.conda-env`尝试在CUDA初始化后、模型加载前因缺PIL失败，原OneVision phase耗时2.735秒、Qwen未启动；原failed_closed记录与费用保留。新输出独立，未清空旧slot。核验后取回证据时，采集器先后发生`sha(str)`及误索不存在PHASE_EXIT两项元数据错误：前者在verifier调用前，后者在两次成功核验后；均不涉及模型或评分。后续只复制已有VERIFIED，没有再次调用verifier。

原字节小证据位于[final_resource_verified_20261027](final_resource_verified_20261027/)，包含两VERIFIED、各STARTED/COMPLETE/EXIT/windows/binding及supervisor/slot/request/环境和取回收据，不含QA正文。VERIFIED SHA：OneVision `36dcffab039c5d47d18eadc1fb31fd43cec397a890e5cd03d717614d510ce2ac`；Qwen `a0ceea709732b26c6fc31d3f3bb6c4559ff20b3da9b27d3e378d56ac65d3a579`。核验器SHA `8618937d50490e63cb559adcfc53920cabae800c9600ca0d04eae5438243f5f7`；取回另逐文件SHA验证，0新评分。
