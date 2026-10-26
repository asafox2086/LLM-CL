# 独立计时与峰值范围审阅

2026-10-26，research_review审阅worker `aedc7e7b…`、FixedWindowTokenClock及继承的readouts._generate/OneVisionBridge.generate。未发现阻止源码发布的问题；仅源码结论，不代表GPU执行通过。

- 计时从input_batch前开始，首token前包含视觉准备。
- `31/(t32-t1)`是后续31个token的同步墙钟解码速度，不是32token总吞吐。
- 每token有限值检查、streamer CPU交付及显式GPU同步均在计时中。必须保留instrumented / not production解释。
- GPU峰在输入准备前重置，完整生成返回后读取；审计张量摘要在峰值捕获之后，符合本窗口范围。
- nvidia-smi运输解析失败实际来自末尾空行，worker原strip已处理，不应改写成warning或模型故障。

主表缺测仍缺测。当前两骨干仅metadata验证通过；没有执行GPU测量，没有新增F1或方法成功结论。
