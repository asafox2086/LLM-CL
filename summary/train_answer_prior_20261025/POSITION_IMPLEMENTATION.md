# 位置诊断的实际准备与尚未执行项

[worker](../../exp/finance_position_diagnostic_20261026/worker.py) SHA `f23db5f717f49efbd047de4a986c528cdc60a1c925a1577350a0b43f56c9dd2b`，只恢复原完整图及执行90生成；不训练、不fold、不调用evaluate／评分、不写原latest或checkpoint。CLI为`--condition pct01/pct10 --stage 15/16 --binding /absolute/binding.json --output /absolute/new_private_directory`，`LLMCL_ROOT`为相应原数据／模型根，`CUDA_VISIBLE_DEVICES`须指定一张已协调的空卡。四态顺序、每态1200秒及完整结果一次消费按[协议](POSITION_DIAGNOSTIC_PROTOCOL.md)，不能直接无监督运行或忽略空卡分配竞态。

已完成的真实输入准备：

- [模型来源清单](POSITION_SOURCE_INVENTORY.json)：四个实际完整CP及其meta大小一致，仅stat／小JSON，未读权重载荷。正式proof封的是stage marker/config/训练源及最终CP，不能说早先已封中间CP载荷；首次实际加载将各自完整SHA一次并检查payload。
- [图像绑定](POSITION_IMAGE_BINDING.json)：两原TRAIN文件的45个既定ID全对象一致，45张实际图片SHA已读取，两根指向相同物理文件。读取原89／887行含gold字节，但只按既定ID核输入／图像，不以gold选题、改题或评价。0.041012秒，0模型／生成／评分／CP读取。
- [变换绑定](POSITION_TRANSFORM_BINDING.json)：在已有去答案资格包上实际构造固定45个循环视图，全部严格逆变换还原原context；仅发布哈希／框架，没有原文。0模型／生成／评分。两视图长度均不触发原6000检索，范围未变。
- [完整调用绑定](POSITION_DIAGNOSTIC_BINDING.json)：明确worker、协议、资格／原输入收据、去答案包、两proof、两个数据根、四个state来源和45图SHA。它只是来源／输入准备，**不是GPU准入或已自动排队**。现仍缺实际协调空卡、首次预算监督、四态完整结果及唯一消费者，不能把代码检查称功能成立。

root完整源码核对、baseline_review一次静态交叉核对原methods/run/runtime/checkpoints/bridge/frozen_vision与两份小proof后，未发现致命接口问题；research_review接受单循环功能用途，均没有模型执行或重评分。作者plain_r8_control只写worker及stdlib AST检查。初版无依据要求context等于metadata.options，已在任何执行前删除：原input_batch只用context，metadata不构成干预接口。root审阅时还漏看run中紧接着设置method.name/revision的两行，错误建议其应为SeqLoRA／None；作者复核指出后root撤回，保留原正确的SeqFT_LoRA／standard revision断言，没有据错误审阅改动模型或增加identity默认值。

全部参数和registered buffers将前后精确核验，普通视觉缓存可按原forward变化；实际资源采样只描述本诊断全流程，不是固定五窗／最坏推理峰／速度优势。诊断的A–D解析允许去首尾空白，与前一消费者的原始字符串严格计数分开，不混称同一个统计。没有heldout／新F1、PIF训练或新方法成功，原两个完整主表与全部最终模型留存继续。
