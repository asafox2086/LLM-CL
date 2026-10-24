# 输入检查失误与准入边界

真实模型启动前的两次CPU检查失败，均0模型/真实QA/CUDA；原stderr、源码SHA和OLD FIRST_/SECOND_源字节保留。首次监督3.138212s，第二次3.718852s，未产生质量读数。

1. 首次门错误假设图像占位符属于tokenizer.all_special_ids。实际added token `<image>`=151646、`<video>`=151647，但特殊属性列表不含它们。修复为显式禁止这些占位符进入OCR body；门使用与真实worker相同AutoProcessor.tokenizer和内容列表模板。
2. 第二次门仍猜测user结束与assistant开始之间有换行。读取原冻结chat_template.json后确认两个special marker之间没有换行；只修尾部验证并核最后完成轮为user，不改原token、插入位置或科学干预。

这是主/子agent未先核完整processor合同的预检查覆盖失误，不是词面证据通路的质量正负结果或研究进展。没有改环境、模型、数据、prompt、donor、评分或预算。

修正后实际原processor/tokenizer、synthetic20组/3个token干预检查通过：内部3.220123s/监督3.749209s重叠不加，0模型/真实QA/CUDA。旧模型严格重载检查没有重做。READY绑定修正后的最终源与原runtime/资产摘要；首次真实模型故障仍按协议关闭，不以门修复作为模型重试许可。
