# 双锚结果自动报告

CPU服务llmcl-onevision-matched-report.service以nohup观察完整case证明；不启GPU、不改训练协调器。唯一acc为Answer F1（原token_f1），不进行选项映射。新增完整case后核验全部artifact SHA、固定IDs/原答案/原F1、共同stage1实际A/B和生成/NLL精确一致，再汇总旧锚质量变化、当前任务学习、同域/跨域配对差值以及loss/LR/梯度/训练内存前后与峰值。额外回放、Fisher、参考梯度成本单列。

[VERIFIED_RESULTS](VERIFIED_RESULTS.md)初始0/48表示尚无实际GPU轨迹，不是效果为0。CPU验证读取真实旧72文件，并成功拒绝23组已知首阶段不一致；输入缺失/重排/金标或分数改变/参数改变均拒绝。CPU接口验证不等于完整新case已通过实际GPU门禁。异常写observer_needs_attention并停止观察器，不停止GPU、不盲重跑。原48GPU轨迹仍按既定规则等待cap1/cap3完整确认。
