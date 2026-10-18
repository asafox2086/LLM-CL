# 七项已准入基线的只读结果发布器

用户进一步移出标准ER-LoRA，当前7项已准入方法×2骨干＝14组；FOREVER是第8项计划比较，待图文适配与GPU准入。ER已有完整结果与未完成运行全部保留为历史，不填当前主表。

入口`analysis/v2_seven_baseline_report.py`复用原始d7aac920逐题核验器，训练Python/数据/评分均不改；原八项发布器计划退役。已完成证明按字节保留；只接受19阶段、74格完整评测、21454最终QA、正LR/有限梯度/参数变化及最终checkpoint。原ER完整证明仍在历史八项/十项目录。

最外层README主表使用`analysis/v2_readme_main_table.py --write`从本目录封存索引与逐份证明生成，只填核验结果，未完成留空。无训练/推理/调参；FOREVER/Ours不会由其他协议研究诊断填补。

实际取消Qwen ER进程、其余GPU进程保全与两机策略同步见[取消记录](../er_removal_20261018/ACTIVATION.json)。
