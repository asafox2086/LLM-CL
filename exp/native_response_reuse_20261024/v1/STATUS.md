# 未准入的局部实现

只有core.py/runtime.py，尚无完整run/common/freeze/dispatch、SOURCE_REVIEW、CPU_GATE、READY、权重下载或模型启动。导入不会执行试验。局部语法检查不是模型准入。

必要性草案见summary/final_weight_reduction_20261023/native_response_reuse_pilot_v1。MECHANISM_DISCUSSION进一步明确：原生FFN条件门不等于承接PASs的RW/RS。当前停止完整worker扩编，不自动下载/激活，不把现成小模型+选择性微调叫新框架。

旧视觉/PAS/r200/ShortFixed等均保持终态。本目录不授权恢复任何旧作业。
