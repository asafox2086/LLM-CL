# 外部重启只中断未尝试等待入口

问题：约08:18:29UTC NEW再次外部重启，boot为c8bddf24-7969-41d6-a0fc-e5287850807f。PAS v2旧CPU13658已消失，旧queue wait与activation保留。核实无STARTED、private、训练/模型产物，原run源码在写STARTED之后才导入Torch/CUDA；源与READY不变。没有重跑CPU准入、模型或评分。

结论与行动：独立评审接受同READY仅接续原CPU等待入口，不视为第二次fresh模型重做。原4h模型预算尚未启动，不改累计6h保守边界；首次worker STARTED后再次中断/失败/超预算必须停止，不能借CPU接续绕过。原activation/state原字节私有备份，新恢复记录绑定旧SHA。新controller10505/start131721是08:40:26UTC恢复快照，后续以实际身份与PHASE/queue为准。

正式：原native协调2497/start10169及合法CL8260/start60876、SMo8265/start60882完全保留，不停协调器、不改formal state、不向GPU信号。仅恢复CPU七范围publisher10495/start131459与原无TRAIN的FOREVER等待10498/start131514，旧owner归档。所有GPU进程恢复前后身份相等，作者/正式/研究科学源不改。编号均为时点快照，不永久live。

效果：恢复等待及只读发布能力，没有新Answer F1/方法收益；已中断v1费用与负结果保留，Ours/正式表不变。端侧仅科研背景，不新增工程/科学实验。重复主机重启是外部可用性问题，不作为科研机制结论。
