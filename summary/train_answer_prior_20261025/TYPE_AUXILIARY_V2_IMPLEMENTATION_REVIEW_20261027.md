接受本轮v2的有限实施差分；这不是GPU运行通过或题型辅助有效性结论。审阅范围为worker与v1的实际差分、生成配置合同、helpers／training逐字节一致性，以及queue的同卡接入。未修改任何科学源、controller或verifier，未读取新结果或执行模型。

`generation_contract`使用已封S15配置证据的公开字典及同编码稳定SHA，要求两层`_from_model_config=True`、对应model config非默认generation参数均为空。在S15加载后、最终导出及完整包重载后调用。它没有刷新原对象hash缓存、改私有标记或改HF生成代码；原最终loader仍沿用既有实现。已实查S15的两层公开／派生配置相同且非默认参数为空，为这项修正提供了具体依据，不是直接删除配置检查。

helpers和training与v1原字节相同；原数据、标签及shuffle对应、λ1、answer loss、训练AST、q/v-r8、优化、fold和生成操作未改变。旧prefix来源不再冒称与v2 worker源相同，而由`controller.validate_reused_prefix`明确核原sealed来源、成功退出及完整CP，再从原S15完整图与RNG加载。worker使用其返回的来源记录，保留实际模型／buffer摘要检查。controller与后置verifier的完整实现由对应作者另审，本记录不把接口核对扩成端到端证明。

审阅发现并已修复一个实际复用风险：原整图loader保留绝对CUDA设备并恢复原设备的RNG；新slot若任意选卡，可能加载到未获配卡且不承接原训练流。当前worker加载前要求`GPU == reused_prefix.GPU == 1`，queue只在该卡通过原idle链时claim；其它卡保持原available结果。原来源验证同时核prefix实际STARTED／COMPLETE的GPU。这是保护原设备／RNG合同的来源修正，不改学习数学。

最终queue接在原corrected-resource外层之后：先调用原`corrected_class`并运行其wrap，再追加v2 hook，使用独立`_type_aux_v2_slot_wrapped`标记、`TYPE_AUXILIARY_V2_SLOT.json`及`run_v2`，不复用旧失败slot或被v1标记跳过。drain先等待原资源槽返回、释放其锁，再进入v2的同一`dispatch_old.lock`收尾；没有递归创建分配器或嵌套持锁。原参数继续转发，阶段child仍走原入口；只有原资源released且GPU1原idle通过后才claim。

已执行的检查仅使用stdlib、AST提取的`generation_contract`及假配置对象：实际S15公开字典两层均被接受；单字段`max_new_tokens`变化、非空nondefault generation参数、`_from_model_config=False`均被拒绝。假对象的`hash()`和`_original_object_hash`访问被设置为抛错，合法case仍通过，说明新函数不依赖这两个进程内值。无Torch／Transformers／模型导入、forward、评分或服务器操作。

v2队列为三个train／eval共六个新phase，不重复prefix训练。原prefix的120生成、69更新、959 FIT及其时间按历史复用记录；旧失败plain的实际训练／读数费用也须保留，不能并入v2成功臂或忽略。旧run保持关闭；本次修正不将其改写为成功。

本次所审源码SHA：

| 文件 | SHA256 |
| --- | --- |
| `exp/type_auxiliary_supervision_20261027/v2/worker.py` | `4a040b66e995b4fd3e4526c47c6ba85fa1589f80df912d6eb8e5ed6ad747a4cd` |
| `exp/type_auxiliary_supervision_20261027/v2/helpers.py` | `b141ef72c7adf9e1c5adc93eab174fe4577d82c8bca2a1759308533f2c7e97f5` |
| `exp/type_auxiliary_supervision_20261027/v2/training.py` | `e9f1ab7fb4df06c17aa0ba1a2a3db7ba211d200dbe6490c18d01761a22f6e297` |
| `exp/type_auxiliary_supervision_20261027/v2/queue_adapter.py` | `6bc952b98c65980812e14da0d124b43264cb2034832d5e900e262627ba0ffa47` |

配置证据文件SHA：`4feb773ed7df6cecaf6365b732e27dbe97e36b27c99de9bcb9f13e23a54ed7e9`；两层公开配置SHA：`b44055c7c9efed87ef181b146c0f07fb4f954aa66dce382e1180259785212de6`。本审阅者仅作本轮只读审查与纯配置测试，未编写这些v2科学源。
