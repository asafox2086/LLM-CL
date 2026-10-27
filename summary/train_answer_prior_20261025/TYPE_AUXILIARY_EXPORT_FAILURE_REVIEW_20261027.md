本次应记录为导出实现故障、三臂质量缺测，不能记录为题型辅助方法的科研阴性。依据root已报告的实际终态，共同prefix完整exit0，plain完成S16–S19的18次更新后，在`worker.export_complete`检查`config._original_object_hash == hash(config)`时失败；controller按`no_retry`关闭。完成训练更新不等于完成自包含导出、独立重载与三臂最终评价，也没有可用的三臂唯一评分结论。本审阅不消费未评分回答或推断其质量。

本地可审证据支持一个具体的运行时解释：Transformers 4.46.3的`generation/configuration_utils.py:477–478`将`GenerationConfig.__hash__`实现为`hash(self.to_json_string(ignore_metadata=True))`。普通loader及`from_model_config`在该文件1055、1059、1257行缓存`_original_object_hash`；`to_dict`在1149–1150行明确删除该缓存字段。本地库版本由源码`__version__="4.46.3"`核对，该文件SHA为`74899c9b641aec9d8b0897b804fef962098181db18f33737625ef26c5ff57921`；未导入该库或模型。

新worker通过原`checkpoints.save/load`整图pickle保存S15的method、processor和RNG。load还原对象及RNG，没有重建GenerationConfig的进程内hash缓存；各分支由controller启动为独立Python进程。这与旧conditional、paired-option各自fresh loader后在同一进程导出的路径不同。一个从旧进程保存的hash整数，不是新进程可移植的配置指纹：公开配置字符串不变，也可能不再等于新进程重新计算的hash。

已完成的本地纯stdlib验证使用Python 3.11.7、两个独立子进程，`PYTHONHASHSEED`均未设置、`sys.flags.hash_randomization=1`。同一字符串`fixed-generation-config-string`的hash分别为`4066426881929380980`和`-6497055251478050786`。该验证没有导入Torch／Transformers，没有模型、QA或评分。这证明了上述比较的跨进程可移植性缺陷；它不是OLD实际运行的config对照，也不能单凭这两个数断言本次故障的唯一根因。

尚未逐值证实本次失败对象的公开配置没有改变；仅凭堆栈也不能确定循环中的外层或语言层GenerationConfig哪一个触发断言，实际进程的hash seed条件也未由这项本地验证确认。因此不能把“缓存失配有具体解释”写成“已证明全部配置语义未变”。另外，4.46.3的`generation/utils.py:1485–1500`把此比较用于legacy兼容分支：满足条件时可能从`model.config`重新派生generation config。它不是纯日志字段，不能简单忽略断言、强行刷新缓存或改`_from_model_config`，然后未经核对就宣称行为相同。旧审阅沿用同进程成功导出经验，没有覆盖这条跨进程缓存边界。

该故障不足以否决训练期题型辅助假说，但当前固定点仍遵守`no_retry`关闭；旧源、失败、已耗训练及生成成本全部保留。人类已有的方法验证授权可涵盖不改变学习问题的必要实现修复；自设`no_retry`不应泛化为所有后续修复都必须重新取得人类批准。独立修正版需要透明记录版本及差分，不能伪装为旧run继续成功。本审阅本身不执行修复、启动新run或重置预算。

最小下一步是一次离线只读提取原sealed S15的外层／语言层公开generation配置、`_from_model_config`及对应model config的非默认generation参数，建立稳定SHA并核对HF legacy分支语义。`PREFIX_STATE`当前没有公开generation配置指纹；export内存中的receipt也未在故障前落盘，不能宣称已具备前后公开配置同值证据，更不能还原已经丢失的plain导出时对象。

若该检查确认跨进程重载及现有最终loader不会经legacy分支派生出不同的公开配置，可以仅将export的进程内Python hash比较替换为对原S15公开配置的稳定SHA不变性检查。该修正不应重写对象的私有标记或修改HF生成路径。若legacy分支可能产生不同公开配置，现有证据就不足以声称I/O等价，不能直接删guard或强行设置新hash。任何修正版还应控制以下已有设计事项：

- 三臂从同一个明确来源的完整S15状态与同一Python／NumPy／Torch CPU／CUDA RNG分支，保留模型、buffers、processor、配置和源的绑定；head初始化不能提前改变训练随机流。不得从已训练plain的后继状态启动其它臂。无论新生成还是明示复用S15，均须如实说明来源与费用。
- 在跨进程保存前、重载后以及导出时，对外层和语言层公开generation配置使用稳定的规范序列化／SHA对照，连同功能性model config、`_from_model_config`及原生成stop／cap／policy共同核对。hash缓存本身的处理需有明确语义依据，尤其确认是否改变legacy重新派生分支，不能仅使原assert变成真。
- 原数据／split、r8、λ1、监督与shuffle对应、优化／生成／评分条件保持预先约定；导出后的完整图和配置需真实严格重载。修正版若仍失败，保留缺测；不能据此追设置，也不能把配置I/O修复本身称为研究方法或效果。

本文件只记录已完成的本地源码审阅、stdlib证据及未来评判边界；未重复上述验证，0新模型、生成、评分或服务器操作，未修改冻结科学源。
