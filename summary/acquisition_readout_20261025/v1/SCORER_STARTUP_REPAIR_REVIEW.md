# 一次核验入口修复：源码裁决

接受限定的元数据修复；优先本新RUN内归档，不建立额外symlink视图。此审阅不执行修复、模型、验证器或评分，不读取预测正文。

原验证器SHA为`f5951d82919691dcf4ad195529e91d075f8702af418642920c0a69b57ca45f2f`。第16行`os.environ.get`调用前必先求值default，即使`LLMCL_ROOT`已经存在，短`/tmp/.../verify.py`路径的`parents[3]`仍会越界。现有stderr表明错误发生在模块导入，早于main、VERIFY_STARTED及scorer调用，因此该失败没有评分；实施前仍应实际确认两个评分标记均不存在。

这是本验证器作者及源码交叉审阅漏检的真实入口错误。模型80次生成与两S19检查已完整成功，故此错误不构成回答质量阴性。不能因这次修复补训练、补生成或改变科学条件。

本地原失败文件SHA：VERIFIER_EXIT.json为`22b5dec7b4f9e1242aa60599b67d8084f32573955f555d3e4fa881e7f5b8d30d`；CONTROLLER_FAILURE.json为`3688ae3ba29fb41b9aebac567fe7ca7baa98fa35e40ea3f0863abe7c7a15c4c7`。

最小修复范围如下：

- 原3个科学源码、READY、所有模型输出与父ARC／Naive RUN保持原字节。仅将本新RUN的CONTROLLER_FAILURE.json和VERIFIER_EXIT.json按原SHA归档至`post_model_startup_failure_20261025/`；明确记录原路径名到归档路径的对应，不能称原顶层文件仍存在或原服务成功。
- 独立外部wrapper在内存AST中唯一匹配ROOT赋值，替换为`ROOT = Path(os.environ['LLMCL_ROOT'])`。逆变换须重建原AST全部节点，原文件及其SHA不变。RUN／ASSET、全部来源、80输入、引用、帽、重构状态、评分器及统计逻辑均不修改。
- 先核原失败两文件SHA、原unit保持failed／exit1、模型worker已退出且父证据链与新科学条件未变、VERIFY_STARTED／VERIFIED不存在；再执行首次且唯一80条新评分。旧末态分数继续只复制，不重新评分。
- 评分成功后由独立修复收据绑定原失败、原0score、新80score一次、原源码／READY、唯一AST差异及新VERIFIED。原unit仍failed；不得伪称原controller完整成功。
- 原activation起2100秒绝对截止不延长。失败核验、审阅等待与修复核验费用全部保留，累计墙钟与各区间重叠说明，不新增模型预算或另起计时。

归档优于视图，因为它不改变RUN、输出位置或来源路径语义；本新RUN自己的失败元数据归档与父科学来源只读要求不冲突。旧失败原字节与路径映射透明保留是必要条件，不能静默删除或通过改glob掩盖。

本意见只接受这一启动修复，不保证科研结果，不授权任何模型重试。当前审阅成本为源码与小故障proof检查，0模型／0生成／0评分。
