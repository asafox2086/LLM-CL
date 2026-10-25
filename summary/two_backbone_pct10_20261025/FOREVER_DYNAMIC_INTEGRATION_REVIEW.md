# FOREVER动态适配：独立集成源码审阅

**当前源码未发现新的明确fatal，可以进行NEW上的metadata-only build；不是模型准入，也不是GPU／checkpoint重载已通过。** 本次只读源码、AST parse／compile与字节SHA；没有导入执行模型模块、调用builder／预检／评分、读取原数据或权重、运输或连接服务器。

## 必须披露的初审遗漏

我此前审阅动态core及adapter后给出“未见fatal”，但没有读完原run的完整preflight控制流，漏掉了致命的 `self.core is None` 前提。原run先 `method.begin_stage(1)`，再对19个task各首题做一次诊断更新；该循环没有调用FOREVER的 `after_update`／`end_stage`，所以进入 `preflight_replay()` 时S1 core仍存在，虽然其drift step／tau仍为0。旧新版adapter当时要求coreNone，会拒绝实际预检。root在集成审阅中识别了这一点，作者在任何GPU执行前修源。不能称初审已覆盖完整集成，也不能把此次修正说成GPU故障已复现或GPU验证通过。

## 本次实际绑定的文件

| 文件 | SHA256 |
| --- | --- |
| 新 core.py | `7dcbadf1a7793cf32b13cbf08e2697a4d5588d5062841c5055eb82ed0d70202b` |
| 新 forever_adapter.py | `68a911e46c1e6db9d8da8125e9243f6480032e98bbbdc1f8cd9e49db9c5327cb` |
| 新 build_runner.py | `b503c2e9f8055c173db4566a2d20f9524f8f36f3534cd1505bf381b228901172` |
| IMPLEMENTATION_NOTES.md | `93b70eab7f80d6993b64a3ac957467f96b88e2a29f9ed831359a886c6d53c54f` |
| 原v3 run.py | `ebc68f61f35d7a54c2021f332769f50cc763cd60dc98051f663d676df2a11116` |
| 原 checkpoints.py | `06c53eab2c6810901f5206edbfb1df9defabdc747b1856ace0afa31e74b37a78` |
| 原 checkpoint_probe.py | `71b38fb9f202fa2fa012e76a883e23322f68ef938c9f8ef9fac78d01886eb23f` |
| 原 analysis/forever_multimodal_core_20261019.py | `2d9d5be87d81df80eab6ddf0095ce399c9a73d9c8c29909a5022d2364fdeeed8` |
| 原 analysis/forever_multimodal_replay_v2_20261019.py | `8024931496a032ea666e44376d79614d7c223b17c73ff11a8074494f87cd3683` |

前三个新文件位于 `exp/two_backbone_pct10_forever_dynamic_20261025/`；原v3文件位于 `exp/two_backbone_forever_20261019_v3/`。本次不修改它们。

## 完整调用链与一次派生检查

1. 新函数只在preflight且遗留core严格为stage1、step／tau均0、未校准、无历史／活跃或完成回放／终态、无stage_summaries时丢弃其调度器。原19次诊断造成的权重变化保留；receipt明确其不是已学S1前缀，19案例的实际证据仍须绑定原外层preflight记录。不会假造19次drift更新。
2. 唯一派生view保留19task结构，T1–12按原task哈希序各首1，T13前385，未来task为空。只有新core接收view，`self.train_tasks`及完整ID→task loss路由不变。core从385题得到25个真实current更新；旧selector确实返回 `selection['history_QA']`，此处要求等于12且memory为实际过去task选中样本，字段没有误用。
3. 25步采用原optimizer／cosine／GradScaler和batch16／micro1，warmup为0。第24步真实校准后自然检查阈值并回放，随后真实执行第25步，再执行final；没有先跑S2七步或改tau强行触发。若自然触发没有实际发生，断言拒绝通过，不追加输入救跑。日志顺序检查明确dynamic@24完成→current25→final。
4. `observed_dynamic`只作为此次预检的实例临时hook。它保存原bound method并调用原 `_execute_replay`，观察外层optimizer／scheduler／scaler对象及状态、参数对象／storage／requires_grad和gradient-checkpoint flags；允许权重与全局随机流改变，不恢复状态来制造相等。hook在finally删除，随后final使用类原方法。
5. Adam的0维step tensor只在观察器的hash输入上 `reshape(-1)`，原shape另写入hash键；原optimizer state未被改写。原raw-byte digest读1维视图，避免直接对0维做不同元素大小的uint8 view。tensor state bytes、对象／data_ptr、非tensor state、group options均参与前后比较。这里是可执行的检查合同，不是已经观察到外层状态不变。
6. 原 `execute_replay`在实际HF回放后调用一次core.replay_complete；新core只更新漂移参考并推进阈值／final状态，不推进current step／tau／mu。adapter finally恢复当前instruction、train模式并清梯度。第25步结束后释放外部optimizer，`end_stage(13)`执行final、保存summary并将core设None；实例临时hook已删除，完整task lists／mapping有末尾不变断言。

## 原T19尾部及可重载模块路径

我已实际读原run完整preflight分支：上述函数返回后仍执行原固定 **T19首TRAIN probe**，随后保存带preflight probe的完整method graph及 `gpu_preflight.json`。不是T13尾probe。独立checkpoint_probe再以原保存的task／row完成loss容差与有限backward验证。preflight的S1 checkpoint编号只是原诊断保存接口，不是完成正式S1的声明；配置preflight_only及checkpoint_phase仍区分诊断。

新receipt明确尾部probe／CP／reload尚须外层proof闭合。19公共诊断、385 current、实际各次两epoch replay、每次两个unwrap TRAIN NLL以及原尾probe／保存／reload均是费用，不能仅报25步或把重叠计时相加。

模块加载为正常 `forever_adapter` 方法类导入；它按同目录core.py加载并在sys.modules注册 `pct10_forever_dynamic_core`。阶段结束checkpoint前core已释放，保存方法只增加字典摘要／回放报告，不保留活跃ReplayTrainer或临时observer closure。builder将core.py、adapter及全部runner源纳入 `*.py` SHA集合，并要求原两helper在source／overlay字节一致；methods.ROOT深度保持。源码没有新的必然模块路径失败，但实际完整序列化／fresh-process reload仍未发生，不能提前宣称通过。

## 构建边界与方法口径

builder只读metadata／源并编译AST，拒绝覆盖已拥有的core／adapter及既有复制目标；支持原v3或冻结pct10来源，只对run的condition／train_fraction／primary_score入口作可逆metadata替换，其余复制不改。新policy独立、记录科学源diff与原helper SHA，不构建／启动旧目录或改现有12项native工作。可以执行这一步metadata-only build，不能将其输出当模型准入或效果。

正式动态core未因预检修正而改写。保留共同task级主optimizer／cosine／原current顺序是明确的多模态适配；作者分段HF重建、外层旧moment在回放后权重上继续及HF重置／消耗全局RNG的差别已披露，不要求字节复现才可用，也不称执行数学完全相同或同总计算。当前接受的是修正后的源码集成合同；GPU中途衔接、完整CP／reload、资源与最终质量仍全部待实际证据。
