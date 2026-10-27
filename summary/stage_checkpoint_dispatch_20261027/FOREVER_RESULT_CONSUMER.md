# 保存版FOREVER单run完整核验

`analysis/stage_checkpoint_forever_complete_verify_20261027.py`覆盖新1% Qwen `two_backbone_pct01_forever_checkpoint_20261027`及新10%两骨干 `two_backbone_pct10_forever_checkpoint_20261027`，不接入observer或分配器，不改旧proof、主表、runner或预算。源码SHA：`03df0919593f86c364e1ad67d85128aadcd1f2c1a0c9ca2bf24933ba3af284bb`。

CLI为`--root ABS --run ROOT_REL --host old2080|new3090 --binding ABS --output ABS [--write]`。默认只读来源、终态、回放及保存元数据，不进入评分循环。显式`--write`才独占登记并执行原80316条raw F1核验：19阶段、74格、21454最终回答。1%保持93次current更新；10%保持828次调用、822正LR及6零LR。评分前在固定独立`forever_score_claims`登记同run claim，换output不能重复评分；成功缓存直接返回，失败不自动重试。

原1%完整FOREVER核验器的18次历史回放检查原样复用；原10%完整核验器的动态阈值、逐current更新触发、全部dynamic/final回放、作者实际更新/梯度/曝光/AMP及资源scope检查原样复用。两者都复用原完整评分函数；AST只调整frame、版本来源、监督目录及“仅保留latest.pt”的旧断言，另插透明评分调用计数。不会把native核验冒充FOREVER核验。

binding公共字段为`condition/frame/sources/policy_sha256/build`，文件引用均为`{path: ROOT相对路径, sha256: ...}`。1%另需`launcher_binding/source_matched_gate/training_complete/terminal_exits`；`terminal_exits`按实际时间列首个training及其后每次resume的wait出口。10%另需`ready/completed_queue_snapshot/terminal_exit`，来源与gate沿新stage READY、BUILD及原队列合同。两比例helper和数学源均按实际producer绑定，原1%admission的旧I/O源不强套给新runner。

全19阶段checkpoint元数据、stage proof及commit必须保留，最终完整CP必须存在并由原核验器流式核SHA；旧中间权重允许人工清理，未观察到的删除原因不推断。暂停核请求、receipt、显式resume和分支记录，current events必须与各stage日志一致，不将归档分支计入学习。FOREVER的`rolling_io.pause_after_commit`明确将`reason=request['reason']`写入receipt；native receipt没有此字段，二者不能混用。1%builder复制该FOREVER模板，10%builder原样移植这段函数，因此这里核receipt.reason有producer依据。

1%waiter EXIT只记录pid/run/returncode/unix，没有完整进程身份；工具核真实wait出口，不声称核了缺失的身份字段。暂停receipt同样不证明未记录的中间进程事实。资源只保原各process/replay scope，不认证跨暂停进程的完整训练峰或减峰。

已完成两套原评分AST及两套FOREVER扩展AST装载检查，评分调用为0；合成元数据覆盖S0暂停/显式恢复、全部stage commit/events、缺中间权重允许、缺permit及缺最终CP拒绝。没有运行真实consumer、读取真实QA/预测/CP、调用评分、加载模型或连接服务器。尚待未来真实完整run验证，不把源码检查当GPU或正式结果通过。
