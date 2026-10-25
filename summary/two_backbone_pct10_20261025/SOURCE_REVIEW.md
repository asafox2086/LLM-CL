# 10%数据、独立runner与调度：源码交叉审阅

**accepted：true，限下列源码与本次独立10%流程。** 这是静态实施审阅，不是数据已生成、GPU准入、训练完成、完整评分或新科研效果的证明。审阅者本轮编写过协议草案与空表，但未编写这三个Python文件；曾参与旧基线CPU调度，披露该背景，不称整体框架从零独立复现。

| 本次源文件 | SHA256 |
| --- | --- |
| `exp/two_backbone_pct10_20261025/prepare_pct10.py` | `f21e6a86fe4f2b49bd269840820061114dcd4271fb5841bee9a47ac8906e5f8b` |
| `exp/two_backbone_pct10_20261025/build_runners.py` | `d622dc94539a15d4a7f58e38750fd192dddf28c98e91eb17c62ef91cd51c393c` |
| `exp/two_backbone_pct10_20261025/dispatch.py` | `0d7a4c849b3f22978b70d6cb77e14db892f97b84ffbea594edc817b5dca4146f` |

只解析并阅读源码与既有小元数据／原协议。三个源AST解析通过；未执行其main、未导入runner／Torch、未连接服务器、未启动模型、未做CPU模型门／评分、未读取新partial或原TRAIN／TEST JSONL／checkpoint载荷。

## 发现并在冻结前修正的真实问题

初版builder只转换run.py的condition／train_fraction／primary_score，但FOREVER原`bind_context`另有两处固定1313的训练量assert，10%会在初始评价／预检前失败。已立即告root与实施者；实施者仅在新clone将这两个guard改为manifest TRAIN总量，原19任务约束及算法不改。

当前builder明确把FOREVER的`forever_adapter.py`列入changed_py及additional_py_diffs；其余原py按byte-identical名单记录。每项精确替换都能逆变重建原始全部字节；不再声称FOREVER仅run.py改变。此为输入规模assert扩展，不是方法／回放／损失修改。未覆盖原FOREVER源。

## 数据契约

已对照原`subset_one_percent`实现：同dataset／原question_type或content_type分层、同ceil比例目标、最大余数及tie顺序、`SHA256(42:ID)`层内前缀；仅比例为0.10。不是逐task独立抽样、补最低题数或按TEST选择。代码硬绑定合法full与pct01清单原SHA及各自split_lock／frozen_ids。

`prepare_pct10`先校验来源文件SHA／计数／非空答案、TRAIN和TEST隔离、全局ID唯一、pct01逐task嵌套及同ID原对象摘要一致；完整TEST要求原路径／SHA／ID顺序及对象一致。新TRAIN直接选取合法full原JSONL字节行，不改QA；唯一既有SLAKE:1622排除保持，不另加排除。13105是准备代码的待实际校验断言，不由此审阅宣称已经真实生成。全部必要源／嵌套检查完成后才新建目标，不覆盖已有pct10目录，产物最后设只读。

## runner／FOREVER兼容

原native及FOREVER的run.py只转换10%入口与准确metadata（FOREVER原primary已经token_f1，其替换为不变操作）；FOREVER另有上述两数据量guard。训练循环、方法实现、学习率、rank、dtype、CPU offload、原公共图像／缓存、原Answer F1、19任务及74格、checkpoint代码全部继承。SMoLoRA的native-BF16 guard不改，OLD ownership排除它。

两clone深度与原一致，`methods.ROOT`保持原计算且解析到新的输出根。ROOT／SOURCE_ROOT必填或显式确定，拒绝已有runner或构建receipt；所有非修改py逐文件记录原SHA和新SHA。输入manifest／frozen_ids／lock校验完成后才输出clone。只调用cloned pipeline.launch，不运行原pipeline.main，因此不会继承其中对full队列的自动添加，也不会读写原pct01 queue。

FOREVER实际`pipeline.sources/idle/launch(job,phase,gpu)`接口与native一致；preflight／独立checkpoint_probe仍是原实现。正式training由launch新建不同run、不传resume，从预训练fresh创建，不能使用预检权重；checkpoint_probe原逻辑会删除其预检checkpoint。FOREVER原rolling commit／完整method graph及外部helpers不被本扩展替换。

## 调度与退出证据

一个独立dispatch按物理卡统一协调两family／两backbone，OLD14个非BF16组合，另2个SMoLoRA明确waiting_native_bf16_on_new。新host默认拒绝启动，只有人工协调1%dispatcher后的显式handoff环境标记才进入；NEW0不分配。当前review不是该handoff授权，也不声称NEW有空卡。

每phase再次检查manifest／全py来源及原ownerless／free-memory／util guard后，复用原pipeline.launch。wrapper与GPU child均记录UID／PID／startticks／隔离PGID；phase status和exit分文件持久记录，只有真实returncode0、preflight／reload proof及manifest／source精确匹配才进入下一步。正式结束仅标completed_unverified，不伪造完整评分或填表。监督丢失不会盲目重启可能活着的worker；没有向现有1%或他人进程发信号的代码。

14组合使用独立pct10路径、manifest、runner与queue，不复用1%结果／checkpoint／重跑sealed1%。每phase的idle guard和单一协调器避免本队列重复占同卡；外部进程仍可能在检查之后竞争，不能把一次空闲快照当永久排他锁。

## root部署条件与结论边界

新overlay的runner文件须实际位于同深度新ROOT；manifest经resolve后必须仍在该ROOT内。模型／既有source assets可使用只读symlink，不能把软链接解析到旧根后误称新ROOT。原`.vision-env`、analysis／作者code与两模型资产要由root按实际路径绑定，此审阅没有SSH确认它们。

OLD quota满而使用RAM／shm可用于计算和新输出，但不等于抗主机重启的耐久归档。dispatch的12GiB瞬时空闲检查不为并发模型及SeqFT19份完整checkpoint预留总容量。root需按实际容量与归档进度协调，在保留证据／必要checkpoint策略的前提下收集到NEW并验证；不能静默删除旧证据来维持队列，也不能把RAM保存称权威耐久副本。

未发现其余启动fatal或科学条件替换。接受当前三源用于下一步元数据生成／独立部署准备；实际数据freeze、原始源镜像、环境／GPU／资源、启动与逐方法完整核验仍分别以真实receipt为准。本文不把源码通过当论文进展，不授权修改1%科学源或启动100% full。
