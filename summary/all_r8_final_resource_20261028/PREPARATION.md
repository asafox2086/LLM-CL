# 普通all-r8原模型：补齐质量—资源参照

已有完整1%结果五域宏F1为44.894422831，但原推理GPU峰缺测。该方法是必要的简单参照：只超过q/v的42.185908、只比PAS省显存，尚不足以说明当前任务梯度策略有净价值。本项只补测既成模型，不新增训练候选，不调整当前TaskGradReuse协议或占用其GPU。

采用原未合并S19图，保持原dtype、A/B和实际payload；原F1与被测参数图直接对应。168个七类位置的r8 A/B共有4399104个FP32坐标，载荷16.78125MiB。这是适配器载荷，不是完整模型权重或GPU峰。普通LoRA数学上可合并，但FP16舍入和计算路径可能改变生成，故不把44.8944直接配到未经评价的合并图；当前不为合并追加完整TEST或分支。

固定资源范围沿用PAS同一物理OLD GPU1、五个原始及准备后输入、每窗32新token，报告完整必要storage、加载峰、五窗allocated/reserved、TTFT及解码速度。PAS原记录无TTFT，不作速度优越性比较。独立重载后确认168个原SeqLinear、r8/scale4/dropout0.1及FP32 A/B，保留隐藏引用审计；不训练、转换dtype、合并、导出或重新评分。两源已发布并整理为NEW独立快照，未运输到OLD、排队或GPU执行，主表不填估计。

## 已实际核对的输入

unix1793136997于NEW只读核对：原最终CP仍存在，1941599549字节，UID1004；stage19元数据与latest、原COMPLETE一致。生产端CP SHA为416ed177803b9bf986dacd73a6d1bf18c6c0c59a1eb65f42785d81a628d31519，本次没有重复读取CP本体或加载模型。

- 原VERIFIED：abbd2da43753c6bbeb137d7bef535983efbcafdaf08686f61550886b2f18598d。
- 原READY：a8c4b99aef2e7b5e803d027e45d31aebd5ad60df49fea7c2b459d719249da628。
- 原COMPLETE：ea9bd2283e2cc7d8948267289ff2e4cc7a20685cdc7b511fc176b628870d6a5c。
- 原config：d346e0174fa6c0f714b94e161f617375268e13c5f1d228c0d03c43af933bfbca。
- 原stage19元数据：f9c7f55f568da722f22a6303036748730046685f06156b9da88edf70fc61cf48。

37项原READY源码中36项仍匹配；全局execution_policy.json后来已有历史修改，不能直接用当前全局目录冒充原运行快照。原3785字节policy已从Git提交da19582f978437b82c3b7294699803cc5369b783中找到，SHA为f6d33cc939ff833ec7799d2d1ec4c238bb97d47d35488034d9174b36f07c2502。未来只在独立测量快照恢复该原blob，不改全局路径或活跃实验。原CP动态TargetLoRA的解析与真正GPU重载仍待验证；上述元数据核对不是部署通过。

plain_r8_control独立评判接受原图最小补测，理由是它填补公平参照，避免为数学合并扩展新F1链；root复核SeqLinear前向与原TargetLoRA来源。当前正式训练及基线不受影响。

## 已准备源码与验证范围

| 独立源 | SHA256 |
| --- | --- |
| exp/all_r8_final_resource_20261028/worker.py | 2553fb619d0ee7b9eb88a6470e4906c4e983e989d8a7f3b888932eb6892ef96d |
| exp/all_r8_final_resource_20261028/verify.py | 117d4c5603a809ed78e9b4912e78be236efb68b68b250901868a2ad8c4ff118d |

仅复用已有storage/readouts/TokenClock三项helper。worker接口为`--root 独立快照 --output 新目录 --binding 绑定文件 --gpu 1`；verifier沿用原资源EXIT契约。600秒上限，不重试、不覆盖输出，不包含新分配器或服务。原TaskGrad资源两源保持不变。

plain_r8_control完成两源AST、真实封存元数据契约及错误方法/错误policy拒绝检查；root独立重复两源AST和真实封存元数据契约通过，未导入Torch或触碰真实权重。原固定TargetLoRA类对象的cloudpickle/pickle元数据往返已由实现者检查，不调用构造器、不创建模型；真实GPU图重载与资源仍未验证。

baseline_review独立只读审阅两源、原动态类、checkpoint保存加载和OneVision别名后接受，未见确定阻断；核对了原168模块、完整权重、元数据链、隐藏引用审计及五窗计时范围。该结论仅接受可执行准备，不是实际CP重载或GPU测量通过，也没有启动、运输或排队。

## 独立资产快照已实际完成

unix1793137965完成NEW独立快照`.deployment/all_r8_resource_snapshot_20261028`；45个文件含37项原READY源码逐SHA一致，原policy仅从上述Git blob恢复到快照。最终CP以同盘同inode硬链接保留，已实际完整读目标核SHA416ed177…31519，前后源/目标device、inode、大小、mtime一致，没有写入或chmod/chown原inode，也没有额外复制1.94GB载荷。data/model只链接既有资产，不改内容。

执行脚本经baseline_review独立只读审阅、root AST核对后执行一次；随后独立只读检查45项文件、源/目标inode和原全局policy未变。耐久SNAPSHOT收据SHA为384cedd0aee02a089ebc1bd580dbb9bfa61c68162f885238a889cd17f7472d0c，[精简实查凭据](ASSET_SNAPSHOT.json)。剩余空间43,417,968,640字节，保留20GiB余量。本步不产生模型、评分、resource binding、GPU启动、OLD运输或队列更改；仍是资产准备，不是资源实测通过。

## OLD资产运输已完成，尚未GPU测量

unix1793199481，47资产已运输到OLD `/dev/shm/llmcl-all-r8-resource-1084-20261028`。46小文件逐SHA、1.941599549GB完整最终CP整SHA通过；原NEW快照/CP保留。数据与模型只链接OLD既有资产。剩余20,001,214,464B，运输前按剩余正式5阶段CP约9.7GB加8GB余量检查，未占研究GPU或改正式运行。root另独立核46小文件SHA、全部47文件stat/UID和两本地链接；NEW/OLD收据JSON内容一致，序列化字节不同，首次字节比较拒绝后显式记录此区别，未重放运输或再次全hash CP。

[运输收据](OLD_TRANSPORT.json)及[独立检查](OLD_TRANSPORT_CHECK.json)已保存。一次性运输脚本已执行，不可重放；原NEW snapshot准备也不可重放。当前还没有resource binding、运行环境核验、GPU重载或实测，不能称资源通过。后续沿原PAS同GPU1/五输入/32token，不启动额外分配器。
