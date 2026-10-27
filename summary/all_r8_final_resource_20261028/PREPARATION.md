# 普通all-r8原模型：补齐质量—资源参照

已有完整1%结果五域宏F1为44.894422831，但原推理GPU峰缺测。该方法是必要的简单参照：只超过q/v的42.185908、只比PAS省显存，尚不足以说明当前任务梯度策略有净价值。本项只补测既成模型，不新增训练候选，不调整当前TaskGradReuse协议或占用其GPU。

采用原未合并S19图，保持原dtype、A/B和实际payload；原F1与被测参数图直接对应。168个七类位置的r8 A/B共有4399104个FP32坐标，载荷16.78125MiB。这是适配器载荷，不是完整模型权重或GPU峰。普通LoRA数学上可合并，但FP16舍入和计算路径可能改变生成，故不把44.8944直接配到未经评价的合并图；当前不为合并追加完整TEST或分支。

固定资源范围沿用PAS同一物理OLD GPU1、五个原始及准备后输入、每窗32新token，报告完整必要storage、加载峰、五窗allocated/reserved、TTFT及解码速度。PAS原记录无TTFT，不作速度优越性比较。独立重载后确认168个原SeqLinear、r8/scale4/dropout0.1及FP32 A/B，保留隐藏引用审计；不训练、转换dtype、合并、导出或重新评分。源码尚在本地准备，未运输、排队或GPU执行，主表不填估计。

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
