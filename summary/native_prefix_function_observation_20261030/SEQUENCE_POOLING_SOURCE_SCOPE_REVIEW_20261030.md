# 序列聚合选择器：源码范围审查，不是已验证模型结论

## 状态与研究去留

本次只读实际源码及既有调用链，没有读取checkpoint载荷、运行模型／新消费者／suffix干预／质量评分，也没有新增模型源码、runner、GPU调度或修改基线。当前不准入为端侧主线；实际checkpoint的后缀是否改变更早的选择或logits仍未验证。不能把未测疑点放入外层已证结论。

**不准入的原因是缺少部署成本机制，不是存在先例就否定适配创新。** 即使未来确认非因果影响，单独修正依赖不证明减少部署参数或内存；完整性核查不能替代用户要求的参数—条件功能—存储关系研究。未来若有独立必要性，可另行论证完整性核查，不能由本记录自动启动或修改冻结设置。

## 三个实际native源：1%／10%逐字相同

| 文件 | 1%实际路径（NEW仓库） | 10%实际路径（NEW仓库） | 共同SHA256 |
| --- | --- | --- | --- |
| methods.py | `exp/two_backbone_native_cl_20261017_v2/methods.py` | `exp/two_backbone_pct10_native_checkpoint_20261027/methods.py` | `e5bd8f51690b006274ed5adcfa4ef3b1a2c3b0f1d79476d948b71d1cfec41c5b` |
| runtime_utils.py | `exp/two_backbone_native_cl_20261017_v2/runtime_utils.py` | `exp/two_backbone_pct10_native_checkpoint_20261027/runtime_utils.py` | `4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240` |
| sparta_adapter.py | `exp/two_backbone_native_cl_20261017_v2/sparta_adapter.py` | `exp/two_backbone_pct10_native_checkpoint_20261027/sparta_adapter.py` | `8a00505b40663e5a233ddceb34e295d9b7b73d55f0679ac3d2b93318e77d7a44` |

这些hash证明两条件的相关数学实现相同，不证明两条件的checkpoint产生同样的实际影响。原模型状态、训练过程与输出需要独立实测，本次未做。

## 具体源码事实

1. **真实TRAIN输入与loss。** `runtime_utils.py:106–127`把gold tokens及EOS拼在prompt后，`mask=torch.ones_like(ids)`，完整序列进入decoder；随后只截取答案预测位置计算CE。该mask是padding／有效token掩码，不能据全1说attention没有因果约束；原因果attention仍由decoder实现。问题是另一路router没有接收该mask或loss选择。`run.py:262–269`对两法调用此原answer_loss，没有为router改成question-only输入。micro1没有跨样本padding可解释掉gold后缀进入聚合这一事实。

2. **SMoLoRA。** `methods.py:59–74,77–115`绑定并调用原作者forward。作者`smolora.py:405–479`在active adapter、r>0且未disable时，425行对x的整个L维作无mask mean，426–429得到hard top1，439–458将选择广播到全部token。attention_mask／labelmask未进入这个选择。`methods.py:87`默认disable_adapters=False，当前训练调用没有关闭该adapter；activation offload不改变数学。指令分支432行只读固定ins_type embedding，不读取gold；`methods.py:118–128`的静态指令embedding平均确实使用padding mask，不能与全序列x均值混为一谈。

3. **SpaRTA。** `sparta_adapter.py:19–34`配置project=0／nomlp=False，并对两分支独立sigmoid；53–57的group入口hook只把args[0]完整hidden送router，未接收attention或labelmask。作者`get_scale:1068–1086`先逐token变换，最终跨L及输出维求和为[B,1,1]，广播到组内各投影的rank2／rank8分支。训练与eval对历史K/A/W切片的差异主要为detach／梯度保护与辅助正则，不将输入裁成prefix。原DATAInjectedLinear:33–38始终计算两个rank分支，没有本训练调用中的adapter关闭路径。

## 对应作者源与hash

| 路径 | SHA256 |
| --- | --- |
| `code/SMoLoRA/source/PEFT_SMoLoRA/peft/tuners/smolora.py` | `46183bcc5d493a6fbe268436ca8b51cca27f5b133823608fb5907ea13ee3a3b3` |
| `code/SpaRTA/source/patch/modeling_qwen2.py` | `f8760fef97f8b300955fa5e93fb53ae678f4ee97058d38a7ebbed8937f795a37` |
| `code/SpaRTA/source/src/llamafactory/model/inject_data.py` | `b9eb2f999898df975b30e091eb3b434ebbf225e7deade3afe6e013c439d0eb9b` |

## 哪些尚未证明

上述源码可证明潜在的“后缀→全序列router→前缀投影”信息路径，不能证明某个实际CP／TRAIN pair的早期logits已被改变，更不能证明它导致低F1或训练—生成质量差。SMo hard top1可能不跨选择边界；SpaRTA sigmoid可能饱和；专家分支贡献可能很小。因此分数改变不等于最终输出改变，少量pair未变也不证明全域因果性。PaRSP等非扩展方法及token-local PAS不由这两个序列聚合实现代表，不能称全部CL共同失效。

## 已核一手先例与创新边界

[Beyond Causal Masks: Auditing and Localizing Hidden Future-Token Leakage in Adaptive Sequence Models，arXiv:2610.31522v1（2026-10-22）](https://arxiv.org/html/2610.31522v1) 已直接讨论全序列summary／动态路由绕过因果attention，并使用固定prefix、改变suffix的prefix-invariance检验（§4、§7.5）。因此“attention causal不保证全部辅助模块causal”和此成对测试本身不能称本项目新发现。我们两种已训练CL实例中的实际影响仍是未测问题；即使测到，其适配研究价值也必须另由明确部署成本与功能机制支持。

本项收束，不自动新增模型／方法／实验／消费者或修改原基线。
