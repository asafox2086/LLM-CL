# 从最强OSFT的额外变动出发

**要回答：** OSFT提高习得能力，是否有一部分来自普通LoRA没有更新的位置？能否用普通小适配器复现这部分收益，而不照搬大训练空间？尚未证明额外位置是唯一原因，也未得到新性能或部署优势。

| 对比 | 实验 | 小测结果 | 结论 |
|---|---|---|---|
| 最强方法与简单基线 | 原OSFT、原q/v顺序LoRA | 待GPU | 同小测参照，不拿正式AP直接代入 |
| 增加位置是否有用 | 普通LoRA覆盖与OSFT相同的七类投影 | 待GPU | 不预设新增位置必有益 |
| 只改额外位置行不行 | 普通LoRA只更新k/o/gate/up/down | 待GPU | 额外位置若有效，再定位必要部分 |

OSFT正式AP46.357、顺序LoRA40.286；OSFT刚习得成绩较好，BWT却不更好。源中还存在更大训练空间、SVD／投影、原学习率差异；不能把6.071分全部归因于位置或保护。

四条件共用冻结1213 FIT／100 DEV，DEV每领域20题，由原1313 TRAIN内按图像和作者场景ID分组留出。训练仍19原生阶段，实际87正LR更新；小任务7/8/9/12没有独立DEV，不能称覆盖全部原生任务。原始问答和图片不修改，官方TEST从不读入。各条件共享FIT／DEV和公共FP16数值条件；普通LoRA沿用rank8／alpha32／dropout0.1，避免另设任意极小预算。

**只有候选DEV五域均分高于两个参照，才考虑用全部原1313从基座重训并完整确认。** 小测点差不是显著性或最终结论；所有正负小测均报告。即使小测成功，普通LoRA合并后整模不自动小于OSFT，真实端侧目标仍未实现。未经完整质量与部署确认不填主表Ours。

运行入口exp/osft_increment_screen_20261019/v1/dispatch.py --watch，沿用唯一nohup服务，CPU1154006/start9767347；权威queue/state.json与queue/ACTIVATION.json。旧1016078及旧容量全测入口不恢复。科学源／SPLIT／协议已冻结；小HF检查只属工程准入。实际GPU1仍他人占用时健康等待，四组未有STARTED／COMPLETE，不说已训练。两机正式基线保全。

[协议与正负结果用途](PROTOCOL.md) · [共同ID划分](SPLIT.json) · [科学源绑定](READY.json) · [CPU门槛](CPU_GATE.json)。初次准入的queue.py命名遮蔽stdlib导致torch导入失败，发生在READY／GPU／模型之前；旧源与失败描述保留在admission_failure，准入前改名dispatch.py后通过，不是方法负结果。
