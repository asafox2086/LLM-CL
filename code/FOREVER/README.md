2026-10-18后续修订：标准ER-LoRA已按用户要求移出当前比较，历史结果保留；唯一回放比较为待准入FOREVER。

2026-10-18当前方法范围：移出未进入正式训练的FV-guided Training与ALER，保留7项已准入方法；新增FOREVER（ACL2026）待图文适配及GPU准入，计划8项。原作者FOREVER为文本任务，不能称原生多模态或直接以作者多轮脚本冒充同预算。见[方法清单及准入要求](../../summary/two_backbone_pct01_20261017/BASELINES.md)。

# FOREVER

论文：[ACL 2026](https://aclanthology.org/2026.acl-long.1144/)；分类：replay。

作者仓库：https://github.com/WoodScene/FOREVER

核验依据：ACL paper footnote and official README

状态：作者源码已下载；待适配。固定版本：`bbeb549a21e59d9490f20a2b02099ee766a67f48`。

备注：原作者模型和任务需适配；未运行新实验。

`source/` 保存作者源码，原文件未修改；未安装或执行上游训练流程。已在CPU限定执行原记忆选择器并核验调度边界，不是模型实验。仍未完成两骨干图文适配和GPU准入。

[本1%预算接入审计](../../summary/two_backbone_pct01_20261017/forever_preflight_20261018/RESEARCH_FINDINGS.md)：24步校准覆盖全部19阶段主更新；ratio=2的原选择器最后选638/1264历史题，不能标为2%上限。最终回放仍保留，未测性能。
