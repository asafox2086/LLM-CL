# 启动前来源绑定修正

首次运输检查在创建实验目录、模型导入或 STARTED 前拒绝，原因是用本地版本计算了继承源摘要。实际冻结 OLD 的 run.py 未含本地后来增加的阶段保存/暂停外层，execution_policy.json 保留原主机卡分配。没有改动 OLD 源或科学设置；改绑定指向已有 OLD 字节，保留最初绑定于 BINDING_LOCAL_PREPARATION_HISTORY.json。

run.py 实际 SHA：0e1f794ff1daf0694bdf960c3f89ccd4078764577db75271d8f78684282a39a9。
execution_policy.json 实际 SHA：f6d33cc939ff833ec7799d2d1ec4c238bb97d47d35488034d9174b36f07c2502。
训练 AST 约束仍为原 2b31ff9e…dda7f。主训练、样本、预算、评分与五个科学源不变。此次不是模型失败或救跑。原差异源码与只读回执私有保留。
