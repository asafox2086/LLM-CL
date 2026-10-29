# 固定范围：已提交S14–17的新习得扩展

只新读取原formal training_v1 S14–17的stage/checkpoint元数据、stat完整CP与各task initial/before/acquired .scores.json；不读QA/预测行/CP payload，不生成模型或新score。复用已封存S1–13 snapshot，绝不重跑其消费者。

主问题：后续领域能否延续前13任务的竞争力，或给出反例？并列T14–17原F1及对Seq/PAS/OSFT差，特别单列T16金融；前17任务等权和各已覆盖域QA加权严格区分，按manifest标识域覆盖完整性。新四任务均分只描述后缀，不当宏平均/因果。OSFT初态不同，不据其gain声称对齐方向因果。

唯一新snapshot及comparison都只执行一次，结果无论正负都保存；不改方法、预算或主表Ours最终行。S18/19不消费。失败中断不等方法负结果。

S14/S15历史payload已按用户授权保留策略删除；完整提交stage/meta及原scores仍保留。此扩展以这些不可变聚合证据为准，记录payload是否存在，不加载权重，不冒充完整最终核验。
