# 同输入的FFN联合删除损伤诊断

问题：已有单删的真实答案NLL变化，能否预测已经失败的固定8层联合删除的损伤？不是新压缩方案，也不调整旧方案。旧质量负结果保持关闭。

证据缺口：原直接干预在37 FIT记录父模型及24个单删NLL，但联合删只在另外100 DEV评价。跨集合差不能证明非加性。用原同37 FIT和原完整联合图补37个NLL即可闭合此问题。

冻结：原selection.json、READY、VERIFIED和pruned_complete.pt；原8层[2,5,13,16,17,18,20,21]、原37 FIT/参考answer、预处理、FP16/autocast/SDPA、EOS和512目标规则均不变。加载同一完整包、严格重载digest/layout，绑定每条prepared input digest与target_tokens。无训练、梯度、生成、DEV/TEST、重评分、bootstrap或再次选择；不能用本结果恢复、缩减幅度、增恢复训练或搜索。

逐QA计算J=新联合NLL-旧父NLL，S=sum(原8单删NLL-旧父NLL)，I=J-S。S使用有符号变化，不用截零cost。按五域各自QA等权报告mean J、mean S、mean I、mean abs(I)，再五域等权宏；公开全部37标量，原题目/图像/答案不入Git。不把各域不等量的QA直接混池称宏。

若abs(mean I)>abs(mean S)，仅描述净非加性余项超过净单删累计项，不当显著性门槛。净I接近0仍检查mean abs(I)，无误差容限就不称可加/等效。不报告分母近零比例。

I包含NLL本身的非线性，不能单独归因隐藏状态漂移、层交互或其唯一因果。若正向非加性明显超过累计项，则不能再以独立层代价之和约束本构造的联合损伤，下一结构选择须评价集合效应；若不支持，则拒绝拿强交互作为追加恢复/漂移修补的依据，承认已有单层代价与固定预算。混合情况按域/QA报告，不强选统一故事。无论结果如何，均没有新的保质小模型或自动下一实验授权。

一次真正空闲OLD GPU1，STARTED起600秒含加载/严格重载/digest/37前向/保存；原模型与其他进程保全，失败不自动重启。已有原父/单删基线只复用封存数值，源VERIFIED与selection及CP完整SHA绑定，不重复925请求。新的独立进程实际exit0且COMPLETE/全部输入/有限数/权重不变通过后，唯一consumer读37新标量并计算；观察超时不能启动另一份。

原始文献：[BlockPruner，Findings ACL 2025](https://aclanthology.org/2025.findings-acl.262.pdf)已有基于perplexity的迭代剪枝；[ShortGPT](https://aclanthology.org/2025.findings-acl.1035/)使用BI评估层影响。此诊断不是对这些论文的推翻，也不将联合评估本身命名为创新。
