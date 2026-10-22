# 一组固定短侧因子的FIT/DEV先导

## 问题、已有证据与竞争解释

原LoRA_all完整确认未达到PASs/OSFT；NormOnly固定配方也未达两个FIT适应锚。这不证明全部原USV坐标必要。PROOF证明固定完整短侧补基不减单阶段精确允许集合；可检验的竞争解释是短侧训练仍有有限步优化价值，或相同功能可由长侧+S承接。不是1%特有因果、创新算法或部署项目。

## 唯一干预与固定条件

只新增OSFT_ShortFixed：m≥n固定V_low，m<n固定U_low，继续训练长侧和S_low；每次原stagewise reSVD后重新冻结。原168模块、七位置、25%尾rank、初始张量/forward/RNG、作者核、LR1e-5、HostAdamW/offload、数据/图像/提示/数值、seed42/1epoch/batch16/micro1及TRAIN AST继承。必要参数投影处理见PROOF，实际短侧各阶段前后全量SHA必须相等；不能仅requires_grad=False却允许回调修改。

不用首批norm匹配校准、不搜LR/左右侧/scale/投影组合。相同LR不匹配权重步幅；记录各阶段首更新的原autocast实际FP16重构权重前后Frobenius变化、原始梯度和clip系数，不作跨方法逐步动力学或受控时间对比。共享初态CPU小HF检查只准入，不代替实际质量。

原1313内冻结1213FIT/100已探索DEV，每域20，19阶段87正LR；T7/8/9/12无DEV。继承原分组SPLIT，不访问官方TEST，teacher不介入。复用原FIT-only OSFT和qv作质量锚；已有DEV不是新的留出。原全部1313 teacher见过DEV，不参与筛选。

## 正负结果用途与退出

Primary为五域原AnswerF1宏均分严格高于旧OSFT40.78380056394307和qv39.870438970139254中较高者；同时实际梯度及两Adam moment tensor字节均低于原全部USV的源代码计数455064576/910129152。此成本锚是原源码推算，不是fresh受控峰值/速度实测。只有两项都过才保留为后续fresh FIT-only PASs参照的条件候选，仍不自动启动。逐域、逐题配对和整已知组10000/seed20261022探索区间全部报告；相近点估计/区间跨0不当等效或稳定收益。

未过上述固定点估计及字节判据、或执行失败即关闭此配方，不搜LR/侧别、不自动1313/21454，不据旧TEST追分。区间跨0限制稳定性解释，但不另改点估计筛选规则：若两个判据都过，仍只保留给后续必要性评审，不能据此自动开PASs或完整确认。机制净作用仍可能未解；不能把“列作条件候选”写成已证明优于原方法。即使过也不填Ours/正式表，不称达到最强基线或完整目标。端侧只背景。

## 成本与完整证明

STARTED→COMPLETE总2h含原19FIT图文门槛38NLL+38短4token、87更新、19个完整checkpoint持久SHA、100DEV生成、逐阶段冻结SHA、首步168重构权重观察及全部保存；CPU小HF和独立核验另计。最多一GPU1研究流，原正式基线保全，不用GPU0/旧卡。原源/READY后冻结，故障留证独立版本、不热改/放宽/拼接/盲重试；只有STARTED/CUDA前资源变化75才是未尝试wait。

实际记录长侧+S每步原梯度、首步完整梯度、19validity、87实际正LR日志；每步optimizer参数ID精确匹配、frozen grad None、actual gradient bytes、HostAdam CPU exp_avg/exp_avg_sq bytes；19首步重构权重变化/clip；基座其余参数首尾digest，阶段短侧首尾digest；同PID资源monitor和TRAIN/RUN峰值按原范围公开。原全部USV张量仍在、临时短侧clone及首步CPU权重副本计入，不以坐标减少称整模压缩。

只完整封存后按原score AST复算全部100ID/全部参考/原AnswerF1、文件/权重SHA、19validity/87有限正LR、原梯度与新冻结/optimizer证据。最终checkpoint不是完整部署重载证明。SCREEN结束正常退出，0正式TEST/新回放/teacher请求。
