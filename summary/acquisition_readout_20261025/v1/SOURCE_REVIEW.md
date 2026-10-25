# 刚习得诊断：独立源码交叉审阅

**结论：接受当前源码进入ROOT资产/READY最终准入；未发现新的明确致命问题。** 这是只读源码审阅，不是实际重构一致、GPU运行成功或质量结果。没有模型/服务器/CPU神经门/评分操作，也未修改实现者文件。已与baseline实现者直接核对最终verify及controller EXIT合同。

## 本次审阅绑定

- worker.py：`a35f0c090524019616a0f0c6b94f2698914ce55b17d13f5a48aeea2001ffe16b`
- controller.py：`66e5cb37d9a3b8f308068716c6acdbcaf6d986a7e7fe40d67a4fbf970cd336b4`
- verify.py：`f5951d82919691dcf4ad195529e91d075f8702af418642920c0a69b57ca45f2f`
- PROTOCOL.md：`39a318bf4aa60c3d352cd5565dd982a4d541b9a6847519c35b8ae2f08065ad4e`

旧ARC/Naive源码和已封小证明只读；旧ARC READY实际14个source条目＝8个py＋6个文档，verify最终已按这个真实schema核验，没有重复上次假设8总条目的错误。ROOT已将controller写出的退出记录统一为`acquisition_readout_EXIT.json{pid,returncode,unix}`，与verify读取一致。

## 科学必要性

唯一动机是共同原法相对Naive的遥感+35、医学−10终态变化。读各QA所属task刚学完状态，直接改变下一步应研究有效习得还是后续迁移/保持的判断；不是重数门率或旧OSFT刷新/几何分支。两个方法同40个已探索DEV、80新回答，不扩域或训练。没有方向/逐域胜出/显著性门；两域acquired聚合不同task-stage模型，不伪称单一域末模型，也不更换原五域总目标。

## 关键代码与字段承接

1. **两S19确实先于任何生成。** worker先循环两个方法的S19重构，逐个完整receipt对照各自parent checkpoint并检查状态不变，随后写S19_CONTROLS；该循环完成后才进入包含唯一`_generate`调用的stage读出循环。两个控制各0生成，不能只过其中一个就采回答。
2. **full receipt比较方向正确。** worker的same_receipt遍历实际`deployment.receipt`全部语义字段，排除device；父original_base/checkpoint含这些字段。它涵盖完整权重/全buffer/layout/别名/配置/attention类等，不会误要求新命名容器的path/bytes/SHA等于父容器。verify另有显式完整RECEIPT_KEYS比较，不能只靠一个局部digest或系数总数。
3. **快照承接完整。** 每stage独立load该方法原initial完整包；源train_COMPLETE、VERIFIED、stage证明和快照SHA成链。96个FP32 working A/B名称/形状/540672系数/2162688bytes、封存adapter_digest与复制后的digest全部核对。原参数对象copy替换值而非叠加；冻结基座和普通buffers不变，不能把末态已fold包作base或累计fold。临时slow buffers不恢复旧EMA且在原fold替换Linear后移除；原EMA不参与working默认推理。
4. **阶段与输入一一对应。** 原manifest/SPLIT确定每域20ID，stage升序和DEV原顺序一致；有实际DEV的10个task固定为2/3/4/5/6/10/11/13/14/15。worker显式产出`method,id,task,domain,acquired_stage,phase=ACQUIRED_DEV`，verify逐题核task映射、参考/cap、实际raw/transport/prepared/image/shape与已封旧输入。上次producer遗漏phase的契约在此明确补齐；旧final prediction只用作输入绑定，不参与评分。
5. **状态/来源不以自报替代。** 每stage重构receipt和评价后invariant、逐题输入、stage事件和文件SHA都由COMPLETE封存；verify检查期望文件集合、两S19、80个唯一(method,ID)、PID/UID/PGID/startticks/boot、EXIT0及源/环境/资产链，并承接旧ARC两次公开修复证明。实际端侧图仍须通过893675552/1787351104及9032buffer计数；源码审不替代GPU核验。
6. **唯一80评分与旧final分开。** controller只启动一个worker并在退出后调用一次verify；verify在全部完整性检查通过后写VERIFY_STARTED，再原scorer评分80新回答。原original/Naive逐ID final_F1直接从各父VERIFIED复制，不重score旧prediction。逐域/逐IDacquired、final、后续变化及方法差的代数分解完整输出，无bootstrap或挑方向。

## 预算与解释边界

单物理OLD GPU1由实际STARTED绑定；每stage不另开预算。1800秒覆盖首次STARTED后的全部模型加载/快照hash/重构/80生成及模型证明；继承Context在STARTED前做的源/prior/environment元数据哈希属于外层监督成本，controller从启动起2100秒覆盖，不应隐去这些成本。首次真实模型故障/超时不重训、不扩QA或自动重试，旧RUN和旧科学源只读。controller的1850秒子进程等待为1800模型时限加启动/退出监督余量，不是把模型预算改为1850。

S19精确通过仍只承接统一fold后的部署函数，不证明当时unfolded训练图逐位回答相同。`final−acquired`包含后续有益/有害迁移与该部署口径，不能单独等同遗忘；ER/EMA/曝光整包和原GPU0/2训练差异不构成纯因果。只要ROOT冻结的源、旧证明、40个ID和预算与本合同一致，当前实现值得作这次有限诊断；本审不授权启动或后续分支。
