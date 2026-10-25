# 三臂完整结果与下一实质问题

科研总目标未完成。结果已push `8ef2ddae31cf034dea0279baed667c617d0e47ae`；正式仍8/16、Ours空，当前100探索DEV不能填正式表。最外层主表现有18列（五域宏与AP19分开）；原18方法行旧17列保留。

六模型phase全部COMPLETE/EXIT0、六身份已无。`VERIFIED` SHA `84ed17f99ed21a01ffc6f7a5ba479ed3670c4ba34b54a0e2037a324a92876aa8`；`CONTINUATION_VERIFIED` SHA `1d8b0aaf71541c98304361ade6f627faf1e21ceadccaa1eb2357e357659e081c`；`POST_MODEL_SCHEMA_VERIFIED` SHA `d48977c265597ce67ec8eab7ab0f7284466fa87d12e61e271ed04dfea551d954`。结果／独立评判／成本分别见OUTCOME、RESULT_REVIEW、FINDINGS。

宏gate43.7198906634／original43.7032623794／shuffle43.6719364985／naive38.7753697307。新增门仅+0.016628pp／+0.047954pp，四个非交通域80题F1逐值同；425/886接受、Hamming392，事前点方向成立但不称显著、稳健或机理确定。共同配方相对朴素：遥感+35pp、医学−10pp，自然/金融0，交通略负。约+4.93pp主要已有于original，不能归门。三图均893675552个FP16系数/1787351104bytes＋9032普通buffers；五窗与朴素r8逐字节同。相对强参照的资源优势来源是共享图、骨干和dtype。

原preSTARTED资源拒绝四件保留于`preSTARTED_guard_attempt_20261025/`。六模型成功后，原verify要求producer未保存的phase而在VERIFY_STARTED前KeyError，0评分；旧两错误原字节保留于`post_model_verifier_schema_failure_20261025/`。外部wrapper只修一条Assert、逆变换恢复其它全部AST，8科学py与6绑定文档/READY/生成不改，原评分300次首次完成。两旧unit保持failed/1，新post-model schema unit已inactive/dead/MainPID0/exit0，扩展CV透明披露修复，不能以旧failed服务误判模型未完成。原绝对截止未延长。

**所有collector、verify、consumer与archive均已完成，不再调用或重新评分/bootstrap/hash。** NEW私有`exp/answer_reliability_consolidation_20261025/v1/private_run`已归档732新输出逐SHA通过，134.355476790秒；OLD原目录保留。客户端ACTIVATION两个CRLF格式变体与RUN JSON值相同，旧客户端原字节另存`client_receipts_before_exact_collection_20261025/`，主文件现为RUN精确字节，见COLLECTION_RECEIPT。

**下一问题：遥感收益/医学损失在各QA所属task刚学完时已形成，还是后来跨域更新中累积？** 先读[源码可行性与必要检查](ACQUISITION_READOUT_FEASIBILITY.md)。两方法均存19个96-working-FP32适配器快照；源码上可从各自未适应initial完整基座重构，0补训练。拟只读两域各20QA/两方法的刚习得函数，共80新生成，旧终态只引用sealed标量。

先用S19快照按原GPU FP32禁TF32/autocast后转FP16的fold重构，完整权重/buffer/config/别名须与各自原末CP精确一致（0新S19生成）。尚未实际加载或通过该检查；不匹配不能补训练救齐。统一fold后的阶段函数不冒充当时unfolded输出；训练GPU差异及ER/EMA整包仍不能称纯因果。尚无新PROTOCOL/READY/模型准入，须先形成具体协议和必要评审，不恢复旧失败分支、不按DEV追rank/LR/gate阈值。

最新基线独立快照见BASELINES_AFTER_ARC：NEW监督1221743和CL-MoE8260/SMoLoRA8265真实身份仍在，5项原队列等待，GPU0不由我方分配；这是时点记录，下次重新检查。只有新完整sealed proof才增正式结果与参数统计。
