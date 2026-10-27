# 成对选项监督实现状态（2026-10-27）

这是实现与输入核对记录，不是方法有效性结果。正式两张主表、Ours状态不变。

- worker使用冻结的普通q/v-r8初始化和原87次更新循环；原1213个QA组不重新分batch。33个FIT在组内进行两次真实前向并平均CE，外层仍除以原minibatch QA数。repeat原样两次；paired第二次采用已认证循环及映射gold。
- 12个DEV只在S16刚习得后额外生成；与正常before/acquired读数一样保存并恢复RNG、训练模式和梯度开关。两个fresh训练，未接旧adapter快照。
- 服务器纯stdlib输入绑定已实际通过：33 FIT、12 DEV，原行字节SHA、图像标识、置换与标签映射检查通过；没有Torch/模型/生成/评分。证据见PAIRED_INPUT_PREPARED_20261027.json。绑定不等于GPU功能通过。
- 四阶段controller已落地：每臂train5400秒/eval1800秒、总634次生成；只允许首次执行。核验器正在实现600份原样回答唯一评分，24份循环回答仅作严格映射读数。
- 新allocator包装草稿只接原分配器idle，不启动第二个分配器。尚未部署或暂停基线。研究GPU进程尚未启动。
- unix1793077342核实OLD原四baseline worker身份仍匹配，PaRSP新保存版本1379913/start777060360已到S0/T16评估。准备好实验后再请求阶段保存暂停；尚无可称已保存的PaRSP暂停checkpoint。

风险：双前向同时持有计算图，训练峰值可能提高，需实测；部署图设计不增参数，但完整参数/重载仍须GPU结果证明。小DEV收益不直接填正式主表，也不能把位置敏感诊断推成全部失分根因。

独立research_review已接受当前worker/helpers/controller实现，逐路径核实两真实前向、原QA归约、33/12隔离、RNG恢复、循环与gold映射及fold/reload接口，无实现阻断。对应SHA见输入收据；这仍不是GPU结果。队列包装另审中。
