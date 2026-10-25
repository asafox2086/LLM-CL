# 预训练读出对应对照：首次真实拟合失败，质量缺测

研究问题是小塔的预训练 patch-channel 读出对应是否能承接原语言骨干的回答功能。Native 与唯一输入列排列控制同容量、同数据、同训练预算；本次没有得到回答质量或部署优势结论。原协议、九源、审阅和 READY 不变，不把静态审阅通过算作实验通过。

OLD unit `llmcl-spatial-readout-transfer-20261025.service` 首次 fit STARTED 为1792908490.297819，PID3933836/start760690988/UID1084/PGID同PID/GPU1。实际首对特征传入 alignment._check_pair 后，同CUDA设备检查通过，`raw.dtype == teacher.dtype == torch.float16` 检查失败。两张量的实际 dtype 未持久化，不能推断哪个张量、哪种类型或具体算子负责。模型已经实际处理首个 FIT QA 的视觉，不能称0模型请求；未产生完整对齐矩、compiled载体或任何训练/回答结果。

FAILURE 为1792908505.110589，EXIT1为1792908505.9614124；STARTED→FAILURE14.812770秒，STARTED→EXIT15.663594秒，二者重叠、不相加，仅墙钟代理，不是完整模型RUN或纯GPU算时。另有事前资产准备0.050925秒、环境元数据检查0.018967秒、准入4.483917秒，以及运输、源码实现、审阅、发布和观察成本；不能只报有效更新为零便忽略这些费用。

六阶段中仅fit STARTED；两臂训练、两臂评价与父资源阶段均未STARTED。0反向、0优化器更新、0新生成、0新评分、0TEST，无COMPLETE／ALL_EXIT／VERIFIED。2026-10-25T06:10:34附近的故障收据实查unit failed/MainPID0/exit1，原worker身份已无；这是时点证据。OLD源目录及原错误字节保留，没有重启、改精度、加cast、放宽断言、重评分或补做模型。

root与两位子agent判断一致：实际特征dtype合同失败属于数值接口，未满足原协议的纯I/O元数据修订例外。改cast/autocast或放宽检查会触及模型/统计输入，不能作为不改变实验的静默修复。因此关闭本次固定实施，不自动追加版本。两位审阅者披露本人组件作者身份和事前审阅未验证实际输出dtype的覆盖遗漏；审阅记录不是相互独立的实验重复。

这次故障不支持或否定预训练读出对应的科学假说，也不支持设计中的591534976完整系数／33.80875%减重实际成立，更没有同卡峰优势。原生紧凑整图的既有完整负结果仍有效，但不能拿来代替本pair缺失的因果对照。科研总目标未实现。

证据：[原协议](PROTOCOL.md)、[原始错误](fit_FAILURE.json)、[故障收据](FAILURE_RECEIPT.json)、[实施审阅](FAILURE_REVIEW.json)、[科学审阅](FAILURE_SCIENCE_REVIEW.json)。
