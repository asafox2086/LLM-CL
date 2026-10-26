# Qwen全19任务跨比例习得：独立结果审阅

接受已生成结果及FINDINGS，未发现实际错误。已完整读源并核sealed输入／结果标量，未重跑派生、评分或读取QA／预测正文、权重载荷、模型或服务器。

来源闭合：两输入bundle实际SHA与结果账本一致；1%固定proof `ff979a21…`下的38份原aggregate scores逐件SHA／count／路径与19任务前后值一致，未假设旧proof存在verified_cells；10%固定proof `23e5d338…`的74格中对应38个前后值与结果一致。各任务仍为R[t−1,t]与R[t,t]，含T1的R[0,1]；两manifest的完整TEST SHA／count相同。原QA域内加权、五域等权宏及AP19任务等权的源码归约正确，10%刚习得与原proof.acquired复核，不新增score。

核定结论：10%对1%的刚习得绝对水平17高、2低、0平；低项为Quantity与Shape。阶段内变化为16高、3低、0平，不能与上述终点比较混称。Quantity的10%本阶段47.191011→56.179775为正，但仍低于1%刚习得59.550562；它不能被描述为10%当前阶段学习退化。10%唯一负变化为Shape，10.833333→9.722222，原TEST仅12题；小分母和未知原因均须保留，不能扩展成共同知识缺陷。

Qwen10%的Position为+3.425348pp、Plane为+7.447318pp；与已有LLaVA10%对应−4.859471／−0.718391不同。已核LLaVA原派生对应字段：其刚习得19高，不能外推成Qwen也19高，更不能把两项LLaVA下降称跨骨干共同的训练伤害或同一根因。两轨迹总体习得提高但任务不均匀，是成立的描述；其原因尚未识别。

数据量、更新数、调度和此前课程共同改变，Qwen1%／10%还分别在NEW3090／OLD2080Ti训练；没有纯样本量因果结论。相同q/v-r8源码预算不等于已学方向相同。该结果不是新方法、显存优势或conditional三臂结果；不能用它代替三臂对输入条件功能的实际控制，也不据此扩展更多任务审计或增加每域胜出要求。

绑定实际审阅版本：

- 源 `analysis/qwen_task_acquisition_pct01_pct10_20261026.py`：`c27130107ac769d1dd600df10605c8ade609cb27433c806545b4735fbee20058`。
- `QWEN_ACQUISITION.json`：`95d00ef0f63bc1e3a6e8a340ec477ae59e693b4dbd05036dd7ec610d01abd2d8`。
- `QWEN_ACQUISITION.md`：`03759eb064fd55278ac2750601db13de4558980b5c700d23a9368ad01c6b43dc`。
- `QWEN_ACQUISITION_FINDINGS.md`：`50021a4bbfeae409574b7dd8d1b00b0d9e49754f8e2b82ac238fdf2f7a6a4eea`。
- 输入bundle：`f90ffdbcc870d6e70d34ef61ab61acd1b45f0b07b1c05587afa850bbf05532fb`；1%38格bundle：`2a0380fd52a4e5be7573b813ae55a2d418281bf854cc6cfa0fddc261d3d96392`。

此次只有既有标量／摘要身份检查和本审阅写入，0新评分、生成、训练或模型。无需额外门，可发布既定结果。
