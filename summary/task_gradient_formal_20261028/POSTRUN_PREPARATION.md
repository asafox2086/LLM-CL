# 正式结果的功能与资源分析：已准备，尚未执行

正式TaskGradReuse仍运行原冻结1%完整协议。本轮仅准备独立后处理，不修改29个训练源、协议、数据、评价范围或原评分；没有产生新的模型质量或部署资源结论。

## 要回答的问题

用本候选自己的完整最终F1、刚习得分数和实际完整图成本，判断它相对普通Seq及强PAS/OSFT保住了多少学习收益。普通Seq是必要低成本参照；仅比昂贵方法小不足以证明方向策略有价值。不同方法的轨迹还改变更新位置、学习率和结构，不能将差异单独归因为容量或方向。

`analysis/task_gradient_acquisition_20261028.py`只消费成功正式核验后的封存元数据与已有聚合分数，对照三份实际sealed proof。它读取76份各任务刚习得聚合，不读QA、预测或checkpoint，不重新评分。报告19任务及五域的刚习得和最终差值，不展开遗忘分支。原主表AP19是`mean R[19,t]`；另报的刚习得19任务均分是`mean R[t,t]`，二者不混用。

## 独立资源测量

`exp/task_gradient_final_resource_20261028/worker.py`和`verify.py`复用原五窗测量，只能在正式成功、唯一质量核验完成、研究slot释放且原进程消失后运行。当前未运输、排队或执行，最终checkpoint SHA及资源binding尚未生成。

原生完整S19 checkpoint独立GPU重载，保留实际payload，不再次合并、不转换dtype。检查48个q/v均为最终dense Linear，其权重摘要与训练端最终fold记录相同；核对无A/B、专家bank、G或TRAIN上下文，并审计模型外仍可达CUDA存储及未知模型内张量。完整系数、原dtype权重字节和加载峰均实测，不用checkpoint容器大小替代。

沿原PAS同一物理GPU1、同五个raw/prepared输入、每窗32新token测量allocated/reserved峰、TTFT及解码速度。原配置、公开generation配置、权重和buffers前后不变。五窗不是完整TEST最坏峰或生产吞吐；PAS原参照仅有内存，不能据此宣称对PAS的速度优势。

OSFT已有标准dense导出记录，应分别报告原保存图和允许合并后的成本。不能用其未合并的2757.984MiB冒称本方法独占减重；GPU函数等价与实际峰仍须对应证据。PAS原完整图的资源差异也只覆盖已测实现和dtype。

## 源码与审阅

| 独立新源 | SHA256 |
| --- | --- |
| acquisition consumer | 62f103f4398272c8e8fdb5d59046334b37396b4d651a65f62d3f472077e51ee8 |
| resource worker | b3f4fc9630c1622d05829f5b32eb5fb9d0fce7d98b8101b9026cbea7a3979ca9 |
| resource verifier | 59db4dac83025fc71d5ca94861a373c8eedbd2b2e8961f7bfede18f2a56c6fb7 |

plain_r8_control实现资源两源；research_review独立只读核接口、最终fold摘要、隐藏CUDA引用、五窗输入与口径，未见确定阻断；root交叉核源码并通过AST检查。consumer亦已核0–100分单位和任务QA分母。以上是源码审阅，不是GPU实测或运行通过。

为继续保留正式阶段checkpoint，已按用户允许的早期CP清理规则，仅删除OLD PaRSP S0–S3四个重复payload，共18165052238字节。NEW归档四份和OLD四份均重新完整SHA一致后才删除；NEW全部历史模型、OLD当前S4恢复点、所有答案/分数/配置/证明JSON和恢复许可不变。原collector会将这四个OLD缺失payload列入`source_deferred`，不代表NEW归档失败。实际凭据另存`EARLY_CHECKPOINT_RETENTION_20261028.json`；不重放一次性清理脚本。
