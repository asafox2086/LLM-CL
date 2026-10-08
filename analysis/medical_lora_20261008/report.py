"""Reader-first medical diagnostics with direct data citations and Chinese captions."""
import csv,json
from pathlib import Path
from statistics import mean
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
METHODS=['seq_lora','migu_lora','olora','sapt_lora'];NAMES=dict(zip(METHODS,['SeqLoRA','MIGU-LoRA','O-LoRA','SAPT-LoRA']))
TASKS=json.loads((ROOT/'summary_cv/process_data.json').read_text())['task_order']
LABELS=['XSum','Quoref','Relation','Dialogue','Cause','MTS','IU-Xray','VQAv2','VQA-RAD']
def rows(name):return list(csv.DictReader((OUT/name).open()))
def doc(name,lines):(OUT/name).write_text('\n'.join(lines).rstrip()+'\n')
def f(x):return f'{float(x):.3f}'
def find(rs,**conditions):return next(r for r in rs if all(str(r[k])==str(v) for k,v in conditions.items()))
a=rows('answer_statistics.csv');prior=rows('constant_answer_baselines.csv');route=rows('routing_statistics.csv');groups=rows('medical_vqa_groups.csv');cases=json.loads((OUT/'cases.json').read_text())
color='蓝色圆点实线为 SeqLoRA，绿色菱形虚线为 MIGU-LoRA，红色三角点划线为 O-LoRA，紫色方块点线为 SAPT-LoRA。'
intro=['# 医学任务为什么学不好，又为什么扰动其他能力？','',
 '分析对象是已完成的 Qwen2-VL-2B 九任务运行。这里新增的是诊断，不改写原实验、参数或评分；重用 72,000 条回答，并补做冻结 checkpoint 的输入与路由干预、梯度检查。','',
 '## 先看两个具体失败','',
 '- 临床记录参考包含导尿、尿液白细胞、导尿后少量血、腹部 X 线与患儿活动情况；SeqLoRA 学完该任务却只回答 **“Normal.”**。这属于细节遗漏，不能简单解释成换一种说法。',
 '- 影像 Impression 参考写“心脏增大、左肺底瘢痕/肺不张”；模型回答 **“No acute cardiopulmonary abnormality.”**，省略了慢性发现。此处“无急性异常”与慢性心脏增大未必矛盾，问题是信息覆盖不足。','',
 '[11 个诊断实例与医学原图](01_data_and_cases.md) 同时保留真正的参考不一致和 “1 / One”“Right / Right hemisphere” 这类词面假失败。实例特意按失败/遗漏选择，用于解释机制，不能代表发生率。','',
 '## 先修正对成绩的理解','',
 '![医学任务学前、刚学完与最终表现](figures/medical_learning_chain.png)','',
 '**图 1｜三种医学任务的学习与保留。** 横轴依次为基座、该任务学前、刚学完、最终；括号内标实际阶段。纵轴是 ROUGE-L（0–100）；'+color+'三个子图分别是临床记录、文字 Findings→Impression、医学看图问答。医学 VQA 的“刚学完”和“最终”同为阶段 9。只在同一子图比较收益，不能把 ROUGE-L 当医学正确率。','',
 '这轮的共性更接近：**共享且受限的适配通道更容易学到短回答、正常模板和答案先验，医学任务所需的病人细节、关系、否定与视觉证据没有同等地被保留。限制更新能减少旧能力变化，也可能让新医学能力几乎不进入模型。** 这是针对本次实现和预算的解释，不是“LoRA 天生不能做医学”的结论。','',
 '| 发现 | 直接证据 | 判断强度 |','|---|---|---|',
 '| 医学报告的 ROUGE 提升大部分可由正常模板解释 | 固定训练集最常见回答的 ROUGE-L=39.567；SeqLoRA 最终=40.438，58% 输出该句 | 已测到模板捷径；没有临床事实评分，不能说全部提升都是捷径 |',
 '| 临床记录过度压缩 | 参考平均38.38词；SeqLoRA刚学完15.76词、最终8.76词 | 已测到长度与遗漏；不能仅凭长度判断临床正确性 |',
 '| SAPT新任务门控几乎关闭 | 医学任务专属 adapter 平均权重约0.002%–0.011%；强制路由可降低医学文字参考的CE | 有固定权重干预；图像强制路由反而更差，说明不只是推理时选错专家 |',
 '| O-LoRA保护约束很强 | 医学阶段正则梯度范数为答案CE梯度的12.3–25.4倍 | 已测到梯度竞争；没有重训消融，尚不能归因全部性能差距 |',
 '| MIGU掩码删除多数答案梯度能量 | 医学小样本保留35.4%–39.0%的梯度能量 | 已测到筛除；与通用问答接近，并非已证明医学特异性伤害 |',
 '| 视觉细节预算与证据利用不足 | 128→512 token补测改善；同部位换图未降低平均分 | 小样本支持瓶颈；最终模型分辨率收益的区间跨0，不能声称稳定显著提升 |',
 '| 某些更新确实影响其他任务，但不是全局一致下降 | 已学五语言任务、外部通用与医学题的阶段变化不同 | 只能说覆盖范围内有选择性扰动，不能推断全局知识被擦除 |','',
 '## 每份分析报告说明什么','',
 '| 报告 | 回答的问题 |','|---|---|',
 '| [数据、评分与实例](01_data_and_cases.md) | 差成绩里哪些是真遗漏，哪些是评分和答案分布造成的？ |',
 '| [LoRA模块机制](02_lora_mechanisms.md) | 共享参数、梯度掩码、正则和路由怎样导致扰动与学习不足？ |',
 '| [原图与视觉干预](03_vision.md) | 模型到底用没用图片？更高视觉token预算是否改善？ |',
 '| [跨任务与外部知识](04_global_effects.md) | 哪一步伤害哪些已测能力？医学是否普遍有害？ |',
 '| [下一轮对照](05_next_experiments.md) | 按证据优先级，如何验证和修复，而不是只增加训练量？ |','',
 '## 与论文的联系','',
 '医学任务异质监督与跨数据集干扰是 [MedQwen](https://arxiv.org/abs/2604.01310) 的研究动机；它采用专家分工与稳定路由。我们的路由实测提供了本地对应现象，但没有复现其模型或证明同样方法一定有效。',
 '[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) 在编程和数学域展示可塑性与保留的权衡，提示不能只以“遗忘少”判好；其结论不能直接证明本轮医学 rank=8 已饱和。',
 '影像报告应区分词面与事实；[RadGraph 语义奖励研究](https://aclanthology.org/2022.findings-emnlp.319/) 提供事实关系评估思路。本次没有运行 RadGraph/CheXbert，未把词面分冒充临床指标。','',
 '## 参数、原始记录与复现','',
 '模型2.209B，语言侧 q_proj/v_proj LoRA；单adapter rank8、alpha32、dropout0.1；基座、视觉编码器与merger冻结。每任务1000/100/200样本、一轮、有效batch16、lr1e-4，共63次更新；视觉输入上限128个合并token，文字prompt768、target256。医学图像训练涉及186张唯一图片，测试200问题对应40张图片。完整协议见 [原实验](../../rules/004_qwen2vl_cv.md)。','',
 '[逐文件SHA与范围](provenance.json)、各诊断protocol、CSV、完整回答JSON、原图副本及PNG/PDF均在此目录。所有GPU诊断以nohup/独立会话启动；没有optimizer更新。高分辨率batch4首次显存不足，改用batch1完成，失败日志保留。','',
 '```bash','.vision-env/bin/python analysis/medical_lora_20261008/audit.py','.vision-env/bin/python analysis/medical_lora_20261008/cases.py','.vision-env/bin/python analysis/medical_lora_20261008/calibration.py',
 '# 以下每条可使用独立空闲GPU；nohup setsid保证断开终端后继续运行',
 'CUDA_VISIBLE_DEVICES=0 nohup setsid .vision-env/bin/python -u analysis/medical_lora_20261008/routing.py > analysis/medical_lora_20261008/routing.log 2>&1 < /dev/null &',
 '# gradients.py、masks.py、interventions.py、vision_ablation.py用相同方式启动',
 '.plot-env/bin/python analysis/medical_lora_20261008/plot.py','.vision-env/bin/python analysis/medical_lora_20261008/report.py','```']
doc('README.md',intro)
# Data and selected instances.
lines=['# 数据、评分和医学实例','',
 '本报告区分任务确实没完成、过度压缩和词面评分误伤，避免把所有低分解释成医学知识不足。以下实例为诊断选择；完整平均数来自200题，不由几个实例推断。','',
 '## 正常模板基线有多强','', '| 任务 | 训练集中最常见答案 | 训练频率 | 恒定回答测试ROUGE-L | 恒定回答测试EM |','|---|---|---:|---:|---:|']
for r in prior:lines.append(f"| {r['task']} | {r['train_most_frequent_answer']} | {float(r['train_frequency'])*100:.1f}% | {f(r['test_ROUGE_L'])} | {f(r['test_EM'])} |")
lines += ['', 'Impression 任务中，该模板仅占训练目标的8.7%，却占 SeqLoRA 最终预测的58%。恒定回答已达39.567，SeqLoRA最终40.438、刚学完42.609。说明模型强烈偏向常见“无急性异常”句式；不能把相对基座的30余点增长全部解释成新增临床事实。该任务输入是 Findings 文字，没有读取原始影像。','',
 '## 短答题与长记录不能只比同一个指标','',
 'MTS参考平均38.375个英文词；SeqLoRA阶段0/6/7/9平均输出69.975/15.755/6.950/8.755词。参考很长而预测过短，会漏掉病史、时间、部位、检查和否定关系；反过来，基座长篇复述也不等于正确。测试仅1/200输入截断，5/200目标超过255 token；训练分别13/1000与36/1000。因此截断是局部问题，不能解释所有MTS低分。所有任务均按每条答案CE均值再按样本均值，长答案不会自然获得更多样本权重。','',
 '医学 VQA 有136条CLOSED和64条OPEN；SeqLoRA最终分别63.97%和18.75%。加入有限显式等价（One↔1、Right hemisphere↔Right、PA全称）后，OPEN升到23.44%，仍很低。该校正只用于诊断，没有替换原分数。',
 'VQAv2测试75条yes/no里有26条的十位参考同时含yes与no，max-reference EM两种回答都可命中；所有200题中26条同时含yes/no。SeqLoRA的84.5% EM在同一归一化、留一标注者共识计算下为78.15%。此诊断不是官方VQA成绩，因为没有采用完整官方文本处理。参见 [VQA官方评价](https://visualqa.org/evaluation.html) 与 [校正CSV](score_calibration.csv)。','',
 '## 临床文字：模型具体遗漏了什么','']
for n,c in enumerate([c for c in cases if c['task']!='medical_vqa_rad'],1):
 stage='6' if c['task']=='medical_mts_dialog_note' else '7'
 lines += [f"### {n}. {c['instance_id']}",'','参考：', '', '> '+c['references'][0],'', '| 方法 | 刚学完回答 | 最终回答 |','|---|---|---|']
 for m,rs in c['methods'].items():lines.append('| '+NAMES[m]+' | '+rs[stage]['prediction'].replace('\n',' ').replace('|','\\|')+' | '+rs['9']['prediction'].replace('\n',' ').replace('|','\\|')+' |')
 lines += ['', '这里比较数据集参考中的信息覆盖，不声称从词面评分完成临床正确性裁定。完整输入、全部回答和逐题分数在 [cases.json](cases.json)。','']
lines += ['## 医学原图：真正的参考不一致与词面假失败','']
for n,c in enumerate([c for c in cases if c['task']=='medical_vqa_rad'],1):
 lines += [f"### {n}. {c['organ']} / {c['answer_type']} / {c['instance_id']}",'',f"![医学测试原图 {c['instance_id']}]({c['image']})",'',
 f"**图 {n}｜该问题的原始医学图片。** 按字节复制、未修图；部位标签为数据集的 {c['organ']}。问题：{c['prompt'].splitlines()[-1]}；参考：{'; '.join(c['references'])}。不能凭该图注认定临床诊断，下面只对照数据集参考与模型输出。",'', '| 方法 | 基座回答 | 最终回答 |','|---|---|---|']
 for m,rs in c['methods'].items():lines.append('| '+NAMES[m]+' | '+rs['0']['prediction'].replace('\n',' ').replace('|','\\|')+' | '+rs['9']['prediction'].replace('\n',' ').replace('|','\\|')+' |')
 ident=c['instance_id']
 comment={'vqarad_0296':'SeqLoRA/MIGU/O-LoRA 的 Right 与参考 Right hemisphere 属于本报告显式等价，EM=0不能代表判断错误；SAPT的Left与参考方向不一致。',
 'vqarad_0352':'四方法的1与参考One数量相同，属于词面假失败。',
 'vqarad_0019':'Supine是体位词，与参考Posterior-Anterior成像方向并不等价，体现问题语义与任务属性混淆。'}.get(ident,'四方法最终回答的Yes/No与参考相反，是比句式差异更直接的参考不一致；不据此独立裁定图像病变。')
 lines += ['',comment,'']
lines += ['## 数据依据与重算','', '[答案分布](answer_statistics.csv) · [数据长度/截断](dataset_audit.csv) · [分组成绩](medical_vqa_groups.csv) · [等价校正](calibration_protocol.json) · [校正救回的实例](alias_rescued_cases.json)。原图和病例从原测试集选择，没有拿来训练。','', '```bash','.vision-env/bin/python analysis/medical_lora_20261008/cases.py','.vision-env/bin/python analysis/medical_lora_20261008/calibration.py','```']
doc('01_data_and_cases.md',lines)
# Mechanistic evidence and its limits.
regs=rows('regularization_gradients.csv');mask=rows('migu_gradient_masks.csv');overrides=rows('routing_override_CE.csv')
lines=['# LoRA模块的共性与方法差异','',
 '共性是所有任务依赖同一语言主干中的注意力Q/V适配路径；医学适配既要改变注意力与输出，又不能扰动旧任务。四方法对这两件事的处理不同，不能概括成同一个代码bug。','',
 '## 先看最明确的路由问题','', '![SAPT最终真实路由权重](figures/sapt_routing.png)','',
 '**图 1｜SAPT 阶段9的实际路由。** 横轴是九个任务的adapter，纵轴是被评测任务；每格为该任务200个测试输入的平均路由权重（%，一行合计100%），色条范围0–100。颜色越深表示使用该adapter越多。XSum=摘要、Quoref=阅读理解、Relation=关系抽取、Dialogue=对话、Cause=因果、MTS=临床记录、IU-Xray=影像报告文字、VQAv2=自然图像、VQA-RAD=医学图像。本图由checkpoint内部router重新计算，是路由热图，不是成绩矩阵。绝大多数输入集中到第一任务adapter，医学专属adapter几乎关闭。','',
 '| 医学任务 | 刚学完时专属adapter平均权重 | 最终专属adapter平均权重 |','|---|---:|---:|']
for task,stage in [('medical_mts_dialog_note',6),('medical_iu_xray_impression',7),('medical_vqa_rad',9)]:
 now=find(route,task=task,stage=stage,split='test');last=find(route,task=task,stage=9,split='test')
 lines.append(f"| {task} | {float(now['task_adapter_weight'])*100:.6f}% | {float(last['task_adapter_weight'])*100:.6f}% |")
lines += ['', 'SAPT共享router使用冻结融合输入的逐维max池化，缺少显式task-ID训练监督；历史KL目标在新任务维度补零。实现允许新专家被低概率选中。历史保护、共享提示和非上下文化池化一起形成“老专家占优—新专家梯度变小—更难获得使用率”的可能反馈链。权重塌缩已实测；是哪一项最先导致塌缩，仍需改变池化/初始化/KL的重训消融。','',
 '固定checkpoint，只把门控设为真实任务对应adapter（oracle task-ID），得到以下参考答案teacher-forced CE：','',
 '| 阶段 | 任务 | 学到的路由CE | 强制专属adapter CE |','|---|---|---:|---:|']
for r in overrides:
 if r['mode']=='learned_routing':
  forced=find(overrides,stage=r['stage'],task=r['task'],mode='oracle_current_task_adapter')
  lines.append(f"| {r['stage']} | {r['task']} | {f(r['teacher_forced_CE'])} | {f(forced['teacher_forced_CE'])} |")
lines += ['', '每任务固定16题，无参数更新。医学文字CE变低，支持选错专家确实限制使用已学文字适配；医学图像CE由1.107升至2.143，说明其专属专家本身也未学好。强制路由不是可部署方法，CE降低也不等于生成准确率一定提高。','',
 '## 为什么共享模块会扰动别的任务','',
 '$$',r'\Delta W=\frac{\alpha}{r}BA,\qquad h\mapsto Wh+\Delta Wh,','$$','',
 '$$',r'\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt d}\right)V.','$$','',
 'Q的更新改变token之间的注意力分配，V的更新改变被聚合的内容；作用于所有走过该层的输入，而不只医学输入。冻结基座只表示原参数不变，输出函数仍随adapter改变。小权重变化也可使原本接近的答案logit换序，导致greedy答案跳变。','',
 '医学任务特别依赖〈实体、部位/侧别、属性、时间、否定/不确定性〉关系。“有/无”“左/右”“急性/慢性”可能只差很少token，但语义完全不同。当前普通答案CE、ROUGE选模和Q/V路径没有显式保护这些关系，短答案或常见正常句式更容易获得稳定损失下降。该解释由实例和模板统计支持，但没有实体级标注来证明每种关系的发生率。','',
 '![任务答案梯度夹角](figures/gradient_conflict.png)','',
 '**图 2｜SeqLoRA在阶段5、医学训练前的局部梯度关系。** 横纵轴为同样的九任务，每格是各任务固定8个测试样本平均答案CE梯度的余弦。红色负值=在小步梯度下降近似下，改善一个任务可能损害另一个；蓝色正值=局部方向较一致，白色≈0=接近正交。对角线为1，色条固定−1到1。梯度只来自共享adapter；dropout关闭、没有optimizer更新。小样本局部关系不能直接预测完整63次AdamW更新。','',
 'MTS与旧关系抽取梯度余弦为−0.151；医学报告与对话为−0.060。也有正方向，例如MTS与因果任务+0.111。实测并不是“所有医学梯度都对抗所有旧任务”。阶段5→6在关系抽取上的有限更新CE增加0.0265；阶段8→9在自然VQA上的CE增加0.0984，支持某些共享更新产生扰动。','',
 '一些一阶预测与完整阶段变化相反，例如阶段6→7的MTS：一阶估计−0.0444，实测+0.0399。原因可能包括有限步长、非线性和训练样本不同；不能把一次梯度角度当作整个遗忘过程的充分解释。数据见 [实际更新与梯度投影](update_gradient_projections.csv)。','',
 '## O-LoRA：约束保护很强，但权重正交不等于功能独立','',
 '| 医学阶段 | 答案CE梯度范数 | 正则梯度范数 | 比值 |','|---|---:|---:|---:|']
for r in regs:lines.append(f"| {r['stage']} | {f(r['CE_grad_norm'])} | {f(r['penalty_grad_norm'])} | {f(r['grad_norm_ratio'])}× |")
lines += ['', '在各阶段已学checkpoint上，每任务固定8题，当前adapter可训练、历史adapter冻结；单独对加权正则和答案CE求导。正则梯度大12.3–25.4倍，且二者夹角接近90度，说明主要优化压力存在于答案损失之外。原训练日志的总梯度也频繁被clip到1。','',
 '该实现把各层、所有历史adapter的A矩阵交叉绝对值求和，权重0.5未按历史任务数/层数/矩阵大小归一化，所以历史越多约束项越大。AdamW会按坐标归一化梯度，不能由范数比直接推出医学梯度只剩1/25；需要重训约束强度消融。','',
 'A矩阵正交限制了输入低秩方向，但所有adapter在推理时同时加到同一输出。权重空间Frobenius近正交，不保证在医学/旧任务的各向异性输入上输出影响独立，也没有语义保护“否定、侧别”等字段。原代码见 [O-LoRA](../../code/olora.py)。','',
 '## MIGU：按幅度筛梯度，不按医学事实重要性','',
 '| 梯度诊断任务 | 被保留的梯度能量 | A/down行保留 | B/up行保留 |','|---|---:|---:|---:|']
for r in mask:lines.append(f"| {r['task']} | {float(r['gradient_energy_retained'])*100:.2f}% | {float(r['down_rows_retained'])*100:.2f}% | {float(r['up_rows_retained'])*100:.2f}% |")
lines += ['', '阶段5的MIGU模型、每任务固定8题。真实实现保留量化阈值0.7以上的通道；rank8的down实际剩3/8行，up剩约30%。医疗任务剩35%–39%梯度能量，通用Quoref也只有36.2%，所以已测到普遍筛除，而非证明专门压制医学。','',
 '幅度统计使用prompt+answer所有有效token；up幅度取基座输出与adapter叠加后的投影值，未使用医学实体或答案梯度重要性。弱幅度的罕见医学概念是否被误筛是合理假设，尚没有医学词/通道归因来证实。不能把“幅度大”等同“临床重要”。原代码见 [MIGU](../../code/migu_lora.py)。','',
 '## 四方法真正共享的限制','',
 '| 共同设置 | 对医学的影响路径 | 已证实到哪一步 |','|---|---|---|',
 '| 只在语言Q/V插入rank8当前adapter；视觉编码器/merger/MLP冻结 | 可改变语言注意力和输出，不能直接学习医学视觉特征或任意更高维表征 | 插入位置和冻结范围已核对；rank8是否不足尚未做rank消融 |',
 '| 新医学任务只有1000样本、63次更新 | 稀有事实、不同部位和记录结构的监督不充分 | 配额、图像数、输出统计已测；需要学习曲线证明预算是主因 |',
 '| 普通CE+ROUGE选模，没有临床关系目标 | 常见句式/短答可以进步，患者特定事实未必同步进步 | 恒定模板与长记录遗漏已测；事实级正确性未完整评测 |',
 '| 没有所有方法统一的原始医学证据回放 | 后续任务改变输出行为时，早先长记录没有直接监督锚点 | Seq/MIGU/O无样本回放；SAPT的反思主要约束路由，不等于医学事实回放 |','',
 'Seq/MIGU整体adapter始终rank8；O-LoRA和SAPT累计多个rank8 adapter，不能把它们的累计表达能力也说成rank8。SAPT权重范数报告未乘输入依赖门控，不能把裸adapter矩阵范数当真实有效更新。','',
 '## 数据与可复现计算','',
 '[各层矩阵诊断](adapter_layer_diagnostics.csv) 使用真正的BA矩阵差、薄QR/SVD和Frobenius内积，避免A/B尺度可互换造成假结论。裸权重更新或奇异值集中都不等于临床知识量。这里未计算相对基座W的奇异向量，因此不声称发现论文中的intruder dimensions。',
 '[训练日志汇总](training_diagnostics.csv) · [router逐题权重](routing_items.json) · [CE/正则梯度](regularization_gradients.csv) · [MIGU掩码](migu_gradient_masks.csv) · [诊断选题与参数](intervention_protocol.json)。所有实验是读取checkpoint后的有限诊断，无训练更新。']
doc('02_lora_mechanisms.md',lines)
# Image evidence.
v=rows('vision_ablation_scores.csv');unc=rows('vision_paired_uncertainty.csv')
lines=['# 医学原图、输入细节与看图依赖','',
 '本报告区分“输入细节少”与“模型没有稳健使用该病例图像”。保留原图，检查官方处理器的真实缩放尺寸，并在固定权重下改变图像输入。','',
 '## 原图与训练实际看到的空间尺寸','',
 '![原始医学图与实际缩放对照](figures/original_and_processed.png)','',
 '**图 1｜三个部位、两类问题的输入尺寸对照。** 每行依次为胸部、头部、腹部；前两列是CLOSED问题的原图与训练缩放重建，后两列是OPEN问题的对应图。标题给出像素宽×高与合并视觉token数；问题ID可在 [实例报告](01_data_and_cases.md) 查到。缩放尺寸从锁定image_grid计算，用官方同类双三次重采样重建；它是空间输入对照，不是视觉神经特征、病灶标注或注意力图。原图副本未改动。','',
 '例如1024×1024的胸片缩到308×308，1024×1311的腹部图缩到252×336；黑边和图中文字也进入输入。细小结构、计数、侧别与成像属性不像自然图像的大物体那样容易从整体形状判断。但是并非所有图都损失同样多：263×324的头部图本身分辨率不高，128-token设置几乎没进一步缩小，增加token不等于恢复原图不存在的信息。','',
 '训练视觉编码器与merger始终冻结；所有更新都在语言侧Q/V。已有的通用视觉表征若不能很好编码医学结构，语言LoRA只能重新利用已有特征，不能直接让视觉编码器学会新的解剖/病变表示。这个范围已经代码核对；具体缺了哪一种视觉表征仍需解冻/双侧LoRA对照。','',
 '## 固定题目与权重，改变图像输入','',
 '![四种图像输入条件的成绩](figures/vision_ablation.png)','',
 '**图 2｜60道配对医学问题的图像干预。** 横轴依次是正确图128-token上限、同部位另一图128-token上限、不提供图、正确图512-token上限；纵轴为归一化EM准确率（%，0–100）。灰色是基座阶段0，蓝色是SeqLoRA最终阶段9。每部位×OPEN/CLOSED固定10题，共60题、30张源图片；不同条件严格使用同一题目/参考和同一checkpoint，未训练。数字是这组分层小样本成绩，不能与全200题成绩直接比较。128/512是设置上限，实际token由图像宽高确定。','',
 '| 权重 | 条件 | 全60题EM | CLOSED 30题 | OPEN 30题 |','|---|---|---:|---:|---:|']
for stage in [0,9]:
 for mode in ['actual128','wrong128','no_image','actual512']:
  vals=[find(v,stage=stage,mode=mode,group=group)['EM'] for group in ['all','CLOSED','OPEN']]
  lines.append('| '+('基座' if stage==0 else 'SeqLoRA最终')+' | '+mode+' | '+' | '.join(f(x) for x in vals)+' |')
lines += ['', '最终模型提高视觉上限后，60题EM由38.33%变45.00%，但OPEN由20.00%降至16.67%；提升主要发生在CLOSED。去图仍达35.00%，正确图的净优势只有3.33点；同部位换图45.00%，没有出现预期整体下降。单个回答仍会随图像改变，因此这不证明模型完全忽略图片。它更支持：当前正确率受答案先验和病例视觉证据混合影响，正确图像的可靠优势尚未建立。换图同部位可能保留粗粒度解剖线索，小样本与答案偏置也可让换图偶然改善。','',
 '原实验使用单图缓存特征；补测使用stock视觉编码器按batch计算。正确图128条件在基座与最终均有59/60回答与原实验逐字相同，1题因计算路径/batch差异改变；比较采用补测自身的正确图条件，避免把原分数硬当干预对照。','',
 '## 小样本不确定性','', '| 阶段 | 条件相对正确图128 | EM差（点） | 按图片成组bootstrap 95%区间 |','|---|---|---:|---|']
for r in unc:lines.append(f"| {r['stage']} | {r['comparison']} | {f(r['delta_EM_points'])} | [{f(r['cluster_bootstrap_95_low'])}, {f(r['cluster_bootstrap_95_high'])}] |")
lines += ['', '固定seed42、2000次重采样，保留同一图片的题目相关性。最终模型512−128区间跨0，不能宣称可靠显著收益；这也不是多训练种子的不确定性。试验只是当前60题的探索诊断，更换题库或图片可能得到不同结果。','',
 '## 数据、处理器和日志','',
 '[480条完整生成记录](vision_ablation_protocol.json) 的清单和图片替换映射固定；逐条文件为 `vision_stage{0,9}_{actual128,wrong128,no_image,actual512}.jsonl`，含问题、原/使用图片、预测、参考、生成token和评分。',
 '[全训练/测试图像尺寸](image_resolution.csv) · [全部分组成绩](vision_ablation_scores.csv) · [配对区间](vision_paired_uncertainty.csv)。高分辨率只改推理输入，没有重训；它不能直接回答更高分辨率训练是否改善。',
 '保留首次高分辨率batch4显存不足日志，随后batch1完成；其他条件batch4。原实验checkpoint及共享图像缓存均未改写。']
doc('03_vision.md',lines)
# Cross-task observations and external probes.
probe=list(csv.DictReader((ROOT/'summary_cv/knowledge_probe_results.csv').open()))
process=json.loads((ROOT/'summary_cv/process_data.json').read_text())
lines=['# 医学训练对其他能力究竟有何影响？','',
 '本报告只判断已有测试覆盖范围内的变化。九任务与198道外部题都不是全部知识，必须定位阶段、任务和评分口径，不能把最终负FWT或一条下降曲线当作全局知识被擦除。','',
 '## 医学更新前后，旧任务是不是都下降？','',
 '| 方法 | 医学学习阶段 | 固定旧任务数 | 旧任务ROUGE-L平均变化 | 外部通用准确率变化（点） | 外部医学准确率变化（点） |','|---|---|---:|---:|---:|---:|']
for m in METHODS:
 for stage in [6,7,9]:
  stats=process['methods'][m]['stage_statistics'][stage]
  prev=find(probe,method=m,stage=stage-1);now=find(probe,method=m,stage=stage)
  lines.append(f"| {NAMES[m]} | {stage} / {LABELS[stage-1]} | {stage-1} | {f(stats['old_fixed_shock'])} | {float(now['general_accuracy'])-float(prev['general_accuracy']):+.3f} | {float(now['medical_accuracy'])-float(prev['medical_accuracy']):+.3f} |")
lines += ['', '每行在同一旧任务集合上比较本次医学更新前后；不同阶段的旧任务集合不同。外部通用/医学题集合始终固定为102/96题。部分行旧任务或知识探针提高，部分下降，二者也不总同向。因此“所有医学训练都伤害全局”的说法不受当前数据支持。','',
 '## 非医学任务也会伤害医学能力','',
 '| 方法 | MTS：刚学完阶段6 | MTS：报告学习后阶段7 | 医学VQA：自然图像前阶段7 | 医学VQA：自然图像后阶段8 |','|---|---:|---:|---:|---:|']
for m in METHODS:
 vals=[find(a,method=m,stage=s,task=t)['rougeL'] for t,s in [('medical_mts_dialog_note',6),('medical_mts_dialog_note',7),('medical_vqa_rad',7),('medical_vqa_rad',8)]]
 lines.append('| '+NAMES[m]+' | '+' | '.join(f(x) for x in vals)+' |')
lines += ['', '长临床记录在学习短Impression后被进一步压缩；自然图像问答阶段也使多个方法的未来医学VQA分数回落。这更符合“输出行为与证据选择在任务切换时重分配”的解释，并非某一种医学内容具有天然破坏性。分数变化包含格式、词面与实际参考一致性的混合。','',
 '## 学前、学后、最终与外部题应分别读','',
 '外部医学知识题最终相对基座：SeqLoRA+3.125、MIGU+2.083、O-LoRA+0、SAPT+1.042点；外部通用题为+0.980、−2.941、−1.961、−2.941点。一个或几个题即可造成这些变化，不能宣称医学知识普遍提高或通用知识明显擦除；已有基座正确题中有丢失，也有原错题被答对。详细保留计数见 [外部知识报告](../../summary_cv/knowledge_probe.md)。','',
 '短任务得分、长记录信息覆盖、临床图像证据利用和外部知识题是不同维度。模型可能学会“回答更短”而没有学会新诊断规则；也可能保留旧平均分却几乎不使用新adapter。因此要一起报告直接收益、超基座收益、常数模板基线、旧任务变化、临床事实和图像依赖。','',
 '## 解释限度与数据依据','',
 '跨任务变化为观测；SeqLoRA局部梯度与有限更新诊断提供可能路径，固定权重图像/路由干预只验证相应输入和门控的作用。没有单任务医学从基座训练、调低正则重训或其他顺序，不能分离顺序、任务预算、模型大小与算法的全部原因。',
 '[原全过程与矩阵](../../summary_cv/process.md) · [阶段平均/差分](../../summary_cv/process_metrics.csv) · [旧正确题回归的完整文本](old_answer_regressions.json) · [局部CE与更新](gradient_CE.csv)。']
doc('04_global_effects.md',lines)
lines=['# 下一轮先验证什么？','',
 '这里是根据诊断提出的对照清单，尚未执行新训练。优先修已测到的瓶颈，再检验容量假设；每次改变一个因素，统一保留学前/学后/最终、旧任务和固定外部探针。','',
 '## 优先级与可否证预期','',
 '| 优先级 | 对照 | 预期与判据 |','|---|---|---|',
 '| 0：把收益测对 | 长记录的事实覆盖/否定/部位；Impression的恒定模板基线；VQA显式等价与闭/开放分组 | 如果只提升模板ROUGE而不提升事实、或换图仍不降，不能称为临床能力学习 |',
 '| 1：SAPT医学专家是否获得训练与使用 | 强制当前adapter训练的诊断对照；router warm-up / task监督 / load balance；逐项改变历史KL和max池化 | 应看到新专家使用率、实际adapter梯度和独立任务得分上升；只推理时强制门控已不足以救回医学图像 |',
 '| 2：O-LoRA约束是否压过学习 | 维持同一数据/学习率，降低正则0.5→0.05→0.005，或按层数×历史任务数归一化；监测CE和正则梯度 | 新任务收益能否提高、旧任务是否恶化；不能只看总loss下降或正则值变小 |',
 '| 3：MIGU筛选是否损失新域信号 | 对同一医学阶段比较不掩码、现幅度掩码与仅答案token/答案梯度重要性掩码 | 医学事实覆盖、收益/保留和实际保留梯度能量共同衡量，不能把通用掩码损失都称医学特异性 |',
 '| 4：视觉输入和视觉域适配 | 128/512/1024视觉token预算；同预算LM-only对比merger/vision末层LoRA；采用医学结构保持的裁剪 | 正确图优势与开放题得分是否增强，而非只提高yes/no先验；裁剪需检查是否删掉诊断相关全局信息 |',
 '| 5：低秩容量还是目标问题 | rank8/16/32；扩展目标至o_proj/MLP与现Q/V对照，控制参数预算 | 若收益随着容量提高且旧能力不恶化，才支持容量不足；保持alpha/r=4，避免把rank和缩放强度混淆 |',
 '| 6：医学特性还是顺序/格式 | 从基座单独训MTS、IU、VQA；医学任务前置与当前顺序；长记录与短Impression分别路由/回放；至少多种子 | 单任务也差偏向数据/表征/目标；只有连续学习差偏向干扰；长短格式隔离改善则支持输出行为竞争 |','',
 '## 为什么从医学特性设计这些对照','',
 '- 临床记录的单位是患者事实关系，而不只是几个关键词。应跟踪症状/检查/时间/否定，保留长答案监督，不让单句常见结论替代整段信息。',
 '- 医学“正常/无急性异常”句式常见，稀有阳性和慢性发现更容易被遗漏。应报告这些子集与模板基线，不只看平均ROUGE。',
 '- 医学视觉依赖部位、尺度、侧别、投影与微小结构。应测图像配对依赖，保留全局视图及有依据的局部视图；先确认信号进入模型，再增加LoRA容量。',
 '- 任务提示共享，但实体关系和监督形态不同。路由应识别临床记录、短报告与图像问题的需求，而不是所有输入归入早期摘要专家。','',
 '## 后续实验如何公平比较','',
 '保留原测试报告作为探索诊断；后续调参与选模只用dev，并用新的或预留未参与设计的测试集做确认。方法独立保存参数/分组、逐题预测与图像身份。医学图像按病例分组报告置信区间；同一图片多问题不能按独立病例计数。临床事实自动指标需核对领域适用性，并抽查模型事实与否定，不直接将ROUGE、医学VQA和MMLU放进一个“医学正确率”。',
 '每项都报告新任务学前→学后、相对基座净收益、后续保留、旧任务变化、外部知识变化及运行成本。SAPT专家数、O-LoRA累计rank和参数量增长需要单列。失败/中断恢复日志也保留，不替换既有主实验。']
doc('05_next_experiments.md',lines)
print('Six reader-first analysis reports generated.')
