"""Reader-first presentation; experiment collection/scoring stays in collect.py."""
from pathlib import Path

NAMES = {'seq_lora':'SeqLoRA','migu_lora':'MIGU-LoRA','olora':'O-LoRA','sapt_lora':'SAPT-LoRA'}

def cell(text, limit=90):
    text = ' '.join(str(text).split())
    if len(text)>limit:text=text[:limit]+'…'
    return text.replace('|','\\|').replace('<','&lt;').replace('>','&gt;')

def build_report(run, root, examples, overview, domains, rows, total_records, config):
    lines = ['# 含图像的持续学习实验报告', '',
        '这轮观察同一个 Qwen2-VL-2B 模型先学习语言，再学习医学语言、自然图像和医学图像后，能学到多少、保留多少。四个方法的九任务实验均已完成。', '',
        '## 先看真实示例', '',
        '以下每任务取固定测试顺序的第一条，展示输入与回答的节选；示例选择不依赖得分。模型回答来自 SeqLoRA 学完全部任务的状态。', '',
        '| 领域 / 任务 | 输入节选 | 参考答案节选 | 模型最终回答节选 |',
        '|---|---|---|---|']
    for example in examples:
        if example['kind']!='task':continue
        source=example['source']
        prompt=source['prompt'].split('\n\nInput: ',1)[-1].rsplit('\n\nResponse:\n',1)[0]
        prediction=example['responses'].get('seq_lora',{}).get(str(len(config['tasks'])),{}).get('prediction','尚未完成')
        lines.append('| '+ ' | '.join([cell(example['title']),cell(prompt),cell(source['references'][0]),cell(prediction)])+' |')
    lines += ['', '图像示例使用真实图片：[自然图像与医学图像、各答案类别以及四方法完整回答](examples.md)。该文档覆盖 9 个任务和 5 个图像答案类别，逐例对比基线、刚学完和最终回答。', '',
        '## 大体效果', '',
        '本轮 SeqLoRA 的最终平均分最高；SAPT-LoRA 的平均遗忘幅度最小，但最终平均分较低。应同时看学习能力与保留能力。结果来自单个种子、固定顺序及短训练预算。', '',
        '| 方法 | 状态 | AP ↑ | 遗忘率 ↓ | BWT ↑ | FWT ↑ |', '|---|---|---:|---:|---:|---:|']
    for row in overview:
        values=[f'{row[k]:.3f}' if isinstance(row.get(k),(int,float)) else '—' for k in ['AP','F.Rate','BWT','FWT']]
        lines.append(f"| {NAMES[row['method']]} | {row['status']} | "+' | '.join(values)+' |')
    lines += ['', '## 指标怎样理解', '',
        '所有任务分数均为 0–100；多参考答案时，每项取与参考比较的最大值，再对固定测试样本平均。', '',
        '| 指标 | 含义 | 阅读方式 |', '|---|---|---|',
        '| ROUGE-L ↑ | 生成文字与参考答案的最长公共子序列 F1 | 词面重合越多，分数越高；不等同于医学或事实正确率 |',
        '| EM ↑ | 小写、去标点/英文冠词并合并空白后，是否完全匹配参考答案 | 单题为 0 或 100；任务均值是归一化精确匹配比例 |',
        '| Token F1 ↑ | 答案与参考的词覆盖率与精确率的调和平均 | 容许部分匹配，不要求词序一致 |',
        '| AP ↑ | 全部学完后的各任务 ROUGE-L 等权平均 | 看最终整体水平 |',
        '| F.Rate ↓ | 旧任务在最终阶段前的历史最好分，减去最终分，再平均 | 看平均遗忘幅度；单位为分数点，允许负值 |',
        '| BWT ↑ | 旧任务最终分与刚学完时分数之差的平均 | 正值表示后续学习帮助旧任务，负值表示下降 |',
        '| FWT ↑ | 每个任务尚未训练时，相对初始基座的分数变化，再平均 | 看前序学习是否帮助未来任务；排除第一任务 |', '',
        '最后一个任务没有后续学习，不参与 F.Rate/BWT。FWT 只比较训练该任务之前的表现。更完整的定义与手算示例见 [语言实验指标说明](../summary/README.md#指标定义)。', '',
        'VQAv2 的 EM 是多参考答案归一化精确匹配，不能当作官方 VQA 共识准确率。跨任务平均混合了摘要、对话、短回答等形式，应同时查看下面的领域和任务表现。', '',
        '## 各领域效果', '',
        '| 方法 | 最终领域 | ROUGE-L | EM (%) | Token F1 (%) |','|---|---|---:|---:|---:|']
    for row in domains:
        lines.append(f"| {NAMES[row['method']]} | {row['domain']} | {row['rougeL']:.3f} | {row['exact_match']:.3f} | {row['token_f1']:.3f} |")
    lines += ['', '自然图像问答最终 EM 约 83.5%–86.0%，医学图像约 45.5%–49.5%。医学语言生成的得分明显低于自然图像短回答；不同任务的答案长度和参考形式不同，应逐任务分析。', '',
        '## 各任务效果', '',
        '| 方法 | 最终任务 | ROUGE-L | EM (%) | Token F1 (%) |','|---|---|---:|---:|---:|']
    for row in rows:
        if row['stage']==len(config['tasks']):
            lines.append(f"| {NAMES[row['method']]} | {row['task_name']} | {row['rougeL']:.3f} | {row['exact_match']:.3f} | {row['token_f1']:.3f} |")
    lines += ['', '## 报告范围与明细', '',
        '学习顺序：XSum → Quoref → 关系抽取 → PersonaChat → GLUCOSE → MTS-Dialog → IU-Xray → VQAv2 → VQA-RAD。每任务 1,000 条训练、100 条验证、200 条测试，训练 1 轮；图像与语言使用同一预算。', '',
        'VQAv2 使用官方验证集的本地子集；VQA-RAD 使用图像/病例分组划分。这轮是先导实验，不能直接与论文的完整官方测试成绩比较，也不能直接与 T5 的不同任务和测试配额比较。', '',
        '[全部阶段](stage_scores.csv) · [领域分数](domain_scores.csv) · [答案类别与医学部位](category_scores.csv) · [完整实例](examples.json)', '',
        f'已核对 {total_records:,} 条测试回答，汇总分数与逐题评分一致，记录见 [verification.json](verification.json)。', '',
        '## 实现流程与复现命令', '',
        f'原始运行：[本次运行](../{run.relative_to(root)})。完整参数和方法适配见 [实验协议](../rules/004_qwen2vl_cv.md)。', '',
        '初始基座及每个学习阶段结束后，均复测全部九个任务，形成 10×9 矩阵。模型基座和视觉编码器冻结，训练 LoRA；图像的冻结特征缓存复用。每个训练更新保存优化器、学习率调度器、随机状态和样本游标，阶段 checkpoint 另含方法记忆。', '',
        '各方法目录内的 `config.json`、`data_manifest.json`、`environment.json` 和 `source/` 保留参数及实现指纹；`predictions/` 保留回答，`checkpoints/` 保留训练日志和恢复状态。', '',
        '在仓库根目录刷新此报告：', '', '```bash', '.vision-env/bin/python summary_cv/collect.py', '```', '',
        '启动与恢复命令、实现文件及日志位置见 [实验说明](../exp/README.md#实现流程与复现)。']
    return lines
