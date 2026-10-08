"""Reconstruct all compatible paper metrics from per-stage task scores."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'exp'))
from common import continual_metrics
METHODS=['seq_lora','migu_lora','olora','sapt_lora']
LABELS={'seq_lora':'SeqLoRA','migu_lora':'MIGU-LoRA','olora':'O-LoRA','sapt_lora':'SAPT-LoRA'}

def statistics(matrix):
    T=len(matrix[0]);assert len(matrix)==T+1
    base=sum(matrix[0])/T
    diagonal=sum(matrix[j+1][j] for j in range(T))/T
    final=sum(matrix[-1])/T
    maa=sum(sum(matrix[s][:s])/s for s in range(1,T+1))/T
    return {'baseline_fixed_mean':base,'MFT':diagonal,'MFN':final,'MAA':maa,
        'final_gain_vs_base':final-base,'mean_new_task_gain':sum(matrix[j+1][j]-matrix[j][j] for j in range(T))/T,
        **continual_metrics(matrix),'SAPT_FWT_vs_independent':None,'oracle_gap':None}

def save_csv(path,rows):
    with path.open('w') as f:
        fields=list(rows[0]);w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def reconstruct():
    cvdata=json.loads((ROOT/'summary_cv/process_data.json').read_text())
    cvrows=list(csv.DictReader((ROOT/'summary_cv/stage_scores.csv').open()))
    totals,pertask=[] ,[]
    for method in METHODS:
        for metric in ['rougeL','exact_match','token_f1']:
            matrix=[[float(next(r for r in cvrows if r['method']==method and int(r['stage'])==s and r['evaluated_task']==task)[metric])
                for task in cvdata['task_order']] for s in range(10)]
            totals.append({'experiment':'qwen2vl9','method':method,'score':metric,**statistics(matrix)})
            for j,task in enumerate(cvdata['task_order']):
                learned=matrix[j+1][j];final=matrix[-1][j]
                historic=max(matrix[s][j] for s in range(j+1,9)) if j<8 else None
                pertask.append({'method':method,'task':task,'score':metric,'base':matrix[0][j],
                    'before_learning':matrix[j][j],'just_learned':learned,'final':final,
                    'direct_learning_gain':learned-matrix[j][j], 'final_change_after_learning':final-learned,
                    'forgetting_from_previous_best':historic-final if historic is not None else None,
                    'forward_change_before_learning':matrix[j][j]-matrix[0][j] if j else None})
    save_csv(ROOT/'summary_cv/paper_metrics.csv',totals)
    save_csv(ROOT/'summary_cv/task_learning_retention.csv',pertask)
    t5totals=[]
    for row in csv.DictReader((ROOT/'summary/t5_large_comparison.csv').open()):
        method=next(m for m in METHODS if LABELS[m]==row['method'])
        base=ROOT/row['run']/'continual'
        tasks=json.loads((base/'score_matrix.json').read_text())['tasks']
        for metric in ['rougeL','exact_match','token_f1']:
            matrix=[[json.loads((base/'scores'/f'stage_{s:02d}'/(task+'.json')).read_text())[metric] for task in tasks] for s in range(8)]
            t5totals.append({'experiment':'t5large7','method':method,'score':metric,**statistics(matrix)})
    save_csv(ROOT/'summary/t5_paper_metrics.csv',t5totals)
    # Process reports expose matrices and diagnosis; raw records are never rewritten.
    for folder,data,title,items,prefix in [('summary_cv',cvdata,'Qwen2-VL 九任务学习全过程',totals,'cv'),
                                          ('summary',json.loads((ROOT/'summary/t5_process_data.json').read_text()),'T5 七任务学习全过程',t5totals,'t5')]:
        T=len(data['task_labels'])
        lines=['# '+title,'','以同一测试集贯穿基线与全部学习阶段，直接展示“什么时候学会、在哪里退化”。分数都是 ROUGE-L（0–100），差值单位为分数点。','',
            '## 先看全过程','',f'![学习和保留过程](../sample/{prefix}_trajectories.png)','',
            '左上固定全部任务，便于观察整体变化；右上只看已学任务，其任务构成会改变；左下在同一旧任务集合上比较本次更新前后；右下展示尚未学习任务相对基座的变化。没有旧任务/未来任务时留空，不填零。','',
            '## 按论文口径重算','',
            '| 方法 | 初始固定均分 | MFT 刚学完 ↑ | MFN / AP 最终 ↑ | MAA 全过程 ↑ | 遗忘幅度 ↓ | BWT ↑ | GEM FWT ↑ |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in METHODS:
            row=next(r for r in items if r['method']==method and r['score']=='rougeL')
            lines.append('| '+LABELS[method]+' | '+' | '.join(f"{row[k]:.3f}" for k in ['baseline_fixed_mean','MFT','MFN','MAA','F.Rate','BWT','FWT'])+' |')
        lines += ['', 'MFT、MFN、MAA 对应 MLLM-CL 的计算结构；本实验以生成任务 ROUGE-L 代替该论文的任务准确率，不能直接比较论文数值。EM 和 Token F1 的同口径重算也保存在 CSV。SAPT 原文 FWT 与 GEM 不同，需要独立单任务训练对照，这里记 N/A。','',
            '## 各任务相对基线的变化','',f'![相对基线变化](../sample/{prefix}_delta_baseline.png)','',
            '每个格子是该阶段在一个固定任务上相对基座的分数变化；蓝色为提高，红色为下降。黑框是该任务刚学完的阶段。所有方法共享同一颜色范围。','']
        if folder=='summary_cv':
            lines += ['## 四个领域如何变化','', '![领域曲线](../sample/cv_domain_trajectories.png)','',
                '各领域使用固定任务集合。竖线标出开始医学语言、自然图像和医学图像之前的边界；任务顺序见下表。','',
                '## 哪一次任务切换影响了旧能力','', '![逐次任务切换变化](../sample/cv_delta_transition.png)','',
                '这里每个格子是本阶段减去上一阶段，可同时看新任务收益、旧任务下降与未来任务变化。它定位变化发生的阶段，不单凭一张图证明某个训练机制的因果效果。','',
                '![逐任务曲线](../sample/cv_task_trajectories.png)','',
                '虚线对应该任务被学习的阶段；每个小图在同一固定任务上对比四方法。','',
                '## 同一问题的回答怎样变化','',
                '[回答演变表](answer_evolution.md) 覆盖 9 个任务、4 方法、10 阶段（360 条真实回答），完整文本和分数见 [JSON](answer_evolution.json)。','']
            lines += ['## 本轮具体观察','', '| 方法 | 最严重的一次旧任务平均下降 | 更新阶段 |','|---|---:|---|']
            for method in METHODS:
                worst=min((r for r in data['methods'][method]['stage_statistics'] if r['old_fixed_shock'] is not None),key=lambda r:r['old_fixed_shock'])
                lines.append(f"| {LABELS[method]} | {worst['old_fixed_shock']:.3f} | {worst['stage']}：{data['task_labels'][worst['stage']-1]} |")
            lines += ['', '例如，MIGU-LoRA 在学习 IU-Xray 的阶段 7，先前六个任务的平均分下降 3.979；SeqLoRA 在学习对话的阶段 4，先前三任务平均分下降 2.658。这类阶段性信息在最终 FWT 中看不到。','',
                '## 这些图是否等于全局知识变化','',
                '这些曲线覆盖本实验的任务表现，包括未到训练阶段的任务；它们不是模型全部知识的直接测量。独立外部知识题的逐阶段补测见 [知识保留报告](knowledge_probe.md)。只测这九个任务，无法判断未覆盖能力、OOD 泛化或内部知识是否擦除。','']
        lines += ['## 阶段与完整矩阵','', '| 阶段 | 本阶段学习的任务 |','|---:|---|','| 0 | 初始基座 |']
        for j,label in enumerate(data['task_labels'],1):lines.append(f'| {j} | {label} |')
        for method in METHODS:
            matrix=data['methods'][method]['matrix']
            lines += ['','### '+LABELS[method],'','| 阶段 | '+' | '.join(data['task_labels'])+' |', '|---:|'+'---:|'*T]
            for stage,row in enumerate(matrix):lines.append('| '+str(stage)+' | '+' | '.join(f'{x:.3f}' for x in row)+' |')
        lines += ['', '## 指标定义与公式','',
            '设 $R_{s,j}$ 是学完第 $s$ 个任务后的任务 $j$ 得分，$R_{0,j}$ 为基座。任务 $j$ 的刚学完分数位于 $R_{j,j}$。', '',
            '$$\n'+r'\begin{aligned}',r'\mathrm{MFT} &= \frac{1}{T}\sum_{j=1}^{T} R_{j,j},\\',r'\mathrm{MFN} &= \frac{1}{T}\sum_{j=1}^{T} R_{T,j},\\',
            r'\mathrm{MAA} &= \frac{1}{T}\sum_{s=1}^{T}\left(\frac{1}{s}\sum_{j=1}^{s}R_{s,j}\right).',r'\end{aligned}'+'\n$$','',
            'MFT 是刚学完各任务的平均分，不保证是真正的上界；后续学习也可能让旧任务更好。MFN 与此处 AP 相同，MAA 汇总整个学习过程。','',
            '$$\n'+r'\begin{aligned}',r'A_s^{\mathrm{fixed}} &= \frac{1}{T}\sum_{j=1}^{T}R_{s,j},\\',
            r'S_s^{\mathrm{old}} &= \frac{1}{s-1}\sum_{j=1}^{s-1}(R_{s,j}-R_{s-1,j}),\quad s\ge 2,\\',
            r'G_s^{\mathrm{new}} &= R_{s,s}-R_{s-1,s},\\',
            r'U_s &= \frac{1}{T-s}\sum_{j=s+1}^{T}(R_{s,j}-R_{0,j}),\quad s<T.',r'\end{aligned}'+'\n$$','',
            '这里的旧任务切换冲击固定了比较的任务集合，避免新任务加入均值造成构成变化；MedCL-Bench 的原文另报告相邻已学任务平均分的差，原口径的差值也保留在过程 CSV 的 `seen_mean_transition` 列。`stage_forgetting` 记录每阶段旧任务历史最好分至当阶段的平均下降。未来任务集合仍随阶段变化，应配合逐任务曲线阅读。','',
            '## 来源、实现与重算','', '[论文指标对照](../paper/evaluation_and_process.md) 逐项说明可复原和缺少对照的项目。一个种子无法重建多种子标准差；没有 oracle、外部 OOD 测试和中间激活时，相关项目标 N/A，不能从最终数字反推。','',
            '精确数据及指标在本目录的过程 JSON / CSV，绘图不做平滑或填补。300 DPI PNG 位于 `sample/`，矢量 PDF 位于 `summary_cv/figures/`。','',
            '```bash','.vision-env/bin/python summary_cv/process.py','.vision-env/bin/python summary_cv/reconstruct_metrics.py',
            '.plot-env/bin/python summary_cv/plot_process.py','```']
        filename='process.md' if folder=='summary_cv' else 'learning_process.md'
        (ROOT/folder/filename).write_text('\n'.join(lines)+'\n')
    print('Reconstructed 24 method/score metric rows and 108 per-task learning/retention rows.')

if __name__=='__main__':reconstruct()
