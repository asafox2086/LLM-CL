"""Measure paired learning gains and retention on identical held-out examples."""
import csv,json
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
NAMES={'seq_lora':'SeqLoRA','migu_lora':'MIGU-LoRA','olora':'O-LoRA','sapt_lora':'SAPT-LoRA'}

def collect():
    data=json.loads((ROOT/'summary_cv/process_data.json').read_text())
    run=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run'])
    scores=list(csv.DictReader((ROOT/'summary_cv/task_learning_retention.csv').open()))
    pairs=[];items=[]
    for method in NAMES:
        for j,task in enumerate(data['task_order'],1):
            records=[]
            for stage in [j-1,j,9]:
                path=run/method/'predictions'/f'stage_{stage:02d}'/(task+'.jsonl')
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                mapped={r['instance_id']:r for r in rows}
                assert len(mapped)==len(rows)==200
                records.append(mapped)
            before,after,final=records;assert before.keys()==after.keys()==final.keys()
            statuses=[]
            for ident,a in before.items():
                b=after[ident];c=final[ident]
                assert a['references']==b['references']==c['references']
                row={'method':method,'task':task,'instance_id':ident}
                for label,r in [('before',a),('learned',b),('final',c)]:row[label+'_EM_correct']=r['scores']['exact_match']==100
                row['newly_correct']=not row['before_EM_correct'] and row['learned_EM_correct']
                row['newly_wrong']=row['before_EM_correct'] and not row['learned_EM_correct']
                row['newly_correct_retained']=row['newly_correct'] and row['final_EM_correct']
                items.append(row);statuses.append(row)
            pairs.append({'method':method,'task':task,'count':200,
                **{key:sum(r[key] for r in statuses) for key in ['before_EM_correct','learned_EM_correct','final_EM_correct','newly_correct','newly_wrong','newly_correct_retained']}})
            p=pairs[-1];p['newly_correct_retention_percent']=100*p['newly_correct_retained']/p['newly_correct'] if p['newly_correct'] else None
            p['EM_net_gain_points']=100*(p['newly_correct']-p['newly_wrong'])/200
            source=next(r for r in scores if r['method']==method and r['task']==task and r['score']=='exact_match')
            assert abs(p['EM_net_gain_points']-float(source['direct_learning_gain']))<1e-8
    for filename,rows in [('learning_answer_transitions.csv',pairs),('learning_answer_transition_items.csv',items)]:
        with (ROOT/'summary_cv'/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    lines=['# 学到了多少，后来还剩多少？','',
        '本报告回答学习收益与保留量：对同一批固定测试题，比较任务训练前、刚学完和最终表现，并核对新增答对的题后来是否仍答对。测试题不参加训练。它量化可观察的任务学习，不能给内部新增知识计数，也不单凭最终分数判断学习量。','',
        '## 四种方法的学习收益','',
        '| 方法 | 学完 − 学前 | 学完 − 基座 | 最终 − 基座 | 最终 − 刚学完 |','|---|---:|---:|---:|---:|']
    for method in NAMES:
        group=[r for r in scores if r['method']==method and r['score']=='rougeL']
        values=[mean(float(r['direct_learning_gain']) for r in group),mean(float(r['just_learned'])-float(r['base']) for r in group),mean(float(r['final'])-float(r['base']) for r in group),mean(float(r['final_change_after_learning']) for r in group)]
        lines.append('| '+NAMES[method]+' | '+' | '.join(f'{v:+.3f}' for v in values)+' |')
    lines += ['', '数值是九任务等权平均的 ROUGE-L 分数点。第一列衡量该阶段训练后改变了多少；第二列同时检查是否超越原始基座；第三列看整条学习路线的最终净收益；第四列看后续学习对刚获得表现的影响。','',
        '**必须同时读前两列。** 若前序学习先损害了某个未来任务，训练该任务的高收益可能包含恢复原有能力，不能全部算新增知识。例如关系抽取上 SeqLoRA 从基座 49.954 降至学前 20.183，训练后为 98.375：直接收益 +78.192，但相对基座的收益为 +48.421。','',
        '## 每个任务实际提高了多少','',
        '| 方法 | 任务 | 基座 | 学前 | 刚学完 | 最终 | 直接学习收益 | 最终相对基座 | 后续改变 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in NAMES:
        for task,label in zip(data['task_order'],data['task_labels']):
            r=next(r for r in scores if r['method']==method and r['task']==task and r['score']=='rougeL')
            values=[float(r[k]) for k in ['base','before_learning','just_learned','final','direct_learning_gain']]+[float(r['final'])-float(r['base']),float(r['final_change_after_learning'])]
            lines.append('| '+NAMES[method]+' | '+label+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines += ['', '## 新增答对的测试题有没有保住','',
        '| 方法 | 新增答对 | 原先答对却变错 | 新增答对且最终仍对 | 新增正确题保留比例 |','|---|---:|---:|---:|---:|']
    for method in NAMES:
        group=[r for r in pairs if r['method']==method]
        new=sum(r['newly_correct'] for r in group);lost=sum(r['newly_wrong'] for r in group);retained=sum(r['newly_correct_retained'] for r in group)
        lines.append(f'| {NAMES[method]} | {new} | {lost} | {retained} | {100*retained/new:.2f}% |')
    lines += ['', '每方法涉及 9×200 个任务测试题。这里“答对”严格按归一化 EM，适合短答案和关系抽取；摘要、开放对话和临床记录很少完全匹配，不能将其 EM=0 解读为没学到东西。因此同时提供上述 ROUGE-L 收益和 CSV 中的 Token F1 收益。该保留比例只跟踪本任务训练时新增答对的题；后来才答对的题不进入它的分子。','',
        '[每任务新增正确、变错、保留计数](learning_answer_transitions.csv) · [7,200 条逐题配对结果](learning_answer_transition_items.csv) · [三项评分的学习与保留](task_learning_retention.csv)','',
        '## 这些结果能够说明什么','',
        '- 同任务学前→学后差值说明固定测试集上的表现变化，提供学习收益，而最终分数反映最终任务水平。两者应一起比较。',
        '- 基座→刚学完帮助区分恢复早期退化与超越初始能力；刚学完→最终、新增正确题的保留比例说明后续学习保住了多少可测表现。',
        '- [全过程](process.md) 定位变化发生在哪一步；[外部知识](knowledge_probe.md) 检查未参与本次训练的知识题是否受到影响；[回答演变](answer_evolution.md) 解释分数背后的文字变化。',
        '- 单顺序、单种子、短预算和有限测试任务不能直接度量内部知识总量或证明一般因果效应。要进一步回答哪个算法带来更多新增能力，需要独立任务训练对照、多种子和更广泛的固定外部评测。','',
        '## 指标定义与重算','',
        '$$',r'\begin{aligned}',r'G_j^{\mathrm{direct}} &= R_{j,j}-R_{j-1,j},\\',r'G_j^{\mathrm{above\ base}} &= R_{j,j}-R_{0,j},\\',r'G_j^{\mathrm{final}} &= R_{T,j}-R_{0,j},\\',r'D_j^{\mathrm{later}} &= R_{T,j}-R_{j,j}.',r'\end{aligned}','$$','',
        '差值保留正负号，不用不稳定的“收益百分比”或接近零的基线作分母。全体概览为上述各任务差值的等权平均。新增正确题保留比例的分母是学前 EM 错、刚学完 EM 对的题数；分子进一步要求最终 EM 对。','',
        '```bash','.vision-env/bin/python summary_cv/learning_gain.py','```']
    (ROOT/'summary_cv/learning_gain.md').write_text('\n'.join(lines)+'\n')
    print('Verified 7,200 paired test items and 36 task-level learning/retention counts.')
if __name__=='__main__':collect()
