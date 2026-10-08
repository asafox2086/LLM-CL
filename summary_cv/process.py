"""Derive stage trajectories from existing scores; no additional training."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METHODS=['seq_lora','migu_lora','olora','sapt_lora']
NAMES={'seq_lora':'SeqLoRA','migu_lora':'MIGU-LoRA','olora':'O-LoRA','sapt_lora':'SAPT-LoRA'}
DOMAINS={'general_text':list(range(5)), 'medical_text':[5,6], 'natural_vision':[7], 'medical_vision':[8]}

def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def derive(matrix):
    T=len(matrix[0]);assert len(matrix)==T+1
    assert all(len(row)==T for row in matrix)
    assert all(isinstance(x,(int,float)) and 0<=x<=100 for row in matrix for x in row)
    records=[]
    for s,row in enumerate(matrix):
        mean=lambda values:sum(values)/len(values) if values else None
        records.append({'stage':s, 'all_fixed_mean':mean(row),
            'all_delta_base':mean([row[j]-matrix[0][j] for j in range(T)]),
            'seen_mean':mean(row[:s]),
            'old_fixed_shock':mean([row[j]-matrix[s-1][j] for j in range(s-1)]) if s>1 else None,
            'new_task_gain':row[s-1]-matrix[s-1][s-1] if s else None,
            'future_delta_base':mean([row[j]-matrix[0][j] for j in range(s,T)]),
            'old_bwt':mean([row[j]-matrix[j+1][j] for j in range(s-1)]) if s>1 else None,
            'seen_mean_transition':mean(row[:s])-mean(matrix[s-1][:s-1]) if s>1 else None,
            'stage_forgetting':mean([max(matrix[k][j] for k in range(j+1,s))-row[j] for j in range(s-1)]) if s>1 else None})
    delta_base=[[row[j]-matrix[0][j] for j in range(T)] for row in matrix]
    delta_step=[[matrix[s][j]-matrix[s-1][j] for j in range(T)] for s in range(1,T+1)]
    return {'stage_statistics':records,'delta_baseline':delta_base,'delta_transition':delta_step}

def collect_process():
    run=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run'])
    config=json.loads((run/'seq_lora/config.json').read_text())
    output={'protocol':config['protocol'],'units':'ROUGE-L points (0-100; differences signed)',
            'task_order':config['tasks'],'task_labels':['XSum','Quoref','Relation','Dialogue','Cause','MTS','IU-Xray','VQAv2','VQA-RAD'],
            'domains':DOMAINS,'methods':{}}
    csv_rows=[]
    for method in METHODS:
        path=run/method/'continual_metrics.json';data=json.loads(path.read_text());matrix=data['matrix']
        metrics=derive(matrix)
        output['methods'][method]={'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'matrix':matrix,**metrics}
        for row in metrics['stage_statistics']:
            csv_rows.append({'method':method,**row})
    write(ROOT/'summary_cv/process_data.json',output)
    with (ROOT/'summary_cv/process_metrics.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csv_rows)
    t5={'protocol':'superni_generation7_v2','task_labels':['XSum','Quoref','Relation','Dialogue','Cause','Reddit','SciQ'],'methods':{}}
    for record in csv.DictReader((ROOT/'summary/t5_large_comparison.csv').open()):
        method=next(m for m in METHODS if NAMES[m]==record['method'])
        path=ROOT/record['run']/'continual/score_matrix.json';raw=json.loads(path.read_text())
        matrix=raw.get('matrix') or raw.get('scores') or raw.get('rows')
        assert isinstance(matrix,list),raw.keys()
        t5['methods'][method]={'source':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                              'matrix':matrix,**derive(matrix)}
    write(ROOT/'summary/t5_process_data.json',t5)
    # Full answer evolution on the same fixed-order examples, all 10 stages.
    examples=json.loads((ROOT/'summary_cv/examples.json').read_text())
    evolution=[]
    for ex in examples:
        if ex['kind']!='task':continue
        item={'task_id':ex['task_id'],'instance_id':ex['instance_id'],'title':ex['title'],
              'references':ex['source']['references'],'prompt':ex['source']['prompt'],
              'display_image':ex.get('display_image'),'methods':{}}
        for method in METHODS:
            item['methods'][method]=[]
            for stage in range(10):
                path=run/method/'predictions'/f'stage_{stage:02d}'/(ex['task_id']+'.jsonl')
                matches=[json.loads(line) for line in path.read_text().splitlines() if json.loads(line)['instance_id']==ex['instance_id']]
                assert len(matches)==1
                record=matches[0]
                item['methods'][method].append({k:record[k] for k in ['stage','prediction','scores','generated_token_ids','hit_generation_limit']})
        evolution.append(item)
    write(ROOT/'summary_cv/answer_evolution.json',evolution)
    lines=['# 同一测试实例的回答演变','','每个任务沿用固定顺序的第一条示例；完整记录覆盖基线和全部九个学习阶段，未按回答变化或得分挑选。表格只节选前 140 个字符，完整文本和逐题分数见 [JSON](answer_evolution.json)。','']
    for item in evolution:
        lines += ['## '+item['title'],'',f"实例：`{item['instance_id']}`；完整输入与参考见 [实例文档](examples.md)。",'']
        if item['display_image']:lines += [f"![{item['title']}](../{item['display_image']})",'']
        lines += ['| 阶段 | SeqLoRA | MIGU-LoRA | O-LoRA | SAPT-LoRA |','|---:|---|---|---|---|']
        for stage in range(10):
            cells=[]
            for method in METHODS:
                text=' '.join(item['methods'][method][stage]['prediction'].split())
                cells.append((text[:140]+('…' if len(text)>140 else '')).replace('|','\\|').replace('<','&lt;'))
            lines.append('| '+str(stage)+' | '+' | '.join(cells)+' |')
        lines.append('')
    (ROOT/'summary_cv/answer_evolution.md').write_text('\n'.join(lines).rstrip()+'\n')
    return output

if __name__=='__main__':collect_process()
