"""Read-only diagnostic audit; no training, resampling, or replacement scores."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path('/data1/liyapeng/program/LLMCL')
EXP = 'ten_task_cl_20261014_geo_fin'
OUT = ROOT / 'summary' / EXP / 'failure_diagnosis_20261015'
SCORES = ['exact_match', 'token_f1', 'rougeL']
CAPS = {1:128,2:64,3:512,4:64,5:256,6:256,7:128,8:128,9:64,10:64}


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def tokens(s):
    return re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', s.lower())


def foundation(row):
    q = row['question'].lower()
    return bool(re.search(r'\borgan\b|\bplane\b|器官|部位|哪个部分|加权方式|颜色', q))


def write_csv(name, rows):
    if not rows:
        return
    with (OUT/name).open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status=json.loads((ROOT/'summary'/EXP/'run_status.json').read_text())
    mp=ROOT/'data/TenTaskCL'/EXP/'manifest.json'
    sha=hashlib.sha256(mp.read_bytes()).hexdigest()
    assert sha==status['manifest_sha256']
    manifest=json.loads(mp.read_text())
    tasks={t['task']:t for t in manifest['tasks']}
    data={t:read_rows(ROOT/x['files']['test']['path']) for t,x in tasks.items()}
    for t,rows in data.items():
        assert len(rows)==50 and len({x['id'] for x in rows})==50
        assert hashlib.sha256((ROOT/tasks[t]['files']['test']['path']).read_bytes()).hexdigest()==tasks[t]['files']['test']['sha256']
    # Tokenization is CPU-only and measures the actual fixed output budget against references.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'model/Qwen2-VL-2B-Instruct',local_files_only=True)
    input_rows=[]
    for t,rows in data.items():
        lens=[min(len(tokenizer.encode(a,add_special_tokens=False)) for a in x['answers']) for x in rows]
        input_rows.append(dict(task=t,n=50,with_context=sum(bool(x.get('context')) for x in rows),
                               with_image=sum(bool(x.get('images')) for x in rows),
                               context_over_6000=sum(len(x.get('context',''))>6000 for x in rows),
                               reference_tokens_mean=sum(lens)/50,reference_tokens_max=max(lens),
                               shortest_reference_over_generation_cap=sum(n>CAPS[t] for n in lens),
                               output_cap=CAPS[t]))
    write_csv('input_conditions.csv',input_rows)
    metrics_rows=[];source_rows=[];med_rows=[];chart_rows=[];stage_rows=[];case_rows=[]
    completed={m:j for m,j in status['jobs'].items() if j['status']=='completed'}
    for method,job in completed.items():
        p=ROOT/job['run_dir'];cfg=json.loads((p/'config.json').read_text())
        assert cfg['manifest_sha256']==sha and cfg['experiment_id']==EXP
        allpred={}
        for stage in range(11):
            for t in tasks:
                rows=read_rows(p/f'evaluations/stage_{stage:02d}/T{t:02d}_test_predictions.jsonl')
                assert len(rows)==50 and {x['id'] for x in rows}=={x['id'] for x in data[t]}
                allpred[stage,t]={x['id']:x for x in rows}
                stage_rows.append(dict(method=method,stage=stage,task=t,
                                       **{s:sum(x['scores'][s] for x in rows)/50 for s in SCORES},
                                       cap_hits=sum(x['output_tokens']>=x['generation_cap'] for x in rows),
                                       mean_output_tokens=sum(x['output_tokens'] for x in rows)/50))
        for t,rows in data.items():
            groups=defaultdict(list);med=defaultdict(list);chart=defaultdict(list)
            for x in rows:
                b=allpred[0,t][x['id']];a=allpred[t,t][x['id']];f=allpred[10,t][x['id']]
                item={'method':method,'task':t,'id':x['id'],'dataset':x['dataset'],
                      'question':x['question'],'answers':x['answers'],'metadata':x.get('metadata',{}),
                      'images':x.get('images',[]),'context':x.get('context',''),
                      'initial':b,'after_own_training':a,'final':f}
                case_rows.append(item);groups[x['dataset']].append(item)
                if t==4:med['visual_foundation' if foundation(x) else 'other_medical_questions'].append(item)
                if t==8:chart[x.get('metadata',{}).get('task_category','unknown')].append(item)
            def agg(group,items):
                out=dict(method=method,task=t,group=group,n=len(items))
                for label,key in [('initial','initial'),('after_own','after_own_training'),('final','final')]:
                    for s in SCORES:out[label+'_'+s]=sum(x[key]['scores'][s] for x in items)/len(items)
                    out[label+'_cap_hits']=sum(x[key]['output_tokens']>=x[key]['generation_cap'] for x in items)
                    # Candidate only: containment can be negated or partial and is NOT semantic correctness.
                    out[label+'_em0_reference_containment_candidates']=sum(
                        x[key]['scores']['exact_match']==0 and any(
                            tokens(a) and ' '.join(tokens(a)) in ' '.join(tokens(x[key]['prediction']))
                            for a in x['answers']) for x in items)
                out['correct_after_own_then_wrong_final']=sum(x['after_own_training']['scores']['exact_match']==100 and x['final']['scores']['exact_match']==0 for x in items)
                out['wrong_after_own_then_correct_final']=sum(x['after_own_training']['scores']['exact_match']==0 and x['final']['scores']['exact_match']==100 for x in items)
                return out
            metrics_rows.append(agg('all',case_rows[-50:]))
            source_rows.extend(agg(g,items) for g,items in groups.items())
            med_rows.extend(agg(g,items) for g,items in med.items())
            chart_rows.extend(agg(g,items) for g,items in chart.items())
    write_csv('task_diagnostics.csv',metrics_rows);write_csv('source_diagnostics.csv',source_rows)
    write_csv('medical_question_groups.csv',med_rows);write_csv('finance_chart_categories.csv',chart_rows)
    write_csv('all_stage_diagnostics.csv',stage_rows)
    with (OUT/'joined_predictions.jsonl').open('w',encoding='utf-8') as f:
        for x in case_rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    summary={'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'manifest_sha256':sha,'methods':list(completed),'status_snapshot':status,
             'verified_predictions':len(completed)*5500,'input_conditions':input_rows,
             'medical_group_rule':r'question regex: organ|plane|器官|部位|哪个部分|加权方式|颜色; exploratory question-type grouping, not semantic grading',
             'warnings':['No official scores changed','Reference containment is only a review flag, not semantic accuracy',
                         'No model checkpoints exist; post-training interventions cannot be reconstructed from predictions',
                         'Question groups are exploratory and combine different source distributions']}
    (OUT/'audit_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False)[:6500])
    for m in ['Hyper_LLaVA','ALER','FV_guided_Training']:
        print('TASK_SUMMARY',m,[(x['task'],round(x['initial_rougeL'],2),round(x['after_own_rougeL'],2),round(x['final_rougeL'],2),x['initial_exact_match'],x['after_own_exact_match'],x['final_exact_match']) for x in metrics_rows if x['method']==m])
    print('MEDICAL',[(x['method'],x['group'],x['n'],round(x['initial_rougeL'],2),round(x['final_rougeL'],2),x['initial_em0_reference_containment_candidates'],x['final_exact_match']) for x in med_rows])


if __name__=='__main__':main()
