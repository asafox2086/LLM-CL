import csv
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root/'summary_cv_med'
run = Path(json.loads((out/'latest.json').read_text())['run'])
def csvrows(name):
    with (out/name).open() as f:
        return list(csv.DictReader(f))
details = csvrows('old_task_impact_details.csv')
stages = csvrows('old_task_impact_stages.csv')
assert len(details)==40 and len(stages)==16
for d in details:
    method = d['method'];s=int(d['stage']);task=d['old_test_task']
    predictions = []
    for stage in (s-1,s):
        path=run/method/'predictions'/f'stage_{stage:02d}'/f'{task}.jsonl'
        predictions.append([json.loads(x) for x in path.read_text().splitlines()])
    before,after=predictions
    assert [r['instance_id'] for r in before]==[r['instance_id'] for r in after]
    assert len(before)==len(after)==int(d['test_count'])
    for rows,col in ((before,'before_accuracy'),(after,'after_accuracy')):
        score=sum(r['scores']['exact_match'] for r in rows)/len(rows)
        assert abs(score-float(d[col]))<1e-9
    assert abs(float(d['after_accuracy'])-float(d['before_accuracy'])-float(d['change_pp']))<1e-9
for s in stages:
    matches=[d for d in details if d['method']==s['method'] and d['stage']==s['stage']]
    assert len(matches)==int(s['stage'])-1
    assert sum(int(d['test_count']) for d in matches)==int(s['old_test_questions'])
    assert abs(sum(float(d['change_pp']) for d in matches)/len(matches)-float(s['old_task_delta_pp']))<1e-9
example=next(r for r in stages if r['method']=='seq_lora' and r['stage']=='2')
assert float(example['old_task_delta_pp'])==-1
for file in (out/'README.md',root/'summary/medical_research_comparison.md'):
    text=file.read_text()
    assert '本研究：诊断标签监督 LoRA' in text and '本研究：冻结特征线性读出' in text
    assert 'A／左上' in text and 'B／右上' in text and 'C／左下' in text and 'D／右下' in text
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if '://' in link: continue
        target=link.split('#')[0]
        if target: assert (file.parent/target).exists(), (file,link)
audit={'old_task_pairs_checked':len(details),'stage_macro_deltas_checked':len(stages),
       'same_test_instance_ids_before_after':True,'scores_recomputed_from_predictions':True,
       'seq_lora_stage2_delta_pp':-1.0,'caption_panels':'A=all_task_em; B=medical_probe; C=old_task_delta; D=general_probe',
       'csv_sha256':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
                     ['old_task_impact_details.csv','old_task_impact_stages.csv']}}
(out/'old_task_impact_verification.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
