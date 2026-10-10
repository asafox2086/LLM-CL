import json,collections
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
def read(name):return [json.loads(x) for x in (OUT/name).read_text().splitlines()]
sel=json.loads((OUT/'selection.json').read_text());old=read('predictions.jsonl');new=read('adaptation_predictions.jsonl')
linear_lookup={r['question_id']:r for r in read('probe_predictions.jsonl')}
assert json.loads((OUT/'adaptation_status.json').read_text())['status']=='completed'
assert len(new)==3*2*sum(len(p['splits']['test']) for p in sel['probes'])
before={(r['arm'],r['condition'],r['question_id']):r for r in old if r['suite']=='probe_test'}
after={(r['seed'],r['split'],r['question_id']):r for r in new}
assert len(after)==len(new)
def ci(x):
    rng=np.random.default_rng(42);x=np.asarray(x);values=[]
    for _ in range(5000):values.append(x[rng.integers(len(x),size=len(x))][:,rng.integers(x.shape[1],size=x.shape[1])].mean()*100)
    return np.percentile(values,[2.5,97.5]).tolist()
out=[]
for source in [p['source'] for p in sel['probes']]+['ALL']:
    rs=[r for p in sel['probes'] if source=='ALL' or source==p['source'] for r in p['splits']['test']]
    item=dict(source=source,n=len(rs));mat={};linear=np.array([linear_lookup[r['question_id']]['correct'] for r in rs])
    for condition,split in [('image','test'),('no_image','test_no_image')]:
        oldx=np.array([[before[(f'SFT_r8_{seed}',condition,r['question_id'])]['candidate_correct'] for seed in [42,43,44]] for r in rs])
        basex=np.array([[before[('base',condition,r['question_id'])]['candidate_correct']] for r in rs])
        newx=np.array([[after[(seed,split,r['question_id'])]['candidate_correct'] for seed in [42,43,44]] for r in rs])
        strict=np.array([[after[(seed,split,r['question_id'])]['strict_first_token_correct'] for seed in [42,43,44]] for r in rs])
        mat[condition]=(oldx,basex,newx)
        classgroups=collections.defaultdict(list)
        for i,r in enumerate(rs):classgroups[(r['dataset'],r['probe_label'])].append(i)
        item[condition]=dict(old_medical_lora_accuracy=float(oldx.mean()*100),base_accuracy=float(basex.mean()*100),new_lora_accuracy=float(newx.mean()*100),new_lora_balanced_accuracy=float(np.mean([newx[ix].mean() for ix in classgroups.values()])*100),new_lora_ci=ci(newx),strict_accuracy=float(strict.mean()*100),new_minus_old_pp=float((newx-oldx).mean()*100),new_minus_old_ci=ci(newx-oldx),new_minus_base_pp=float((newx-basex).mean()*100),new_minus_base_ci=ci(newx-basex),linear_accuracy=float(linear.mean()*100),linear_minus_new_lora_pp=float((linear[:,None]-newx).mean()*100),linear_minus_new_lora_ci=ci(linear[:,None]-newx))
    oi,bi,ni=mat['image'];on,bn,nn=mat['no_image']
    item['visual_contribution']=dict(new_image_minus_no_image_pp=float((ni-nn).mean()*100),new_image_minus_no_image_ci=ci(ni-nn),change_vs_old_pp=float(((ni-nn)-(oi-on)).mean()*100),change_vs_old_ci=ci((ni-nn)-(oi-on)),change_vs_base_pp=float(((ni-nn)-(bi-bn)).mean()*100),change_vs_base_ci=ci((ni-nn)-(bi-bn)))
    out.append(item)
result=dict(results=out,training_scores=read('adaptation_scores.jsonl'),unit='independent image SHA plus seed resampling; patient IDs unavailable',scope='Exploratory same-label supervision control, three datasets with four selected classes each; base vision frozen')
(OUT/'adaptation_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
