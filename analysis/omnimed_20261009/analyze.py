import json,collections,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
def read(name):return [json.loads(x) for x in (OUT/name).read_text().splitlines()]
sel=json.loads((OUT/'selection.json').read_text());records=read('predictions.jsonl')
assert len({(r['arm'],r['suite'],r['condition'],r['question_id']) for r in records})==len(records)
arms=['base','SFT_r8_42','SFT_r8_43','SFT_r8_44'];by={(r['arm'],r['suite'],r['condition'],r['question_id']):r for r in records}
expected=(len(sel['primary'])+2*len(sel['matched_pairs'])+sum(len(p['splits']['test']) for p in sel['probes']))*8
assert len(records)==expected,(len(records),expected)
def boot(values,clusters,seed=42):
    # columns are checkpoints, rows are independent image clusters (not patients).
    buckets=collections.defaultdict(list)
    for i,c in enumerate(clusters):buckets[c].append(i)
    items=list(buckets.values());x=np.array([np.asarray(values)[ix].sum(0) for ix in items]);weights=np.array([len(ix) for ix in items])
    rng=np.random.default_rng(seed);means=[]
    for _ in range(5000):
        s=rng.integers(len(x),size=len(x));k=rng.integers(x.shape[1],size=x.shape[1]);means.append(x[s][:,k].sum()/weights[s].sum()/len(k))
    return [float(v*100) for v in np.percentile(means,[2.5,97.5])]
scores=[]
for category in sorted({r['question_type'] for r in sel['primary']})+['ALL']:
    rs=[r for r in sel['primary'] if category=='ALL' or r['question_type']==category]
    for group,aa in [('base',['base']),('SFT_r8_mean',arms[1:])]:
        matrices={}
        for condition in ['image','no_image']:
            x=np.array([[by[(a,'primary',condition,r['question_id'])]['candidate_correct'] for a in aa] for r in rs])
            strict=np.array([[by[(a,'primary',condition,r['question_id'])]['strict_first_token_correct'] for a in aa] for r in rs])
            scores.append(dict(category=category,arm=group,condition=condition,n=len(rs),accuracy=float(x.mean()*100),strict_first_token_accuracy=float(strict.mean()*100),image_cluster_ci=boot(x,[r['image_sha256'] for r in rs])))
            matrices[condition]=x
        delta=matrices['image']-matrices['no_image']
        scores.append(dict(category=category,arm=group,condition='image_minus_no_image',n=len(rs),difference_pp=float(delta.mean()*100),image_cluster_ci=boot(delta,[r['image_sha256'] for r in rs])))
matched=[]
for a in arms:
    table=np.zeros((2,2),dtype=int)
    for diagnosis,coarse in sel['matched_pairs']:
        d=by[(a,'matched','image',diagnosis['question_id'])]['candidate_correct'];c=by[(a,'matched','image',coarse['question_id'])]['candidate_correct'];table[c,d]+=1
    matched.append(dict(arm=a,n_pairs=len(sel['matched_pairs']),rows='coarse wrong, coarse correct',columns='diagnosis wrong, diagnosis correct',table=table.tolist(),coarse_accuracy=float(table[1].sum()/table.sum()*100),diagnosis_accuracy=float(table[:,1].sum()/table.sum()*100)))
probe=read('probe_predictions.jsonl');probe_scores=read('probe_scores.jsonl');probe_comparisons=[]
for p in sel['probes']:
    rs=p['splits']['test'];pp={r['question_id']:r for r in probe if r['source']==p['source']}
    y=np.array([r['probe_label'] for r in rs]);linear=np.array([pp[r['question_id']]['correct'] for r in rs])
    shuffle=np.array([pp[r['question_id']]['shuffle_correct'] for r in rs])
    rec=dict(source=p['source'],n=len(rs),classes=p['classes'],linear_accuracy=float(linear.mean()*100),linear_ci=boot(linear[:,None],[r['image_sha256'] for r in rs]),shuffle_accuracy=float(shuffle.mean()*100),linear_balanced_accuracy=float(np.mean([linear[y==i].mean() for i in range(len(p['classes']))])*100))
    for group,aa in [('base',['base']),('SFT_r8_mean',arms[1:])]:
        lm=np.array([[by[(a,'probe_test','image',r['question_id'])]['candidate_correct'] for a in aa] for r in rs])
        rec[group]=dict(accuracy=float(lm.mean()*100),accuracy_ci=boot(lm,[r['image_sha256'] for r in rs]),balanced_accuracy=float(np.mean([lm[y==i].mean() for i in range(len(p['classes']))])*100),linear_minus_lm_pp=float((linear[:,None]-lm).mean()*100),difference_ci=boot(linear[:,None]-lm,[r['image_sha256'] for r in rs]),linear_correct_lm_wrong_mean_count=float(((linear[:,None]==1)&(lm==0)).sum(0).mean()))
    probe_comparisons.append(rec)
summary=dict(primary=scores,matched=matched,probe=probe_comparisons,probe_selection=probe_scores,prediction_rows=len(records),unit='SHA image clusters; patient identity unavailable; checkpoint means not independent extra patients',interpretation='Frozen feature readout measures recoverability of dataset labels after supervised readout fitting, not verified clinical finding recognition or causal explanation of diagnostic failure.')
(OUT/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
