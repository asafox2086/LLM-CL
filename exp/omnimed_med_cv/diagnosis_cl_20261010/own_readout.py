"""Incremental version of our diagnostic linear readout; standard ridge.

One 12-class matrix. Mean/std and ridge fixed using first-task train/dev only.
Historical learning retained as cumulative second-order statistics, without
raw-image replay. Deterministic; one run, no artificial duplicate seed results.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
exp=Path(__file__).resolve().parent;root=exp.parents[2]
out=exp/'runs/own_inc_readout_seed42';out.mkdir(parents=True,exist_ok=True)
def write(path,data):path.write_text(json.dumps(data,indent=2))
def status(phase,**kw):write(out/'status.json',dict(status='running',phase=phase,time=time.time(),**kw))
status('loading_features')
probes=json.loads((root/'analysis/omnimed_20261009/selection.json').read_text())['probes']
data=[]
for i,p in enumerate(probes):
    splits={}
    for split,rows in p['splits'].items():
        x=torch.stack([torch.load(root/'analysis/omnimed_20261009/features128'/(r['image_sha256']+'.pt'),map_location='cpu',weights_only=True)['features'].double().mean(0) for r in rows])
        y=torch.tensor([r['probe_label']+4*i for r in rows]);splits[split]=(x,y,rows)
    data.append(splits)
x,y,_=data[0]['train'];mean=x.mean(0);std=x.std(0,unbiased=False).clamp(min=.1)
def transform(x):return (x-mean)/std
h=transform(x);labels=torch.nn.functional.one_hot(y,12).double()
u,s,vh=torch.linalg.svd(h,full_matrices=False);uy=u.T@labels
dx,dy,_=data[0]['dev'];best=-1;selected=None
grid=[1e-4,1e-3,1e-2,.1,1.,10.,100.,1000.,10000.]
for lam in grid:
    W=vh.T@((s/(s.square()+lam)).unsqueeze(1)*uy)
    acc=float(((transform(dx)@W)[:,:4].argmax(-1)==dy).double().mean())
    if acc>best:best=acc;selected=lam
write(out/'config.json',dict(method='own_inc_readout',deterministic=True,planned_runs=1,
   feature='same frozen Qwen merger token mean',standardization='first-task train mean/std, frozen thereafter',
   ridge=selected,ridge_grid=grid,selection='first-task dev only, earliest grid value on ties',
   objective='cumulative sum ||XW-Y||^2 + lambda ||W||^2',historical_state='aggregate Gram factor and XTY',
   parameters='one shared 1536x12 classifier matrix, no source-specific heads',
   parent_protocol_sha256=hashlib.sha256((exp/'protocol.json').read_bytes()).hexdigest()))
factor=torch.empty((0,h.shape[1]),dtype=torch.float64);Q=torch.zeros(h.shape[1],12,dtype=torch.float64);matrix=[]
for stage,task in enumerate(data,1):
    status('analytical_update',stage=stage);x,y,_=task['train'];h=transform(x)
    Q+=h.T@torch.nn.functional.one_hot(y,12).double()
    _,s,vh=torch.linalg.svd(torch.cat([factor,h]),full_matrices=False)
    keep=s>float(s.max())*1e-12;s=s[keep];vh=vh[keep];factor=s.unsqueeze(1)*vh
    W=vh.T@((vh@Q)/(s.square()+selected).unsqueeze(1));scores=[]
    for i,(p,task) in enumerate(zip(probes,data)):
        x,y,rows=task['test'];z=transform(x)@W;t=y-4*i;a=z[:,4*i:4*i+4].argmax(-1);b=z[:,:4*stage].argmax(-1)
        records=[dict(method='own_inc_readout',seed=42,stage=stage,task=p['source'],condition='image',question_id=row['question_id'],
          image_sha256=row['image_sha256'],answer_index=int(label),candidate_prediction=int(pred),candidate_correct=int(label==pred),
          all_seen_prediction=int(seen),all_seen_correct=int(seen==label+4*i)) for row,label,pred,seen in zip(rows,t,a,b)]
        dest=out/'predictions'/f'stage_{stage:02d}'/f"{p['source']}_image.jsonl";dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join(json.dumps(r)+'\n' for r in records))
        score=dict(stage=stage,task=p['source'],condition='image',n=len(y),accuracy=100*float((a==t).double().mean()),
          balanced_accuracy=100*np.mean([float((a[t==k]==k).double().mean()) for k in range(4)]),all_seen_accuracy=100*float((b==y).double().mean()))
        print(json.dumps(score),flush=True);scores.append(score)
    matrix.append(scores);write(out/'scores.json',matrix)
    write(out/f'selection_stage{stage}.json',dict(ridge=selected,first_task_dev_accuracy=best,
         classifier_parameters=W.numel(),statistics_elements=factor.numel()+Q.numel(),normalization_elements=mean.numel()+std.numel()))
    cp=out/'checkpoints';cp.mkdir(exist_ok=True);torch.save(dict(W=W,factor=factor,Q=Q,mean=mean,std=std,ridge=selected),cp/f'stage_{stage:02d}.pt')
write(out/'status.json',dict(status='completed',method='own_inc_readout',seed=42,deterministic=True,stages=3,time=time.time()))
