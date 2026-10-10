"""RA-LDL Qwen feature port: paper eq.3/4 and official replace_fc.

Optional backbone tuning omitted, shared frozen features mean-pooled. M=3840,
r=64 follows released args. Use eq.4 ReLU for LRP at both training/inference;
released inc_net.py omitted inference ReLU, unlike paper and replace_fc.
Ridge solved in the empirical row span in float64, algebraically equivalent to
(G+lambda I)^-1 Q because Q lies in span(G). Retains aggregate factor and Q,
never retains historical image feature rows for replay.
"""
import argparse,json,hashlib,math,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
torch.set_num_threads(4)
exp=Path(__file__).resolve().parent;root=exp.parents[2]
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))
def main(args):
    out=exp/'runs'/f'{args.method}_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    def status(state,**kw):write(out/'status.json',dict(status=state,method=args.method,seed=args.seed,time=time.time(),**kw))
    status('running',phase='loading_features')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    probes=json.loads((root/'analysis/omnimed_20261009/selection.json').read_text())['probes']
    datasets=[]
    for i,p in enumerate(probes):
        splits={}
        for split,rows in p['splits'].items():
            x=torch.stack([torch.load(root/'analysis/omnimed_20261009/features128'/(r['image_sha256']+'.pt'),map_location='cpu',weights_only=True)['features'].float().mean(0) for r in rows])
            y=torch.tensor([r['probe_label']+4*i for r in rows]);splits[split]=(x,y,rows)
        datasets.append(splits)
    d=datasets[0]['train'][0].shape[1];m=3840;r=64
    rand=torch.randn(d,m);down=nn.Linear(d,r);up=nn.Linear(r,m);act=nn.GELU()
    classifier=nn.Linear(m,4,bias=False);classifier.weight.requires_grad_(False)
    def features(x):return (x@rand).relu()+up(act(down(x))).relu()
    cfg=dict(seed=args.seed,method='ra_ldl',feature_dimension=d,M=m,rank=r,epochs=3,batch=8,
             optimizer='SGD',lr=.01,momentum=.9,weight_decay=.0005,
             optional_backbone_tuning=False,LRP_activation='GELU then ReLU consistently per paper eq.4',
             source_revision='3630a3d2ff894e9bf5bc34fff8f3d6468d242327',
             protocol_sha256=hashlib.sha256((exp/'protocol.json').read_bytes()).hexdigest(),
             ridge_selection='current training 80/20 internal split, 10**(-8..8), MSE, follows official optimise_ridge_parameter',
             classifier_solver='float64 empirical row-span SVD; sufficient-statistic factor and Q only')
    write(out/'config.json',cfg)
    optimizer=torch.optim.SGD([*down.parameters(),*up.parameters()],lr=.01,momentum=.9,weight_decay=.0005)
    x,y,_=datasets[0]['train'];best=-1;best_state=None
    for epoch in range(1,4):
        status('running',phase='first_session_LRP',stage=1,epoch=epoch)
        order=list(range(len(y)));np.random.RandomState(args.seed+epoch+100).shuffle(order)
        losses=[]
        for start in range(0,len(order),8):
            ids=order[start:start+8];optimizer.zero_grad();loss=nn.functional.cross_entropy(classifier(features(x[ids])),y[ids])
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([*down.parameters(),*up.parameters()],1.);optimizer.step();losses.append(float(loss.detach()))
        with torch.no_grad():
            vx,vy,_=datasets[0]['dev'];acc=float((classifier(features(vx)).argmax(-1)==vy).float().mean())
        print('LRP',json.dumps(dict(epoch=epoch,loss=float(np.mean(losses)),dev_accuracy=acc)),flush=True)
        if acc>best:
            best=acc;best_state={k:v.detach().clone() for k,v in nn.ModuleDict({'down':down,'up':up}).state_dict().items()}
    nn.ModuleDict({'down':down,'up':up}).load_state_dict(best_state);down.requires_grad_(False);up.requires_grad_(False)
    factor=torch.empty((0,m),dtype=torch.float64);Q=torch.zeros(m,12,dtype=torch.float64);matrix=[]
    def ridge_weights(h,y,lam):
        u,s,vh=torch.linalg.svd(h,full_matrices=False)
        return vh.T@((s/(s.square()+lam)).unsqueeze(1)*(u.T@y))
    for stage,data in enumerate(datasets,1):
        status('running',phase='analytical_update',stage=stage)
        x,y,rows=data['train']
        with torch.no_grad():h=features(x).double()
        labels=nn.functional.one_hot(y,12).double();split=int(.8*len(h));candidates=10.**np.arange(-8,9)
        losses=[]
        # One small SVD evaluates the released ridge grid without test access.
        u,s,vh=torch.linalg.svd(h[:split],full_matrices=False);uy=u.T@labels[:split]
        for lam in candidates:
            w=vh.T@((s/(s.square()+lam)).unsqueeze(1)*uy)
            losses.append(float((h[split:]@w-labels[split:]).square().mean()))
        lam=float(candidates[int(np.argmin(losses))]);Q+=h.T@labels
        _,s,vh=torch.linalg.svd(torch.cat([factor,h]),full_matrices=False)
        # Numerical zero rows contain no information in G; remove below relative epsilon.
        keep=s>float(s.max())*1e-12;s=s[keep];vh=vh[keep];factor=s.unsqueeze(1)*vh
        W=vh.T@((vh@Q)/(s.square()+lam).unsqueeze(1))
        results=[]
        for i,(test,p) in enumerate(zip(datasets,probes)):
            tx,ty,trs=test['test']
            with torch.no_grad():z=features(tx).double()@W
            pred=z[:,4*i:4*i+4].argmax(-1);truth=ty-4*i
            all_seen=z[:,:4*stage].argmax(-1)
            records=[dict(method='ra_ldl',seed=args.seed,stage=stage,task=p['source'],condition='image',
                     question_id=row['question_id'],image_sha256=row['image_sha256'],answer_index=int(t),
                     candidate_prediction=int(a),candidate_correct=int(a==t),all_seen_prediction=int(b),
                     all_seen_correct=int(b==t+4*i)) for row,t,a,b in zip(trs,truth,pred,all_seen)]
            path=out/'predictions'/f'stage_{stage:02d}'/f"{p['source']}_image.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps(row)+'\n' for row in records))
            acc=100*float((pred==truth).double().mean());ba=100*np.mean([float((pred[truth==k]==k).double().mean()) for k in range(4)])
            rec=dict(stage=stage,task=p['source'],condition='image',n=len(ty),accuracy=acc,balanced_accuracy=ba,
                     all_seen_accuracy=100*float((all_seen==ty).double().mean()))
            results.append(rec);print('SCORE',json.dumps(rec),flush=True)
        matrix.append(results);write(out/'scores.json',matrix)
        write(out/f'selection_stage{stage}.json',dict(ridge=lam,current_train_internal_holdout_mse=min(losses),
              projection_trainable_parameters=sum(p.numel() for p in [*down.parameters(),*up.parameters()]),
              frozen_random_parameters=rand.numel(),classifier_parameters=W.numel(),sufficient_statistics_elements=factor.numel()+Q.numel()))
        cp=out/'checkpoints';cp.mkdir(exist_ok=True)
        torch.save(dict(rand=rand,down=down.state_dict(),up=up.state_dict(),factor=factor,Q=Q,W=W,stage=stage,config=cfg),cp/f'stage_{stage:02d}.pt')
    status('completed',stages=3)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    try:main(args)
    except BaseException:
        write(exp/'runs'/f'{args.method}_seed{args.seed}'/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise
