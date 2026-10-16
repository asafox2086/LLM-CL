"""Read-only CPU audit of saved OneVision adapters; never changes an active runner.

Run on the server with .vision-env/bin/python analysis/onevision_parameter_audit.py.
Only complete torch.save files are admitted. No GPU, training or test selection.
"""
import hashlib,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'exp/onevision_diagnostics_20261016'
sys.path.insert(0,str(HERE))
from geometry import inner,spectrum

def basis(matrix):
    u,s,_=torch.linalg.svd(matrix.double(),full_matrices=False)
    if not s.numel() or not float(s[0]):return u[:,:0]
    return u[:,s>s[0]*1e-8]

def overlap(a,b):
    # Mean squared canonical correlations; None when a subspace has zero rank.
    qa,qb=basis(a),basis(b)
    if not qa.shape[1] or not qb.shape[1]:return None
    s=torch.linalg.svdvals(qa.T@qb)
    return float(s.square().mean())

def change(current,previous):
    # Exact BA difference, represented at rank<=16 instead of allocating dense W.
    return {'B':torch.cat([current['B'],-previous['B']],dim=1),
            'A':torch.cat([current['A'],previous['A']],dim=0)}

def audit(run,base_norms=None):
    output={'run':str(run.relative_to(ROOT)),'checkpoint_audits':{},'isolated_pair_geometry':{},
            'notes':['No full per-update gradient tensors or optimizer states were saved by the active runner.',
                     'The first sequential displacement from random initialization needs reconstructed initial adapter; not inferred as zero.',
                     'Gradient cosine files use only two TRAIN probes/domain and do not prove a causal mechanism.']}
    saved={}
    for path in sorted((run/'adapters').glob('*.pt')):
        try:obj=torch.load(path,map_location='cpu',weights_only=True)
        except (OSError,EOFError,RuntimeError):continue  # worker may still be writing a file
        state=obj['adapter'];assert len(state)==48
        assert sum(x.numel() for row in state.values() for x in row.values())==540672
        assert all(torch.isfinite(x).all() for row in state.values() for x in row.values())
        saved[path.stem]=state;layers={}
        for key,a in state.items():
            n=4*math.sqrt(max(0.,inner(a,a)))
            layers[key]={'A_norm':float(a['A'].double().norm()),'B_norm':float(a['B'].double().norm()),
                         'deltaW_norm':n,**spectrum(a)}
            if base_norms and key in base_norms:layers[key]['deltaW_over_baseW']=n/max(base_norms[key],1e-30)
        output['checkpoint_audits'][path.stem]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'bytes':path.stat().st_size,'base_revision':obj['base_revision'],'rank':obj['rank'],
                'alpha':obj['alpha'],'parameter_count':540672,'finite':True,'layers':layers}
    isolated={k:v for k,v in saved.items() if k.startswith('isolated_')}
    for name,a in isolated.items():
        output['isolated_pair_geometry'][name]={}
        for other,b in isolated.items():
            aa=sum(inner(x,x) for x in a.values());bb=sum(inner(x,x) for x in b.values())
            cosine=sum(inner(a[k],b[k]) for k in a)/math.sqrt(aa*bb) if aa*bb>0 else None
            output['isolated_pair_geometry'][name][other]={'deltaW_cosine':cosine,
                 'layer_input_subspace_overlap':{k:overlap(a[k]['A'].T,b[k]['A'].T) for k in a},
                 'layer_output_subspace_overlap':{k:overlap(a[k]['B'],b[k]['B']) for k in a}}
    output['sequential_increment_geometry']={}
    for stage in range(2,20):
        now=f'sequential_T{stage:02d}';previous=f'sequential_T{stage-1:02d}'
        if now not in saved or previous not in saved:continue
        output['sequential_increment_geometry'][now]={}
        for key,a in saved[now].items():
            delta=change(a,saved[previous][key]);old=saved[previous][key]
            norm=math.sqrt(max(0.,inner(delta,delta)));old_norm=math.sqrt(max(0.,inner(old,old)))
            output['sequential_increment_geometry'][now][key]={
                'increment_deltaW_norm':4*norm,
                'increment_vs_previous_adapter_cosine':inner(delta,old)/(norm*old_norm) if norm*old_norm>0 else None,
                **spectrum(delta)}
    if (run/'events.jsonl').exists():
        events=[]
        for line in (run/'events.jsonl').read_text().splitlines():
            try:events.append(json.loads(line))
            except json.JSONDecodeError:break
        output['optimizer_schedule_audit']={}
        for label in sorted({x['label'] for x in events}):
            rows=[x for x in events if x['label']==label]
            output['optimizer_schedule_audit'][label]={'logged_updates':len(rows),
                'positive_lr_updates':sum(x['learning_rate_used']>0 for x in rows),
                'zero_lr_updates':sum(x['learning_rate_used']==0 for x in rows),
                'first_train_loss':rows[0]['loss'],'last_train_loss':rows[-1]['loss'],
                'max_preclip_gradient_norm':max(x['gradient_l2_preclip'] for x in rows),
                'note':'Last logged update may be incomplete stage; use checkpoint presence for completion.'}
    return output

if __name__=='__main__':
    report=ROOT/'summary/onevision_diagnostics_20261016'
    status=json.loads((report/'status.json').read_text());run=ROOT/status['run']
    from safetensors import safe_open
    norms={}
    with safe_open(str(ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/model.safetensors'),framework='pt',device='cpu') as f:
        for key in f.keys():
            if key.startswith('language_model.model.layers.') and key.endswith(('.q_proj.weight','.v_proj.weight')):
                norms[key[:-7]]=float(f.get_tensor(key).double().norm())
    result=audit(run,norms);target=report/'parameter_audit.json';tmp=target.with_suffix('.json.partial')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(target)
    print(json.dumps({'complete_adapter_snapshots':len(result['checkpoint_audits']),'report':str(target)}))
