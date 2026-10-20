"""Independent v2: preserve original CPU autocast on folding; no GPU or TEST."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/cpu_function_space_20261020/v2'
PRIVATE=ROOT/'exp/cpu_function_space_20261020/v2'
TEACHER=ROOT/'exp/edge_teacher_transfer_20261019/onevision_osft_final'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(name,value):
    REPORT.mkdir(parents=True,exist_ok=True)
    p=REPORT/name;tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def selection():
    policy=ROOT/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json'
    info=json.loads(policy.read_text())['conditions']['pct01'];manifest=ROOT/info['manifest']
    assert sha(manifest)==info['manifest_sha256']
    m=json.loads(manifest.read_text());data={};sources={str(manifest.relative_to(ROOT)):sha(manifest)}
    for t in m['tasks']:
        f=t['files']['train'];p=ROOT/f['path'];assert sha(p)==f['sha256'];sources[str(p.relative_to(ROOT))]=f['sha256']
        for line in p.read_text().splitlines():
            x=json.loads(line);assert x['id'] not in data;data[x['id']]=(t,x)
    split=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
    s=json.loads(split.read_text());sources[str(split.relative_to(ROOT))]=sha(split)
    rows=[]
    for domain in ['natural','medical','geography','finance','traffic']:
        groups=[(k,v) for k,v in s['groups'].items() if v['domain']==domain and not v['dev']]
        groups.sort(key=lambda x:hashlib.sha256(('cpu-function-space42:'+x[0]).encode()).digest())
        assert len(groups)>=2
        for role,(gid,g) in zip(['calibration','probe'],groups[:2]):
            key=min(g['ids']);task,row=data[key]
            assert key in s['fit_ids'][str(task['task'])] and key not in s['dev_ids'][str(task['task'])]
            for rel in row['images']:sources[rel]=sha(ROOT/rel)
            rows.append({'role':role,'group':gid,'domain':domain,'task':task['task'],'id':key})
    assert len({x['group'] for x in rows})==10
    return rows,sources,data


def controls():
    import torch
    g=torch.Generator().manual_seed(42)
    # Calibration sees small-weight/high-activation direction; held probe shares it.
    d=torch.diag(torch.tensor([10.,1.],dtype=torch.float64))
    x=torch.tensor([[.01,.02],[10.,20.]],dtype=torch.float64)
    u,s,v=torch.linalg.svd(d,full_matrices=False);plain=(u[:,:1]*s[:1])@v[:1]
    q=torch.linalg.svd(d@x,full_matrices=False).U[:,:1];f=q@q.T@d
    assert float(((d-f)@x).square().sum())<float(((d-plain)@x).square().sum())
    assert torch.linalg.matrix_rank(f)==1
    return {'known_activation_counterexample':True,'rank_matched':True}


def prepare_function(source,config):
    from PIL import Image
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='input_batch')
    assert sha(source)==config['execution_source_sha256']['runtime_utils.py']
    cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Import) and any(x.name=='onevision_bridge' for x in n.names))
    fn.body=fn.body[:cut]+[ast.Return(ast.Name(id='batch',ctx=ast.Load()))]
    import retrieval
    retrieval.ROOT=ROOT
    env={'Image':Image,'math':math,'ROOT':ROOT,'retrieve':retrieval.retrieve}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),env)
    return env['input_batch']


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    ready=json.loads((REPORT/'READY.json').read_text());assert ready['source_sha256']==sha(__file__)
    assert ready['protocol_sha256']==sha(REPORT/'PROTOCOL.md')
    assert ready['inherited_protocol_sha256']==sha(REPORT.parent/'v1/PROTOCOL.md')
    assert ready['repair_protocol_sha256']==sha(REPORT/'REPAIR_PROTOCOL.md')
    gate=json.loads((REPORT/'repair_gate/COMPLETE.json').read_text())
    assert gate['status']=='CPU_CONTEXT_REPAIR_ALL168_EXACT' and gate['operators']==168
    assert gate['evidence_sha256']==sha(REPORT/'repair_gate/EVIDENCE.json')
    assert gate['source_sha256']==sha(ROOT/'analysis/cpu_function_context_gate_20261020.py')
    assert not (REPORT/'STARTED.json').exists()
    write('STARTED.json',{'pid':os.getpid(),'CPU_only':True});start=time.monotonic()
    selected,sources,data=selection();assert selected==ready['selected'] and sources==ready['input_source_sha256']
    cp=TEACHER/'checkpoints/stage_19.pt';assert sha(cp)=='f93481daa528bdca4cd6bdb68ca6ec522698dfaf4703cb277cf5548b19aba541'
    sys.path.insert(0,str(TEACHER/'runner_source'))
    obj=torch.load(cp,map_location='cpu',pickle_module=pickle,weights_only=False)
    model=obj['method'].model;processor=obj['processor'];model.eval()
    import onevision_bridge,frozen_vision,runtime_utils
    frozen_vision.ROOT=ROOT;runtime_utils.ROOT=ROOT;onevision_bridge.bind(model)
    # Replace saved cache/closure with a fresh CPU-only empty official cache.
    frozen_vision.install(model)
    cfg=obj['config'];prepare=prepare_function(TEACHER/'runner_source/runtime_utils.py',cfg)
    first=next(x for x in selected if x['role']=='calibration' and x['domain']=='medical');task,row=data[first['id']]
    # Native bridge packing retains full visual positions; use original manual loss.
    def batch_for(task,row):
        b=prepare(row,task,processor,'cpu')
        b,_=onevision_bridge.pack_native_batch(b,row)
        return b
    b=batch_for(task,row)
    with torch.inference_mode(),torch.autocast('cpu',dtype=torch.float16):
        original_loss,_=runtime_utils.answer_loss(model,b,row['answer'],processor)
    assert torch.isfinite(original_loss)
    changes={};base_path=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/model.safetensors'
    with safe_open(base_path,framework='pt',device='cpu') as base,torch.inference_mode():
        for name,m in list(model.core.named_modules()):
            if type(m).__name__!='OSFTLinear':continue
            with torch.autocast('cpu',dtype=torch.float16):
                w=m.kernel['reconstruct_weight_matrix'](m.svd(),torch.float32,m.output_dtype).detach()
            if name.endswith(('self_attn.q_proj','mlp.down_proj')):
                delta=w.double()-base.get_tensor(name+'.weight').double()
                u,s,v=torch.linalg.svd(delta,full_matrices=False)
                changes[name]={'delta':delta,'weight_U8':u[:,:8],'cal_Y':[]}
            linear=torch.nn.Linear(w.shape[1],w.shape[0],bias=m.bias is not None,dtype=w.dtype)
            linear.weight=torch.nn.Parameter(w,requires_grad=False)
            if m.bias is not None:linear.bias=torch.nn.Parameter(m.bias.detach(),requires_grad=False)
            parent,child=name.rsplit('.',1);setattr(model.core.get_submodule(parent),child,linear)
    assert len(changes)==48
    for p in model.parameters():p.requires_grad_(False)
    state={'row':None,'begin':None,'end':None,'observations':None}
    handles=[]
    def make_hook(name):
        def hook(module,args):
            if state['row'] is None:return
            x=args[0][0,state['begin']:state['end']].detach().double().T
            assert x.shape[1]>0 and torch.isfinite(x).all()
            q=changes[name];y=q['delta']@x;plain=q['weight_U8']@(q['weight_U8'].T@y)
            observation={'name':name,'tokens':x.shape[1],'reference_energy':float(y.square().sum()),
                         'weight_rank8_error':float((y-plain).square().sum())}
            if state['row']['role']=='calibration':q['cal_Y'].append(y.clone())
            else:
                u=q['function_U8'];adapted=u@(u.T@y)
                observation['function_rank8_error']=float((y-adapted).square().sum())
            state['observations'].append(observation)
        return hook
    for name in changes:handles.append(model.core.get_submodule(name).register_forward_pre_hook(make_hook(name)))
    records=[];cache_batches={first['id']:b};calibrators=None
    # All five calibration inputs before fitting, then all five probes.
    ordered=[first]+[x for x in selected if x['role']=='calibration' and x['id']!=first['id']]+[x for x in selected if x['role']=='probe']
    before_hash=None
    def parameter_hash():
        h=hashlib.sha256()
        for name,p in model.core.named_parameters():
            h.update(name.encode());h.update(p.detach().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    before_hash=parameter_hash()
    for index,item in enumerate(ordered):
        assert time.monotonic()-start<7200,'CPU wallclock budget: INCOMPLETE, no partial conclusion'
        if index==5:
            calibrators=[]
            for name,q in changes.items():
                y=torch.cat(q['cal_Y'],dim=1);u,s,v=torch.linalg.svd(y,full_matrices=False)
                assert u.shape[1]>=8,'Insufficient calibration dimensions'
                q['function_U8']=u[:,:8];calibrators.append({'name':name,'calibration_tokens':y.shape[1],
                    'calibration_energy':float(y.square().sum()),'function_rank8_error':float((y-u[:,:8]@(u[:,:8].T@y)).square().sum())})
                q['cal_Y']=[]
        task,row=data[item['id']];batch=cache_batches.pop(item['id']) if item['id'] in cache_batches else batch_for(task,row)
        labels=processor.tokenizer(row['answer'],add_special_tokens=False,truncation=True,max_length=512)['input_ids']
        assert labels;count=len(labels)+1;prompt=batch['input_ids'].shape[1]
        state.update(row=item,begin=prompt-1,end=prompt+count-1,observations=[])
        tick=time.monotonic()
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.float16):
            loss,n=runtime_utils.answer_loss(model,batch,row['answer'],processor)
        assert n==count and torch.isfinite(loss)
        if index==0:
            write('CPU_FOLD_GATE.json',{'id':item['id'],'original_NLL':float(original_loss),
                 'folded_NLL':float(loss),'exact':torch.equal(loss,original_loss),'CPU_only':True})
            assert torch.equal(loss,original_loss),'CPU folded model gate differs'
        assert len(state['observations'])==48 and len({x['name'] for x in state['observations']})==48
        records.append(dict(item,NLL=float(loss),target_tokens=count,seconds=time.monotonic()-tick,observations=state['observations']))
        state['row']=None
        write('PROGRESS.json',{'finished_QA':len(records),'required_QA':10,'last_id':item['id'],'seconds':time.monotonic()-start})
        print('QA_COMPLETE',len(records),item['role'],item['domain'],item['id'],flush=True)
    for h in handles:h.remove()
    assert before_hash==parameter_hash() and not torch.cuda.is_initialized()
    for rel,h in sources.items():assert sha(ROOT/rel)==h
    evidence={'CPU_only':True,'official_TEST_read':False,'DEV_answers_used':False,
              'TRAIN_files_contain_DEV_rows':True,'model_forward_calls':11,
              'new_optimizer_updates':0,'source_sha256':sha(__file__),'protocol_sha256':sha(REPORT/'PROTOCOL.md'),
              'checkpoint_sha256':sha(cp),'base_sha256':sha(base_path),'input_source_sha256':sources,
              'repair_gate_evidence_sha256':gate['evidence_sha256'],
              'fold_reconstruction_context':'same CPU FP16 autocast as original forward',
              'first_CPU_original_folded_NLL_exact':True,'model_parameter_SHA_unchanged':before_hash,
              'records':records,'calibrators':calibrators,'seconds':time.monotonic()-start,
              'teacher_has_trained_on_all_probe_QA':True,'functional_probe_not_AnswerF1_or_unseen_task_generalization':True,
              'not_a_deployed_compressed_model':True,'not_original_SVD_LLM_reproduction':True}
    write('EVIDENCE.json',evidence)
    write('COMPLETE.json',{'status':'CPU_TRAIN_FUNCTION_SPACE_COMPLETE','evidence_sha256':sha(REPORT/'EVIDENCE.json'),
                           'source_sha256':sha(__file__),'QA':10,'matrices_per_QA':48})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controls',action='store_true');p.add_argument('--select',action='store_true');a=p.parse_args()
    try:
        if a.controls:
            import torch
            torch.set_num_threads(2);print(json.dumps(controls()))
        elif a.select:
            s,h,_=selection();print(json.dumps({'selected':s,'input_source_sha256':h}))
        else:main()
    except Exception:
        write('FAILURE.json',{'source_sha256':sha(__file__),'traceback':traceback.format_exc(),'no_blind_retry':True})
        raise
