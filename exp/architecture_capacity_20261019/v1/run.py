"""Independent GPU1 capacity experiment; original baseline training AST retained.

This is an authored-mechanism ablation, not an Ours result. No TEST is loaded
during training. Every declared condition retains its complete final test.
"""
import argparse
import ast
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

from capacity_core import ROOT, PORT, CASES, make_method, parameter_inventory

REPORT = ROOT/'summary/edge_deployment_20261019/architecture_capacity/v1'
WORK = ROOT/'exp/architecture_capacity_20261019/v1/private'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def dataset(include_test=False):
    policy=json.loads((PORT/'execution_policy.json').read_text())
    meta=policy['conditions']['pct01'];path=ROOT/meta['manifest']
    assert sha(path)==meta['manifest_sha256']
    manifest=json.loads(path.read_text());assert len(manifest['tasks'])==19 and manifest['seed']==42
    data={}
    for task in manifest['tasks']:
        for split in ['train','test'] if include_test else ['train']:
            f=task['files'][split];p=ROOT/f['path'];assert sha(p)==f['sha256']
            rows=[json.loads(s) for s in p.read_text().splitlines()]
            assert len(rows)==f['count'] and len({r['id'] for r in rows})==len(rows)
            data[task['task'],split]=rows
    assert sum(len(data[t['task'],'train']) for t in manifest['tasks'])==1313
    return path,manifest,data


def training_ast():
    """Execute exactly the original per-stage TRAIN body, before TEST I/O."""
    text=(PORT/'run.py').read_text();tree=ast.parse(text)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='stage')
    cutoff=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='wanted' for t in n.targets))
    # The immediately preceding statement only marks the start of evaluation.
    prefix=copy.deepcopy(loop.body[:cutoff-1])
    assert isinstance(prefix[-1],ast.If)  # final training_after resource snapshot
    canonical=ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False)
    return prefix,hashlib.sha256(canonical.encode()).hexdigest()


def prepare():
    from capacity_core import cpu_gate
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    REPORT.mkdir(parents=True,exist_ok=True)
    # This gate is engineering admission; its results never enter scientific tables.
    gate_path=REPORT/'CPU_GATE.json'
    if not gate_path.exists():write(gate_path,cpu_gate())
    gate=json.loads(gate_path.read_text());assert len(gate['six_capacity_controls'])==6
    assert gate.get('source_sha256',sha(Path(__file__).parent/'capacity_core.py'))==sha(Path(__file__).parent/'capacity_core.py')
    path,manifest,data=dataset(False)
    body,digest=training_ast()
    sources={str(p.relative_to(ROOT)):sha(p) for p in PORT.glob('*.py')}
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
    sources[str((PORT/'execution_policy.json').relative_to(ROOT))]=sha(PORT/'execution_policy.json')
    base=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    base_files={str(p.relative_to(ROOT)):sha(p) for p in base.glob('*.safetensors')}
    assert base_files
    ready=dict(sources=sources,base_files=base_files,protocol_sha256=sha(REPORT/'PROTOCOL.md'),
        CPU_GATE_sha256=sha(gate_path),manifest_sha256=sha(path),cases=CASES,
        train_body_AST_sha256=digest,TRAIN_QA=1313,stages=19,updates=93,
        model_training_started=False,only_physical_GPU=1,TEST_not_loaded_during_training=True,
        all_six_final_21454_required=True,not_a_new_algorithm=True)
    if (REPORT/'READY.json').exists():assert json.loads((REPORT/'READY.json').read_text())==ready
    else:write(REPORT/'READY.json',ready)


def startup(case,phase):
    sys.path.insert(0,str(ROOT/'analysis'))
    import onevision_research_device_entry as device
    cards=device.inventory();ours={g for g,c in cards.items() if any(o['uid']==os.getuid() for o in c['owners'])}
    assert not (ours & {0,1}) and len(ours|{1})<=3
    assert not cards[1]['owners'] and cards[1]['memory_MiB']<500 and cards[1]['utilization']<10
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
    ready=json.loads((REPORT/'READY.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in ready['sources'].items())
    assert all(sha(ROOT/p)==h for p,h in ready['base_files'].items())
    assert sha(REPORT/'PROTOCOL.md')==ready['protocol_sha256']
    assert sha(REPORT/'CPU_GATE.json')==ready['CPU_GATE_sha256']
    marker=REPORT/(case+'_'+phase+'_STARTED.json')
    assert not marker.exists(),'Never blindly retry an attempted scientific run'
    write(marker,dict(pid=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21],
        inventory=cards,ready_sha256=sha(REPORT/'READY.json')))
    import torch,numpy as np
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    sys.path.insert(0,str(PORT))
    import importlib.util
    spec=importlib.util.spec_from_file_location('capacity_frozen_formal_run',PORT/'run.py')
    formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal)
    # NativeIO imports `run.input_batch`; bind only in this independent process.
    sys.modules['run']=formal
    path,manifest,data=dataset(phase=='full')
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV',local_files_only=True)
    dtype=torch.bfloat16 if CASES[case]['family']=='SMoLoRA' else torch.float16
    model=formal.onevision_bridge.Factory.from_pretrained(ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV',
        torch_dtype=dtype,attn_implementation='sdpa',local_files_only=True).cuda()
    formal.onevision_bridge.bind(model)
    method=make_method(model,case,[t['instruction'] for t in manifest['tasks']])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    run=WORK/case;run.mkdir(parents=True,exist_ok=True);(run/'logs').mkdir(exist_ok=True)
    if hasattr(method,'bind_context'):method.bind_context(processor,manifest,data,run,False)
    method.vision_original=formal.install(model)
    formal.onevision_bridge.NATIVE_GATE_MODE=False
    return formal,model,method,processor,manifest,data,run


def auxiliary(method):
    if method.name!='CL_MoE':return dict(embedding=method.embedding)
    return dict(anchors=method.anchors,feature_cache=method.feature_cache,
        layers=[dict(seen=l.seen,task_weights=l.task_weights,global_weight=l.global_weight) for l in method.layers])


def adapter_weights(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters()
        if any(s in n for s in ['.lora_', '.A.loraA.', '.B.loraB.', '.router.'])}


def save_stage(method,stage,run):
    import torch
    destination=run/'states'/f'stage_{stage:02d}.pt';destination.parent.mkdir(exist_ok=True)
    temp=destination.with_suffix('.pt.tmp')
    torch.save(dict(stage=stage,weights=adapter_weights(method.model),auxiliary=auxiliary(method),
        capacity=method.capacity_spec,ready_sha256=sha(REPORT/'READY.json')),temp)
    with temp.open('rb') as f:os.fsync(f.fileno())
    temp.replace(destination)
    meta=dict(stage=stage,path=str(destination.relative_to(ROOT)),sha256=sha(destination),bytes=destination.stat().st_size)
    pointer=run/'latest.json';prior=json.loads(pointer.read_text()) if pointer.exists() else None
    write(pointer,meta)
    if prior:
        old=(ROOT/prior['path']).resolve();assert old.is_relative_to((run/'states').resolve())
        if old!=destination:old.unlink()
    return meta


def train_probe(formal,model,method,processor,manifest,data):
    import torch
    model.eval();records=[]
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        for task in manifest['tasks']:
            row=min(data[task['task'],'train'],key=lambda r:hashlib.sha256(('42:capacity-probe:'+r['id']).encode()).digest())
            method.set_instruction(task['task'])
            batch,_=formal.input_batch(row,task,processor,model.device)
            formal.prepare(batch,row,model,method.vision_original)
            if hasattr(method,'prepare_row'):method.prepare_row(row,batch)
            with torch.autocast('cuda',dtype=model.dtype):loss,_=formal.answer_loss(model,batch,row['answer'],processor)
            assert torch.isfinite(loss)
            records.append(dict(id=row['id'],NLL=float(loss)))
    return records


def train(case):
    formal,model,method,processor,manifest,data,run=startup(case,'train')
    import torch
    from transformers import get_cosine_schedule_with_warmup
    monitor=formal.ResourceMonitor(run);monitor.snapshot('model_loaded_before_initial_evaluation',model=model)
    write(run/'config.json',dict(case=case,capacity=method.capacity_spec,seed=42,epochs=1,TRAIN_QA=1313,
        TEST_QA=0,ready_sha256=sha(REPORT/'READY.json'),framework='frozen_formal_v2_TRAIN_AST',
        complete_FWT_BWT=None,inventory=parameter_inventory(model)))
    formal.onevision_bridge.NATIVE_GATE_MODE=True
    gates=[]
    for task in manifest['tasks']:
        row=data[task['task'],'train'][0]
        method.set_instruction(task['task'])
        gates.append(formal.onevision_bridge.forward_gate(row,task,processor))
        write(run/'NATIVE_GATE_partial.json',dict(complete=False,cases=gates))
    formal.onevision_bridge.NATIVE_GATE_MODE=False
    write(run/'NATIVE_GATE.json',dict(complete=True,cases=gates,TRAIN_QA=19,TEST_QA=0))
    # Keep initial engineering examples separate from the 1313 main TRAIN QA.
    torch.cuda.empty_cache();monitor.snapshot('training_before',model=model)
    save_stage(method,0,run)
    namespace=dict(vars(formal),model=model,method=method,processor=processor,manifest=manifest,data=data,
        run=run,monitor=monitor,args=SimpleNamespace(method=CASES[case]['family']),resume_obj=None,
        save_stage=lambda m,s,r:save_stage(m,s,r))
    prefix,digest=training_ast();assert digest==json.loads((REPORT/'READY.json').read_text())['train_body_AST_sha256']
    # Raw gradient logging is outside the authored update rule; no tensor is changed.
    clip=torch.nn.utils.clip_grad_norm_;steps={}
    def record_raw(parameters,*args,**kwargs):
        stage=monitor.stage;steps[stage]=steps.get(stage,0)+1
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        values={n:dict(norm=float(p.grad.detach().float().norm()),finite=bool(torch.isfinite(p.grad).all()))
                for n,p in named if p.grad is not None}
        assert values and all(v['finite'] for v in values.values())
        write(run/'gradients'/f'S{stage:02d}_step{steps[stage]:03d}.json',dict(stage=stage,step=steps[stage],
            raw_statistics=values,missing=[n for n,p in named if p.grad is None]))
        if steps[stage]==1:
            path=run/'gradients'/f'S{stage:02d}_first_raw.pt'
            torch.save({n:p.grad.detach().cpu().clone() for n,p in named if p.grad is not None},path)
        return clip(parameters,*args,**kwargs)
    torch.nn.utils.clip_grad_norm_=record_raw
    try:
        tail=ast.parse("save_stage(method,stage,run)\nwrite(run/'run_status.json',dict(status='stage_complete',completed_stage=stage))").body
        node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+tail,orelse=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<exact-formal-TRAIN-prefix>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=clip
    summary=monitor.finish(model)
    monitor.snapshot('final_TRAIN_reload_probe_before',19,model)
    torch.cuda.reset_peak_memory_stats()
    probe=train_probe(formal,model,method,processor,manifest,data)
    monitor.snapshot('final_TRAIN_reload_probe_after',19,model)
    write(run/'FINAL_TRAIN_RELOAD_REFERENCE.json',dict(records=probe,TEST_QA=0,not_TEST_quality=True))
    events=[json.loads(s) for s in (run/'logs/events.jsonl').read_text().splitlines()]
    assert len(events)==93 and all(any(lr>0 for lr in e['learning_rate_used']) for e in events)
    assert sum(e['total_qa'] if e['step']==1 else 0 for e in events)==1313
    write(REPORT/(case+'_train_COMPLETE.json'),dict(complete=True,case=case,TRAIN_QA=1313,updates=93,
        stages=19,checkpoint=json.loads((run/'latest.json').read_text()),source_sha256=sha(__file__),
        ready_sha256=sha(REPORT/'READY.json'),memory_summary=summary,not_TEST_quality=True,
        reload_reference_sha256=sha(run/'FINAL_TRAIN_RELOAD_REFERENCE.json'),
        full_21454_still_required=True))


def full(case):
    proof=json.loads((REPORT/(case+'_train_COMPLETE.json')).read_text());assert proof['complete']
    formal,model,method,processor,manifest,data,run=startup(case,'full')
    import torch
    meta=proof['checkpoint'];path=ROOT/meta['path'];assert sha(path)==meta['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    own=dict(model.named_parameters())
    with torch.no_grad():
        for name,p in saved['weights'].items():own[name].copy_(p.to(own[name].device))
    assert all(torch.equal(own[n].detach().cpu(),v) for n,v in saved['weights'].items())
    if method.name=='CL_MoE':
        method.anchors=saved['auxiliary']['anchors'];method.feature_cache=saved['auxiliary']['feature_cache']
        for layer,aux in zip(method.layers,saved['auxiliary']['layers']):
            layer.seen=aux['seen'];layer.task_weights=aux['task_weights'];layer.global_weight=aux['global_weight']
    else:assert method.embedding==saved['auxiliary']['embedding']
    reference_path=run/'FINAL_TRAIN_RELOAD_REFERENCE.json'
    assert sha(reference_path)==proof['reload_reference_sha256']
    got=train_probe(formal,model,method,processor,manifest,data)
    assert got==json.loads(reference_path.read_text())['records'],'Fresh checkpoint NLL differs; retain evidence'
    write(run/'FRESH_RELOAD.json',dict(complete=True,all_adapter_parameters_exact=True,
        nineteen_TRAIN_NLL_exact=True,TEST_QA=0,checkpoint_sha256=meta['sha256']))
    assert sum(len(data[t['task'],'test']) for t in manifest['tasks'])==21454
    # Outputs remain independent of original formal publication directories.
    formal.evaluate(method,processor,manifest['tasks'],data,run,19)
    rows=[];domains={}
    for task in manifest['tasks']:
        p=run/'evaluations/stage_19'/f"T{task['task']:02d}_test_predictions.jsonl"
        records=[json.loads(s) for s in p.read_text().splitlines()]
        assert [r['id'] for r in records]==[r['id'] for r in data[task['task'],'test']]
        scores=[]
        for r,gold in zip(records,data[task['task'],'test']):
            value=formal.__dict__['evaluate'].__globals__['score'](r['prediction'],gold['answers'])['token_f1']
            assert value==r['scores']['token_f1'];scores.append(value)
        rows.append(sum(scores)/len(scores));domains.setdefault(task['domain'],[]).extend(scores)
    result={k:dict(F1=sum(v)/len(v),QA=len(v)) for k,v in domains.items()}
    write(REPORT/(case+'_full_COMPLETE.json'),dict(complete=True,case=case,TEST_QA=21454,
        AP19=sum(rows)/19,domains=result,domain_macro=sum(v['F1'] for v in result.values())/len(result),
        FWT=None,BWT=None,checkpoint_sha256=meta['sha256'],source_sha256=sha(__file__),
        ready_sha256=sha(REPORT/'READY.json'),deployment_NOT_measured=True,no_formal_Ours_publication=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--case',choices=CASES)
    ap.add_argument('--phase',choices=['train','full']);args=ap.parse_args()
    try:
        if args.prepare:prepare()
        elif args.phase=='train':train(args.case)
        else:full(args.case)
    except Exception:
        write(REPORT/((args.case or 'admission')+'_'+(args.phase or 'CPU')+'_FAILURE.json'),
              dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
