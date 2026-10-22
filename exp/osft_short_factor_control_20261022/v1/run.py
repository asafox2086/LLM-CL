"""One fresh OSFT short-factor FIT/DEV condition; official TEST is never loaded."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import traceback
from types import SimpleNamespace
from core import ROOT,PORT,REPORT,WORK,study,sha,write,ShortFixedOSFT,measured_optimizer,digest
from common import check_ready,data,read_predictions
class ResourceBusy(RuntimeError):pass

def main():
    ready=check_ready()
    path,manifest,_,split,fit,dev,mapped,plan=data()
    assert sha(path)==ready['manifest_sha256'] and plan==ready['training_plan']
    sys.path.insert(0,str(ROOT/'analysis'));import onevision_research_device_entry as device
    cards=device.inventory();ours={g for g,v in cards.items() if any(p['uid']==os.getuid() for p in v['owners'])}
    if cards[1]['owners'] or cards[1]['memory_MiB']>=500 or cards[1]['utilization']>=10 or (ours&{0,1}) or len(ours|{1})>3:
        raise ResourceBusy('Resource changed before STARTED/CUDA; unattempted healthy wait')
    import shutil
    assert shutil.disk_usage(ROOT).free>32*1024**3,'Insufficient private checkpoint/gradient storage; no STARTED'
    assert not (REPORT/'STARTED.json').exists() and not WORK.exists()
    write(REPORT/'STARTED.json',dict(pid=os.getpid(),startticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),ready_sha256=sha(REPORT/'READY.json'),only_GPU=1))
    def budget_expired(*args):raise TimeoutError('Frozen 2h actual-model budget expired; preserve and stop')
    signal.signal(signal.SIGALRM,budget_expired);signal.alarm(7200)
    os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
    import torch,numpy as np
    from transformers import AutoProcessor
    sys.path.insert(0,str(PORT))
    spec=importlib.util.spec_from_file_location('short_factor_control_formal_run',PORT/'run.py')
    formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal);sys.modules['run']=formal
    run=WORK;run.mkdir(parents=True);(run/'logs').mkdir()
    monitor=formal.ResourceMonitor(run);monitor.snapshot('process_before_model')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    base=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    model=formal.onevision_bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
    formal.onevision_bridge.bind(model)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    method=ShortFixedOSFT(model)
    layout=method.layout()
    assert layout['trainable_coordinates']==89485056 and layout['frozen_short_coordinates']==24281088
    assert layout['S_low_coordinates']==28416 and len(method.layers)==168
    write(run/'LAYOUT.json',layout)
    low_ids={id(getattr(l,n)) for l in method.layers for n in ['U_low','S_low','V_low']}
    frozen_base=[p for p in model.parameters() if id(p) not in low_ids]
    assert all(not p.requires_grad for p in frozen_base)
    frozen_before=digest(frozen_base)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    method.vision_original=formal.install(model)
    config=dict(method='OSFT_ShortFixed',learning_rate=1e-5,unfreeze_rank_ratio=.25,
        short_side_frozen=True,host_optimizer=method.host_optimizer,
        offload_activations=method.offload_activations,original_method_name=method.name,
        manifest_sha256=ready['manifest_sha256'],split_sha256=ready['split_sha256'],
        trainable_coordinates=89485056,target_modules=168,
        FIT_QA=1213,DEV_QA=100,TEST_QA=0,updates=87,seed=42,
        dtype='FP16 base/outer CUDA autocast; original FP32 USV',fresh_base=True,
        checkpoint_payload_phase_note='Inherited payload phase label is historical; actual saves occur after each TRAIN stage and before the sole final DEV',
        source_sha256=sha(__file__),ready_sha256=sha(REPORT/'READY.json'),
        same_single_stage_ideal_set_not_same_finite_optimizer_trajectory=True)
    write(run/'config.json',config)
    formal.onevision_bridge.NATIVE_GATE_MODE=True
    gates=[formal.onevision_bridge.forward_gate(fit[t['task']][0],t,processor) for t in manifest['tasks']]
    formal.onevision_bridge.NATIVE_GATE_MODE=False
    write(run/'NATIVE_GATE.json',dict(complete=True,gates=gates,FIT_QA=19,NLL_paths=38,short_generations=38,max_new_tokens=4,TEST_QA=0))
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    monitor.snapshot('training_before',model=model)
    namespace=dict(vars(formal),model=model,method=method,processor=processor,manifest=manifest,data=mapped,
                   run=run,monitor=monitor,args=SimpleNamespace(method=method.name),resume_obj=None)
    prior=None
    def save_current(stage):
        nonlocal prior
        meta=formal.save_checkpoint(method,processor,run,stage,config);assert sha(run/meta['path'])==meta['sha256']
        if prior is not None:
            old=(run/prior['path']).resolve();assert old.is_relative_to((run/'checkpoints').resolve());old.unlink()
        prior=meta
    namespace['save_current']=save_current
    optimizer_records=[]
    namespace['make_optimizer']=lambda m: measured_optimizer(m,formal.make_optimizer,optimizer_records.append)
    prefix,h=study.training_ast();assert h==ready['training_ast_sha256']
    node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_current(stage)').body,orelse=[])
    raw_clip=torch.nn.utils.clip_grad_norm_;counts={}
    def record_raw(parameters,*args,**kwargs):
        stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
        grads={n:dict(norm=float(p.grad.detach().float().norm()),finite=bool(torch.isfinite(p.grad).all())) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert grads and all(v['finite'] for v in grads.values())
        if counts[stage]==1:torch.save({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},run/'gradients'/f'S{stage:02d}_first_raw.pt')
        if counts[stage]==1:method.observe_before_first_step()
        norm=raw_clip(parameters,*args,**kwargs)
        n=float(norm);assert __import__('math').isfinite(n)
        write(run/'gradients'/f'S{stage:02d}_{counts[stage]:03d}.json',dict(raw=grads,
            preclip_total_norm=n,clip_coefficient=min(1.,1./(n+1e-6))))
        return norm
    torch.nn.utils.clip_grad_norm_=record_raw
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<original-formal-TRAIN-prefix>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=raw_clip
    assert len(counts)==19
    assert len(method.stage_records)==19 and len(optimizer_records)==87
    frozen_after=digest(frozen_base);assert frozen_before==frozen_after
    write(run/'SHORT_STAGE_EVIDENCE.json',method.stage_records)
    write(run/'OPTIMIZER_EVIDENCE.json',optimizer_records)
    checkpoint=json.loads((run/'checkpoints/latest.json').read_text())
    write(run/'TRAIN_COMPLETE.json',dict(complete=True,checkpoint=checkpoint,FIT_QA=1213,updates=87,TEST_QA=0,short_stage_records=method.stage_records))
    before=formal.parameter_digest([p for p in model.parameters() if p.requires_grad])
    assert not (run/'evaluations').exists()
    formal.evaluate(method,processor,[t for t in manifest['tasks'] if dev[t['task']]],mapped,run,19,split='dev')
    after=formal.parameter_digest([p for p in model.parameters() if p.requires_grad]);assert before==after
    quality,_=read_predictions(run,manifest,dev)
    memory=monitor.finish(model)
    write(REPORT/'COMPLETE.json',dict(complete=True,quality=quality,config=config,checkpoint=checkpoint,
        short_stage_records=method.stage_records,
        frozen_base_digest_before=frozen_before,frozen_base_digest_after=frozen_after,
        short_stage_evidence_sha256=sha(run/'SHORT_STAGE_EVIDENCE.json'),
        optimizer_evidence_sha256=sha(run/'OPTIMIZER_EVIDENCE.json'),layout_sha256=sha(run/'LAYOUT.json'),
        trainable_digest_before_DEV=before,trainable_digest_after_DEV=after,
        events_sha256=sha(run/'logs/events.jsonl'),gradient_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (run/'gradients').iterdir()},
        config_sha256=sha(run/'config.json'),native_gate_sha256=sha(run/'NATIVE_GATE.json'),
        memory_summary=memory,ready_sha256=sha(REPORT/'READY.json'),source_sha256=sha(__file__),TEST_QA=0,
        mechanism_only_no_full_candidates=True,not_new_algorithm_or_deployment_advantage=True))
    signal.alarm(0)
if __name__=='__main__':
    try:main()
    except ResourceBusy as e:print(str(e),flush=True);sys.exit(75)
    except Exception:write(REPORT/'FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
