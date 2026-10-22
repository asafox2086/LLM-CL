"""One fresh FIT-only original PASs reference; no official TEST access."""
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
from core import ROOT,PORT,REPORT,WORK,study,sha,write,digest,gradient_directory,save_first_raw_gradients,make_method,observed_optimizer,learner_digest,DurableRecords
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
    assert shutil.disk_usage(ROOT).free>40*1024**3,'Insufficient private stage-checkpoint/gradient storage before STARTED'
    assert not (REPORT/'STARTED.json').exists() and not WORK.exists()
    write(REPORT/'STARTED.json',dict(pid=os.getpid(),startticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),ready_sha256=sha(REPORT/'READY.json'),only_GPU=1))
    def expired(*args):raise TimeoutError('Amended v2 STARTED-to-COMPLETE 4h budget expired; no further automatic retry; preserve and stop')
    signal.signal(signal.SIGALRM,expired);signal.alarm(ready['worker_budget_seconds'])
    os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
    import torch,numpy as np
    from transformers import AutoProcessor
    sys.path.insert(0,str(PORT))
    spec=importlib.util.spec_from_file_location('pas_reference_formal_run',PORT/'run.py')
    formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal);sys.modules['run']=formal
    run=WORK;run.mkdir(parents=True);(run/'logs').mkdir();gradient_directory(run)
    monitor=formal.ResourceMonitor(run);monitor.snapshot('process_before_model')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    base=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    model=formal.onevision_bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
    formal.onevision_bridge.bind(model)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    method=make_method(model,[t['instruction'] for t in manifest['tasks']])
    method.state_root=run/'method_state';method.save_checkpoints=False
    assert method.name=='PASs_MoE' and len(method.layers)==168 and not method.learner._I_agg
    assert not hasattr(method,'learning_rate')
    layout={n:dict(numel=p.numel(),shape=list(p.shape),dtype=str(p.dtype)) for n,p in model.named_parameters() if p.requires_grad}
    assert layout and all(p['dtype']=='torch.float32' for p in layout.values())
    write(run/'LAYOUT.json',dict(parameters=layout,target_modules=len(method.layers),trainable_coordinates=sum(p['numel'] for p in layout.values())))
    frozen=[p for p in model.parameters() if not p.requires_grad];frozen_before=digest(frozen)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    method.vision_original=formal.install(model)
    config=dict(method='PASs_MoE',reference='fresh_FIT_only_for_ShortFixed_after_sealed_external_reboot',learning_rate=1e-4,seed=42,
        original_revision=method.original_revision,rank=128,alpha=256,dropout=.05,experts=2,gate_rank=16,
        gate_router='a',gate_tie='A_T',gate_tau_start=1.5,gate_tau_end=.7,lambda_stab=1e-4,B_reg_scale=.1,importance_top_p=.4,
        host_optimizer=True,optimizer_witness_durable_every_step=True,worker_budget_seconds=14400,conservative_cumulative_budget_seconds=21600,author_adapter_internal_FP32_and_CPU_offload=True,save_checkpoints_for_author=False,
        actual_learner_defaults={name:getattr(method.learner,name) for name in ['ema_decay','reg_warmup_steps','reg_warmup_ratio','importance_clip','reg_enable','random_importance']},
        manifest_sha256=ready['manifest_sha256'],split_sha256=ready['split_sha256'],
        trainable_coordinates=sum(p['numel'] for p in layout.values()),parameter_tensors=len(layout),target_modules=168,
        FIT_QA=1213,DEV_QA=100,TEST_QA=0,updates=87,epoch=1,batch_size=16,micro_batch_size=1,
        dtype='FP16 base/outer CUDA autocast; FP32 original PAS adapters internally autocast disabled',fresh_base=True,
        source_sha256=sha(__file__),ready_sha256=sha(REPORT/'READY.json'),
        checkpoint_payload_phase_note='Inherited payload after_stage_evaluation label historical: actual saves after each TRAIN stage, before sole final DEV')
    write(run/'config.json',config)
    model.eval();method.recording(False)
    formal.onevision_bridge.NATIVE_GATE_MODE=True
    gates=[formal.onevision_bridge.forward_gate(fit[t['task']][0],t,processor) for t in manifest['tasks']]
    formal.onevision_bridge.NATIVE_GATE_MODE=False
    assert not method.learner._I_agg and not method.learner._cur_I_ema and method.learner._cur_I_steps==0
    write(run/'NATIVE_GATE.json',dict(complete=True,gates=gates,FIT_QA=19,NLL_paths=38,short_generations=38,max_new_tokens=4,TEST_QA=0,usage_unpolluted=True))
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    monitor.snapshot('training_before',model=model)
    namespace=dict(vars(formal),model=model,method=method,processor=processor,manifest=manifest,data=mapped,
        run=run,monitor=monitor,args=SimpleNamespace(method='PASs_MoE'),resume_obj=None)
    prior=None
    def save_current(stage):
        nonlocal prior
        meta=formal.save_checkpoint(method,processor,run,stage,config);assert sha(run/meta['path'])==meta['sha256']
        if prior is not None:
            old=(run/prior['path']).resolve();assert old.is_relative_to((run/'checkpoints').resolve());old.unlink()
        prior=meta
    namespace['save_current']=save_current
    optimizer_records=DurableRecords(run/'OPTIMIZER_EVIDENCE.json')
    namespace['make_optimizer']=lambda m:observed_optimizer(m,formal.make_optimizer,optimizer_records)
    prefix,h=study.training_ast();assert h==ready['training_ast_sha256']
    node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_current(stage)').body,orelse=[])
    raw_clip=torch.nn.utils.clip_grad_norm_;counts={}
    def record_raw(parameters,*args,**kwargs):
        stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
        grads={n:dict(norm=float(p.grad.detach().float().norm()),finite=bool(torch.isfinite(p.grad).all())) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert grads and all(v['finite'] for v in grads.values()) and set(grads)<=set(layout)
        if counts[stage]==1:save_first_raw_gradients(model,run,stage)
        norm=raw_clip(parameters,*args,**kwargs);n=float(norm)
        assert __import__('math').isfinite(n)
        write(run/'gradients'/f'S{stage:02d}_{counts[stage]:03d}.json',dict(raw=grads,preclip_total_norm=n,clip_coefficient=min(1.,1./(n+1e-6))))
        return norm
    torch.nn.utils.clip_grad_norm_=record_raw
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<original-formal-TRAIN-prefix>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=raw_clip
    assert len(counts)==len(method.stage_records)==19 and len(optimizer_records)==optimizer_records.commits==87
    assert digest(frozen)==frozen_before
    write(run/'CALLBACK_EVIDENCE.json',method.stage_records);write(run/'OPTIMIZER_EVIDENCE.json',optimizer_records)
    checkpoint=json.loads((run/'checkpoints/latest.json').read_text())
    write(run/'TRAIN_COMPLETE.json',dict(complete=True,checkpoint=checkpoint,FIT_QA=1213,updates=87,TEST_QA=0))
    before=formal.parameter_digest([p for p in model.parameters() if p.requires_grad]);history_before=learner_digest(method)
    assert not (run/'evaluations').exists()
    formal.evaluate(method,processor,[t for t in manifest['tasks'] if dev[t['task']]],mapped,run,19,split='dev')
    after=formal.parameter_digest([p for p in model.parameters() if p.requires_grad]);history_after=learner_digest(method)
    assert before==after and history_before==history_after
    quality,_=read_predictions(run,manifest,dev);memory=monitor.finish(model)
    write(REPORT/'COMPLETE.json',dict(complete=True,quality=quality,config=config,checkpoint=checkpoint,
        optimizer_witness_commits=optimizer_records.commits,callback_records=method.stage_records,callback_evidence_sha256=sha(run/'CALLBACK_EVIDENCE.json'),
        optimizer_evidence_sha256=sha(run/'OPTIMIZER_EVIDENCE.json'),layout_sha256=sha(run/'LAYOUT.json'),
        frozen_base_digest_before=frozen_before,frozen_base_digest_after=digest(frozen),
        trainable_digest_before_DEV=before,trainable_digest_after_DEV=after,learner_digest_before_DEV=history_before,learner_digest_after_DEV=history_after,
        events_sha256=sha(run/'logs/events.jsonl'),gradient_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (run/'gradients').iterdir()},
        importance_snapshot_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (run/'method_state/I_snapshots').glob('*.pt')},
        config_sha256=sha(run/'config.json'),native_gate_sha256=sha(run/'NATIVE_GATE.json'),
        memory_summary=memory,ready_sha256=sha(REPORT/'READY.json'),source_sha256=sha(__file__),TEST_QA=0,
        one_reference_only=True,automatic_full_confirmation=False))
    signal.alarm(0)

if __name__=='__main__':
    try:main()
    except ResourceBusy as e:print(str(e),flush=True);sys.exit(75)
    except Exception:write(REPORT/'FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
