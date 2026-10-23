"""One complete Spectral or Answer flow; never official TEST or partial resume."""
import argparse,ast,gc,importlib.util,json,os,random,signal,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
from core import ROOT,PORT,REPORT,WORK,CASES,BASE,FINAL,TRAINABLE,RANK,EXTRA,study,sha,write,compress,Renewal,layout,digest,consolidate,save_gradient,measured_optimizer
from common import check_ready,data,read_predictions
class ResourceBusy(RuntimeError):pass

def main(case):
    ready=check_ready();path,manifest,split,fit,dev,mapped,plan,groups=data()
    assert sha(path)==ready['manifest_sha256'] and plan==ready['training_plan']
    case_report=REPORT/'cases'/case;run=WORK/case
    sys.path.insert(0,str(ROOT/'analysis'));import onevision_research_device_entry as device
    cards=device.inventory();ours={g for g,c in cards.items() if any(p['uid']==os.getuid() for p in c['owners'])}
    if cards[1]['owners'] or cards[1]['memory_MiB']>=500 or cards[1]['utilization']>=10 or ours&{0,1} or len(ours|{1})>3:
        raise ResourceBusy('Resource changed before STARTED/CUDA: healthy unattempted wait')
    import shutil
    assert shutil.disk_usage(ROOT).free>32*1024**3
    assert not run.exists() and not any((case_report/n).exists() for n in ('STARTED.json','FAILURE.json','COMPLETE.json'))
    if case=='Answer':assert json.loads((REPORT/'cases/Spectral/COMPLETE.json').read_text())['complete']
    worker_begin=time.monotonic()
    write(case_report/'STARTED.json',dict(case=case,pid=os.getpid(),
        startticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        ready_sha256=sha(REPORT/'READY.json'),only_GPU=1,UTC=time.time()))
    def expired(*args):raise TimeoutError('Frozen per-flow 7200-second total model budget expired; stop entire pilot')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,7200-(time.monotonic()-worker_begin)))
    os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
    import torch,numpy as np
    from transformers import AutoProcessor
    sys.path.insert(0,str(PORT))
    spec=importlib.util.spec_from_file_location('answer_renewal_formal_run',PORT/'run.py')
    formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal);sys.modules['run']=formal
    run.mkdir(parents=True);(run/'logs').mkdir();(run/'gradients').mkdir()
    monitor=formal.ResourceMonitor(run);monitor.snapshot('process_before_model')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    config=dict(method='CompactRenewal',case=case,manifest_sha256=ready['manifest_sha256'],split_sha256=ready['split_sha256'],
        rank=RANK,temporary_rank=EXTRA,final_coefficients=FINAL,base_coefficients=BASE,trainable_coordinates=TRAINABLE,
        FIT_QA=1213,DEV_QA=100,TEST_QA=0,updates=87,learning_rate=1e-4,scale=4,dropout=.1,seed=42,
        worker_budget_seconds=7200,aggregate_model_budget_seconds=14400,
        dtype='FP16 compact/full base and outer CUDA autocast; FP32 temporary factors, gates, QR/SVD',
        temporary_initialization='A=0 B=Uniform[-1/sqrt(n),1/sqrt(n)] local CPU stage/name generator',
        host_optimizer=True,offload_activations=True,source_sha256=sha(__file__),ready_sha256=sha(REPORT/'READY.json'),
        checkpoint_payload_phase_note='after consolidation, not after TEST; initial checkpoint is untrained',
        memory_phase_note='original training_after is before final consolidation, retaining temporary A/B; use consolidation_after_saved for compact stage boundary')
    write(run/'config.json',config)
    if case=='Spectral':
        base=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV';processor=AutoProcessor.from_pretrained(base,local_files_only=True)
        model=formal.onevision_bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
        formal.onevision_bridge.bind(model)
        tick=time.monotonic();records=compress(model)
        assert len(records)==72 and all(sorted(r['shape'])==[896,4864] for r in records)
        assert sum(p.numel() for p in model.parameters())==FINAL
        method=Renewal(model);method.vision_original=formal.install(model)
        init_root=WORK/'shared_init';init_root.mkdir()
        meta=formal.save_checkpoint(method,processor,init_root,0,config)
        initial=dict(checkpoint=meta,layout=layout(model),weights_digest=digest(model.parameters()),
            compression_records=records,shared_initialization_seconds_including_save=time.monotonic()-tick,untrained=True,
            no_teacher=True,no_QA_or_input_calibration=True)
        write(WORK/'INITIAL.json',initial)
    else:
        initial=json.loads((WORK/'INITIAL.json').read_text());assert initial['untrained']
        restored=formal.load_checkpoint(WORK/'shared_init');assert restored['stage']==0
        method=restored['method'];model=method.model;processor=restored['processor'];del restored
        assert layout(model)==initial['layout'] and digest(model.parameters())==initial['weights_digest']
        assert not method.bank and not method.current
        formal.onevision_bridge.bind(model)
    write(run/'INITIAL_BINDING.json',dict(initial_sha256=sha(WORK/'INITIAL.json'),weights_digest=initial['weights_digest'],
        checkpoint_sha256=initial['checkpoint']['sha256'],untrained=True))
    assert layout(model)['coefficients']==FINAL and layout(model)['trainable']==0
    write(run/'INITIAL_LAYOUT.json',layout(model))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    stable=[p for n,p in model.named_parameters() if not any(f'.mlp.{x}.' in n for x in ('gate_proj','up_proj','down_proj'))]
    stable_before=digest(stable)
    formal.onevision_bridge.NATIVE_GATE_MODE=True
    gates=[formal.onevision_bridge.forward_gate(fit[t['task']][0],t,processor) for t in manifest['tasks']]
    formal.onevision_bridge.NATIVE_GATE_MODE=False
    write(run/'NATIVE_GATE.json',dict(complete=True,gates=gates,NLL_paths=38,short_generations=38,max_new_tokens=4,TEST_QA=0))
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    monitor.snapshot('training_before',model=model)
    namespace=dict(vars(formal),model=model,method=method,processor=processor,manifest=manifest,data=mapped,
        run=run,monitor=monitor,args=SimpleNamespace(method=method.name),resume_obj=None)
    prior=None;counts={}
    def pre_stage(stage):method.choose_current(fit[stage],groups,manifest['tasks'][stage-1])
    def finish_stage(stage):
        nonlocal prior
        frozen_after=digest([p for _,l in method.layers for p in (l.left,l.right)]+[l.bias for _,l in method.layers if l.bias is not None])
        assert frozen_after==method.stage_frozen_digest
        def loss_fn(entry):
            batch={k:v.to(model.device) for k,v in entry['batch'].items()}
            with formal.activation_storage(method),torch.autocast('cuda',dtype=model.dtype):
                loss,_=formal.answer_loss(model,batch,entry['answer'],processor)
            return loss
        monitor.snapshot('consolidation_before',stage,model)
        record=consolidate(method,case,loss_fn)
        record['stage_frozen_digest_before_TRAIN']=method.stage_frozen_digest
        record['stage_frozen_digest_after_TRAIN']=frozen_after
        assert record['count_before']==FINAL+TRAINABLE and record['count_after']==FINAL
        write(run/'consolidation'/f'S{stage:02d}.json',record)
        if stage==19:
            method.remove_training_memory()
            model._llmcl_visual_cache.clear();model._llmcl_visual_cache_bytes=0
        write(run/'layouts'/f'S{stage:02d}.json',layout(model))
        meta=formal.save_checkpoint(method,processor,run,stage,config)
        if prior is not None:
            old=(run/prior['path']).resolve();assert old.is_relative_to((run/'checkpoints').resolve());old.unlink()
        prior=meta
        monitor.snapshot('consolidation_after_saved',stage,model)
    def commit_optimizer(v):write(run/'optimizer'/f'S{monitor.stage:02d}_{counts[monitor.stage]:03d}.json',v)
    namespace.update(pre_stage=pre_stage,finish_stage=finish_stage,
        make_optimizer=lambda m:measured_optimizer(m,formal.make_optimizer,commit_optimizer))
    prefix,h=study.training_ast();assert h==ready['training_ast_sha256']
    node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),
        body=ast.parse('pre_stage(stage)').body+prefix+ast.parse('finish_stage(stage)').body,orelse=[])
    raw_clip=torch.nn.utils.clip_grad_norm_
    def record_raw(parameters,*args,**kwargs):
        stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
        assert layout(model)['trainable']==TRAINABLE
        grads={n:dict(norm=float(p.grad.detach().float().norm()),finite=bool(torch.isfinite(p.grad).all())) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert len(grads)==144 and all(v['finite'] for v in grads.values())
        write(run/'gradients'/f'S{stage:02d}_{counts[stage]:03d}.json',dict(raw=grads))
        if counts[stage]==1:save_gradient(model,run,stage)
        return raw_clip(parameters,*args,**kwargs)
    torch.nn.utils.clip_grad_norm_=record_raw
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<original-formal-TRAIN-prefix>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=raw_clip
    assert len(counts)==19 and sum(counts.values())==87 and stable_before==digest(stable)
    assert not method.bank and not method.current
    final_layout=layout(model);assert final_layout['coefficients']==FINAL and final_layout['trainable']==0
    checkpoint=json.loads((run/'checkpoints/latest.json').read_text())
    write(run/'TRAIN_COMPLETE.json',dict(complete=True,updates=87,FIT_QA=1213,checkpoint=checkpoint,TEST_QA=0))
    model.eval();before_reload=digest(model.parameters())
    obj=formal.load_checkpoint(run);recovered=obj['method'];assert layout(recovered.model)==final_layout
    assert digest(recovered.model.parameters())==before_reload and not recovered.bank and not recovered.current
    namespace.clear();del stable,obj,model,method
    method=recovered;model=method.model;formal.onevision_bridge.bind(model);gc.collect();torch.cuda.empty_cache()
    formal.evaluate(method,processor,[t for t in manifest['tasks'] if dev[t['task']]],mapped,run,19,split='dev')
    assert digest(model.parameters())==before_reload
    quality,_=read_predictions(run,manifest,dev)
    memory=monitor.finish(model)
    folders=('gradients','optimizer','consolidation','layouts','validity')
    hashes={p.relative_to(ROOT).as_posix():sha(p) for f in folders for p in (run/f).iterdir()}
    for p in [run/'resources.jsonl',run/'memory_summary.json',*sorted((run/'checkpoints').glob('*.json'))]:
        hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    assert time.monotonic()-worker_begin<=7200,'Do not seal a completed case beyond its frozen budget'
    write(case_report/'COMPLETE.json',dict(complete=True,case=case,quality=quality,checkpoint=checkpoint,
        model_RUN_seconds=time.monotonic()-worker_begin,worker_pid=os.getpid(),
        config=config,final_layout=final_layout,all_weight_digest_before_DEV=before_reload,all_weight_digest_after_DEV=before_reload,
        stable_digest_before_TRAIN=stable_before,stable_digest_after_TRAIN=stable_before,
        full_checkpoint_reload=True,initial_binding_sha256=sha(run/'INITIAL_BINDING.json'),
        initial_layout_sha256=sha(run/'INITIAL_LAYOUT.json'),file_sha256=hashes,events_sha256=sha(run/'logs/events.jsonl'),
        native_gate_sha256=sha(run/'NATIVE_GATE.json'),memory_summary=memory,ready_sha256=sha(REPORT/'READY.json'),
        source_sha256=sha(__file__),TEST_QA=0,automatic_full_confirmation=False,not_accepted_novel_or_stable_gain=True))
    signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES,required=True);a=p.parse_args()
    try:main(a.case)
    except ResourceBusy as e:print(str(e),flush=True);sys.exit(75)
    except Exception:write(REPORT/'cases'/a.case/'FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
