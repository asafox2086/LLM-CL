"""Fresh full1313 training, then one original final21454 evaluation; GPU1 only."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import traceback
from types import SimpleNamespace
from common import ROOT,PORT,REPORT,WORK,CASE,study,sha,write,check_ready,original_data,load_test_after_training,prediction_summary
class ResourceBusy(RuntimeError): pass

def main():
    sys.path.insert(0,str(ROOT/'analysis'))
    import onevision_research_device_entry as device
    ready=check_ready()
    path,manifest,original,plan,_=original_data()
    assert sha(path)==ready['manifest_sha256'] and plan==ready['training_plan']
    # Bind every original TRAIN image to the already frozen screen image ledger.
    for image,h in ready['TRAIN_image_sha256'].items(): assert sha(ROOT/image)==h
    cards=device.inventory();ours={g for g,v in cards.items() if any(p['uid']==os.getuid() for p in v['owners'])}
    if cards[1]['owners'] or cards[1]['memory_MiB']>=500 or cards[1]['utilization']>=10 or (ours & {0,1}) or len(ours|{1})>3:
        raise ResourceBusy('Resource changed before STARTED/CUDA; unattempted healthy wait')
    assert not (REPORT/'STARTED.json').exists() and not WORK.exists(), 'No retry or prediction continuation'
    write(REPORT/'STARTED.json',dict(pid=os.getpid(),startticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),inventory=cards,ready_sha256=sha(REPORT/'READY.json')))
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
    import torch,numpy as np
    from transformers import AutoProcessor
    sys.path.insert(0,str(PORT))
    spec=importlib.util.spec_from_file_location('confirmation_formal_run',PORT/'run.py')
    formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal);sys.modules['run']=formal
    data={(t,'train'):rows for t,rows in original.items()}
    run=WORK;run.mkdir(parents=True);(run/'logs').mkdir()
    monitor=formal.ResourceMonitor(run);monitor.snapshot('process_before_model')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    base=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    model=formal.onevision_bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
    formal.onevision_bridge.bind(model)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    method=study.make_method(model,CASE,[t['instruction'] for t in manifest['tasks']])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    method.vision_original=formal.install(model)
    config=dict(method=CASE,condition='pct01',manifest_sha256=ready['manifest_sha256'],seed=42,rank=8,alpha=32,dropout=.1,
        learning_rate=1e-4,TRAIN_QA=1313,stages=19,updates=93,TEST_QA=21454,
        dtype='FP16 base/outer CUDA autocast; inherited FP32 LoRA tensors',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),target_modules=len(method.layers),
        execution_source_sha256=sha(__file__),ready_sha256=sha(REPORT/'READY.json'),fresh_base_not_FIT_checkpoint=True,
        coverage='final_only_19_tasks_not_74_cells',checkpoint_payload_phase_note='Saved after TRAIN stage before any TEST; inherited payload phase label is not actual evaluation timing')
    assert config['trainable_parameters']==4399104 and config['target_modules']==168
    write(run/'config.json',config)
    formal.onevision_bridge.NATIVE_GATE_MODE=True
    gates=[formal.onevision_bridge.forward_gate(original[t['task']][0],t,processor) for t in manifest['tasks']]
    formal.onevision_bridge.NATIVE_GATE_MODE=False
    write(run/'NATIVE_GATE.json',dict(complete=True,gates=gates,TRAIN_QA=19,NLL_paths=38,short_generation_calls=38,max_new_tokens=4,TEST_QA=0))
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    monitor.snapshot('training_before',model=model)
    namespace=dict(vars(formal),model=model,method=method,processor=processor,manifest=manifest,data=data,run=run,monitor=monitor,
        args=SimpleNamespace(method=method.name),resume_obj=None)
    prior=None
    def save_current(stage):
        nonlocal prior
        meta=formal.save_checkpoint(method,processor,run,stage,config)
        assert sha(run/meta['path'])==meta['sha256']
        if prior is not None:
            old=(run/prior['path']).resolve();assert old.is_relative_to((run/'checkpoints').resolve());old.unlink()
        prior=meta
    namespace['save_current']=save_current
    prefix,h=study.training_ast();assert h==ready['train_AST_sha256']
    node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),
        args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_current(stage)').body,orelse=[])
    raw_clip=torch.nn.utils.clip_grad_norm_;counts={}
    def record_raw(parameters,*args,**kwargs):
        stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
        grads={n:dict(norm=float(p.grad.detach().float().norm()),finite=bool(torch.isfinite(p.grad).all()))
            for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert grads and all(v['finite'] for v in grads.values())
        write(run/'gradients'/f'S{stage:02d}_{counts[stage]:03d}.json',dict(raw=grads))
        if counts[stage]==1:
            torch.save({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},
                run/'gradients'/f'S{stage:02d}_first_raw.pt')
        return raw_clip(parameters,*args,**kwargs)
    torch.nn.utils.clip_grad_norm_=record_raw
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<original-formal-TRAIN-prefix>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=raw_clip
    events=[json.loads(s) for s in (run/'logs/events.jsonl').read_text().splitlines()]
    assert len(events)==93 and len(counts)==19 and all(any(lr>0 for lr in e['learning_rate_used']) for e in events)
    checkpoint=json.loads((run/'checkpoints/latest.json').read_text())
    write(run/'TRAIN_COMPLETE.json',dict(complete=True,checkpoint=checkpoint,TRAIN_QA=1313,updates=93,TEST_QA=0))
    # The official TEST QA files are opened for the first time by this worker here.
    data=load_test_after_training(manifest,original)
    write(run/'TEST_OPENED.json',dict(after_TRAIN_COMPLETE_sha256=sha(run/'TRAIN_COMPLETE.json'),TEST_QA=21454))
    digest_before=formal.parameter_digest([p for p in model.parameters() if p.requires_grad])
    monitor.snapshot('final_evaluation_before',19,model)
    assert not (run/'evaluations').exists(), 'One final generation pass; no existing predictions'
    formal.evaluate(method,processor,manifest['tasks'],data,run,19,split='test')
    digest_after=formal.parameter_digest([p for p in model.parameters() if p.requires_grad]);assert digest_after==digest_before
    quality=prediction_summary(manifest,data)
    memory=monitor.finish(model)
    gradient_files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((run/'gradients').iterdir())}
    write(REPORT/'COMPLETE.json',dict(complete=True,case=CASE,TRAIN_QA=1313,updates=93,quality=quality,config=config,
        checkpoint=checkpoint,coverage='final_only_19_tasks_not_74_cells',
        trainable_digest_before_TEST=digest_before,trainable_digest_after_TEST=digest_after,
        events_sha256=sha(run/'logs/events.jsonl'),gradient_sha256=gradient_files,
        memory_summary=memory,ready_sha256=sha(REPORT/'READY.json'),source_sha256=sha(__file__),
        deployment_not_measured=True,not_new_algorithm_or_significance=True))
if __name__=='__main__':
    try:main()
    except ResourceBusy as e:print(str(e),flush=True);sys.exit(75)
    except Exception:
        write(REPORT/'FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
