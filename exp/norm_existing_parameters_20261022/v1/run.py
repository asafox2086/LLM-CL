"""One fresh existing RMSNorm-only FIT/DEV condition; official TEST is never loaded."""
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
from core import ROOT,PORT,REPORT,WORK,study,sha,write,NormOnly
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
    spec=importlib.util.spec_from_file_location('norm_only_formal_run',PORT/'run.py')
    formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal);sys.modules['run']=formal
    run=WORK;run.mkdir(parents=True);(run/'logs').mkdir()
    monitor=formal.ResourceMonitor(run);monitor.snapshot('process_before_model')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    base=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    model=formal.onevision_bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
    formal.onevision_bridge.bind(model)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    method=NormOnly(model)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==43008 and len(method.layers)==48
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    method.vision_original=formal.install(model)
    from core import original_norm_source
    assert original_norm_source()==ready['original_norm_source']
    torch.save(method.snapshot(),run/'initial_gamma.pt')
    config=dict(method='RMSNorm_only',learning_rate=1e-4,
        host_optimizer=False,offload_activations=False,master_dtype='float32',forward_weight_dtype='float16',
        optimizer_state_dtype='float32',manifest_sha256=ready['manifest_sha256'],split_sha256=ready['split_sha256'],
        trainable_coordinates=43008,target_modules=48,FIT_QA=1213,DEV_QA=100,TEST_QA=0,updates=87,seed=42,
        dtype='FP16 base/outer CUDA autocast; FP32 master gains cast back to original FP16 for forward',fresh_base=True,
        checkpoint_payload_phase_note='Inherited phase label historical; actual saves after TRAIN and before sole final DEV',
        source_sha256=sha(__file__),ready_sha256=sha(REPORT/'READY.json'),
        functional_alternative_recipe_not_matched_weight_step_or_new_algorithm=True)
    write(run/'config.json',config)
    formal.onevision_bridge.NATIVE_GATE_MODE=True
    gates=[formal.onevision_bridge.forward_gate(fit[t['task']][0],t,processor) for t in manifest['tasks']]
    formal.onevision_bridge.NATIVE_GATE_MODE=False
    write(run/'NATIVE_GATE.json',dict(complete=True,gates=gates,FIT_QA=19,NLL_paths=38,short_generations=38,max_new_tokens=4,TEST_QA=0))
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    config['frozen_digest_before_TRAIN']=formal.parameter_digest([p for p in model.parameters() if not p.requires_grad])
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
    prefix,h=study.training_ast();assert h==ready['training_ast_sha256']
    node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_current(stage)').body,orelse=[])
    raw_clip=torch.nn.utils.clip_grad_norm_;counts={}
    def record_raw(parameters,*args,**kwargs):
        stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
        grads={n:dict(norm=float(p.grad.detach().float().norm()),finite=bool(torch.isfinite(p.grad).all())) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert grads and all(v['finite'] for v in grads.values())
        write(run/'gradients'/f'S{stage:02d}_{counts[stage]:03d}.json',dict(raw=grads))
        if counts[stage]==1:torch.save({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},run/'gradients'/f'S{stage:02d}_first_raw.pt')
        return raw_clip(parameters,*args,**kwargs)
    torch.nn.utils.clip_grad_norm_=record_raw
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<original-formal-TRAIN-prefix>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=raw_clip
    assert len(counts)==19
    assert len(method.update_evidence)==87
    write(run/'gain_updates.json',method.update_evidence)
    torch.save(method.snapshot(),run/'final_gamma.pt')
    config['frozen_digest_after_TRAIN']=formal.parameter_digest([p for p in model.parameters() if not p.requires_grad])
    assert config['frozen_digest_before_TRAIN']==config['frozen_digest_after_TRAIN']
    write(run/'config.json',config)
    first=torch.load(run/'initial_gamma.pt',map_location='cpu',weights_only=True);last=method.snapshot()
    final_change={kind:dict(changed=sum(int((first[n][kind]!=last[n][kind]).sum()) for n in first),
        l2=float(torch.cat([(last[n][kind].float()-first[n][kind].float()).reshape(-1) for n in first]).norm()))
        for kind in ['master','forward']}
    checkpoint=json.loads((run/'checkpoints/latest.json').read_text())
    write(run/'TRAIN_COMPLETE.json',dict(complete=True,checkpoint=checkpoint,FIT_QA=1213,updates=87,TEST_QA=0,update_evidence_sha256=sha(run/'gain_updates.json')))
    before=formal.parameter_digest([p for p in model.parameters() if p.requires_grad])
    forward_before=formal.parameter_digest([v['forward'] for v in method.snapshot().values()])
    assert not (run/'evaluations').exists()
    formal.evaluate(method,processor,[t for t in manifest['tasks'] if dev[t['task']]],mapped,run,19,split='dev')
    after=formal.parameter_digest([p for p in model.parameters() if p.requires_grad]);assert before==after
    forward_after=formal.parameter_digest([v['forward'] for v in method.snapshot().values()]);assert forward_before==forward_after
    quality,_=read_predictions(run,manifest,dev)
    memory=monitor.finish(model)
    write(REPORT/'COMPLETE.json',dict(complete=True,quality=quality,config=config,checkpoint=checkpoint,
        update_evidence_sha256=sha(run/'gain_updates.json'),
        trainable_digest_before_DEV=before,trainable_digest_after_DEV=after,
        forward_digest_before_DEV=forward_before,forward_digest_after_DEV=forward_after,final_gain_change=final_change,
        gamma_sha256={n:sha(run/n) for n in ['initial_gamma.pt','final_gamma.pt']},
        events_sha256=sha(run/'logs/events.jsonl'),gradient_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (run/'gradients').iterdir()},
        config_sha256=sha(run/'config.json'),native_gate_sha256=sha(run/'NATIVE_GATE.json'),
        memory_summary=memory,ready_sha256=sha(REPORT/'READY.json'),source_sha256=sha(__file__),TEST_QA=0,
        mechanism_only_no_full_candidates=True,not_new_algorithm_or_deployment_advantage=True))
    signal.alarm(0)
if __name__=='__main__':
    try:main()
    except ResourceBusy as e:print(str(e),flush=True);sys.exit(75)
    except Exception:write(REPORT/'FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
