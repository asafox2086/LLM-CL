"""Fresh native pilot case. No resume path, no score calculation, no work on import."""
import ast
from collections import Counter
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from types import SimpleNamespace
from core import ROOT,PORT,REPORT,WORK,BASE,ATTENTION,CASES,BUDGET,TRAINABLE,study,sha,write,digest,layout,make_method,compile_final,restore_compiled,cpu_tree,save_tensor_file,observed_optimizer
from common import check_ready,data

class ResourceBusy(RuntimeError):pass

def identity(pid):
 p=Path('/proc')/str(pid);s=p.joinpath('stat').read_text().rsplit(') ',1)[1].split()
 uid=int(next(x.split()[1] for x in p.joinpath('status').read_text().splitlines() if x.startswith('Uid:')))
 return dict(pid=pid,startticks=s[19],uid=uid,pgid=int(s[2]))

def main(case):
 ready=check_ready();path,manifest,split,fit,dev,mapped,plan,_=data()
 assert sha(path)==ready['manifest_sha256'] and plan==ready['training_plan']
 assert sha(study().REPORT/'SPLIT.json')==ready['split_sha256']
 if case=='R':assert json.loads((REPORT/'GATE_DECISION.json').read_text())['run_R']
 if case!='E':
  prior='E' if case=='Q' else 'Q';assert (REPORT/f'cases/{prior}/CASE_VERIFIED.json').exists()
 sys.path.insert(0,str(ROOT/'analysis'));import onevision_research_device_entry as device
 cards=device.inventory();ours={g for g,v in cards.items() if any(p['uid']==os.getuid() for p in v['owners'])}
 if cards[1]['owners'] or cards[1]['memory_MiB']>=500 or cards[1]['utilization']>=10 or (ours&{0,1}) or len(ours|{1})>3:
  raise ResourceBusy('Pre-STARTED resource change; no CUDA/model attempt')
 import shutil
 assert shutil.disk_usage(ROOT).free>40*1024**3
 report=REPORT/'cases'/case;run=WORK/case
 assert not (report/'STARTED.json').exists() and not run.exists()
 started=time.monotonic();who=identity(os.getpid());assert who['uid']==os.getuid() and who['pgid']==who['pid']
 write(report/'STARTED.json',dict(**who,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
  started_monotonic=started,started_unix=time.time(),budget_seconds=BUDGET[case],case=case,only_GPU=1,TEST_QA=0,
  ready_sha256=sha(REPORT/'READY.json')))
 def expired(*unused):raise TimeoutError('Frozen STARTED-to-COMPLETE case budget expired; whole pilot must close')
 signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,BUDGET[case]-(time.monotonic()-started))
 os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8';os.environ['TOKENIZERS_PARALLELISM']='false'
 import numpy as np
 import torch
 from transformers import AutoProcessor,Idefics3ForConditionalGeneration,GenerationConfig
 from runtime import NativeRuntime,policy
 sys.path.insert(0,str(PORT))
 spec=importlib.util.spec_from_file_location('native_reuse_formal_run',PORT/'run.py')
 formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal);sys.modules['run']=formal
 run.mkdir(parents=True);(run/'logs').mkdir();(run/'gradients').mkdir()
 monitor=formal.ResourceMonitor(run);monitor.snapshot('process_before_model')
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
 assert torch.cuda.is_bf16_supported(),'Native carrier BF16 required; no precision fallback'
 processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
 stop=policy(processor,GenerationConfig.from_pretrained(BASE,local_files_only=True));assert stop==ready['native_policy']
 model,loading=Idefics3ForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,attn_implementation=dict(ATTENTION),local_files_only=True,output_loading_info=True)
 assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),loading
 assert model.model.vision_model.config._attn_implementation=='eager'
 assert model.model.text_model.config._attn_implementation=='sdpa'
 model=model.cuda()
 parent_layout=layout(model);assert parent_layout['weight_coefficients']==ready['base_weight_coefficients']
 # Every case starts from the same pinned, untrained original weight bytes.
 base_digest=digest(model.state_dict().values())
 method=make_method(model,case,run/'method_state');method.vision_original=None
 train_layout=layout(model);assert train_layout['trainable_coordinates']==TRAINABLE[case]
 assert len(method.common_names)==64 and (len(method.expert_layers)==96 if case=='E' else not method.expert_layers)
 train_names=sorted(n for n,p in model.named_parameters() if p.requires_grad)
 assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
 common_qv_digest=digest([p for n,p in model.named_parameters() if '.A.' in n or '.B.' in n])
 write(run/'INITIAL.json',dict(case=case,base_files=ready['base_files'],base_digest=base_digest,
  common_qv_digest=common_qv_digest,base_layout=parent_layout,untrained=True))
 write(run/'TRAIN_LAYOUT.json',train_layout)
 frozen=[p for p in model.parameters() if not p.requires_grad];frozen_before=digest(frozen)
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
 model.config.use_cache=False
 runtime=NativeRuntime(processor,model,run,stop,image_SHA256=split['image_sha256'])
 config=dict(case=case,source_sha256=sha(__file__),ready_sha256=sha(REPORT/'READY.json'),manifest_sha256=ready['manifest_sha256'],
  split_sha256=ready['split_sha256'],FIT_QA=1213,DEV_QA=100,TEST_QA=0,updates=87,seed=42,
  learning_rate=1e-4,epoch=1,batch_size=16,micro_batch_size=1,attention=dict(ATTENTION),base_dtype='torch.bfloat16',master_dtype='torch.float32',
  common_qv_rank=8,common_qv_alpha=32,common_qv_dropout=.1,worker_budget_seconds=BUDGET[case],aggregate_budget_seconds=36000,
  case_extra='native_full_down' if case=='R' else 'original_FFN_local_PAS' if case=='E' else 'none',
  E_FP32_internal_autocast_disabled_exception=(case=='E'),no_resume=True,not_formal_PAS_reproduction=True)
 write(run/'config.json',config)
 monitor.snapshot('training_before',model=model)
 original_begin=method.begin_stage
 def begin_stage(stage):
  # Only the current stage first FIT row; recording remains off for the gate.
  original_begin(stage);model.eval()
  if method.learner is not None:method.learner.enable_record(False)
  runtime.gate(fit[stage][0],manifest['tasks'][stage-1],stage)
 method.begin_stage=begin_stage
 previous=None
 def save_current(stage):
  nonlocal previous
  weights=cpu_tree(model.state_dict());checkpoint=run/'checkpoints'/f'S{stage:02d}.pt'
  payload=dict(kind='native_reuse_training_graph_v1',case=case,stage=stage,config=model.config.to_dict(),weights=weights,
   method_state=cpu_tree(method.method_state()),graph_spec=dict(common_qv=True,extra=config['case_extra']),
   training_config=config,state_digest=digest(weights.values()),no_resume_path=True)
  meta=save_tensor_file(checkpoint,payload);meta.update(stage=stage,path=checkpoint.relative_to(run).as_posix(),full_graph_spec_and_method_state=True)
  write(run/'checkpoints'/f'S{stage:02d}.json',meta)
  if previous is not None:
   old=(run/previous['path']).resolve();assert old.is_relative_to((run/'checkpoints').resolve());old.unlink()
  previous=meta
 def training_input(*a,**kw):
  runtime.phase='TRAIN';return runtime.input_batch(*a,**kw)
 namespace=dict(vars(formal),model=model,method=method,processor=processor,manifest=manifest,data=mapped,run=run,monitor=monitor,
  args=SimpleNamespace(method='NativeResponseReuse'),resume_obj=None,input_batch=training_input,answer_loss=runtime.answer_loss,
  prepare=lambda *a:None,visual_cache_stats=lambda *a:{},save_current=save_current,
  make_optimizer=lambda m:observed_optimizer(m,formal.make_optimizer,run,lambda:[r['id'] for r in namespace['minibatch']] ))
 prefix,h=study().training_ast();assert h==ready['training_ast_sha256']
 loop=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),
  args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_current(stage)').body,orelse=[])
 raw_clip=torch.nn.utils.clip_grad_norm_;counts={}
 def record_raw(parameters,*args,**kwargs):
  stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
  grads={n:dict(norm=float(p.grad.detach().float().norm()),finite=bool(torch.isfinite(p.grad).all()))
   for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
  assert set(grads)==set(train_names) and all(v['finite'] and math.isfinite(v['norm']) for v in grads.values())
  if counts[stage]==1:
   save_tensor_file(run/'gradients'/f'S{stage:02d}_first_raw.pt',
    {n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None})
  norm=raw_clip(parameters,*args,**kwargs);assert math.isfinite(float(norm))
  write(run/'gradients'/f'S{stage:02d}_{counts[stage]:03d}.json',dict(raw=grads,preclip_total_norm=float(norm)))
  return norm
 torch.nn.utils.clip_grad_norm_=record_raw
 try:exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'<original-TRAIN-only-prefix>','exec'),namespace)
 finally:torch.nn.utils.clip_grad_norm_=raw_clip
 assert sum(counts.values())==87 and len(counts)==len(method.records)==19
 assert digest(frozen)==frozen_before
 write(run/'CALLBACKS.json',method.records)
 write(run/'TRAIN_COMPLETE.json',dict(complete=True,updates=87,FIT_QA=1213,TEST_QA=0))
 compiled=compile_final(method);compiled_layout=layout(compiled)
 weights=cpu_tree(compiled.state_dict())
 final_payload=dict(kind='native_reuse_compiled_v1',case=case,config=compiled.config.to_dict(),weights=weights,
  state_digest=digest(weights.values()),gate_tau=[m.gate_tau for m in method.expert_layers])
 compiled_meta=save_tensor_file(run/'COMPILED.pt',final_payload)
 compiled_meta.update(path='COMPILED.pt',weight_coefficients=compiled_layout['weight_coefficients'])
 # Clear all training-model holders before constructing the final reload.
 frozen_after=digest(frozen);assert frozen_after==frozen_before
 frozen.clear();namespace.clear();del namespace,weights,final_payload,compiled,model,method,original_begin,begin_stage
 runtime.model=None;gc.collect();torch.cuda.empty_cache()
 reloaded=restore_compiled(run/'COMPILED.pt',device='cuda');runtime.model=reloaded
 assert layout(reloaded)==compiled_layout
 before=digest(reloaded.state_dict().values());monitor.snapshot('compiled_before_DEV',19,reloaded)
 runtime.evaluate(manifest,dev);after=digest(reloaded.state_dict().values());assert before==after
 memory=monitor.finish(reloaded)
 with (run/'INPUT_RECORDS.jsonl').open('w',encoding='utf-8') as f:
  for r in runtime.input_records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  f.flush();os.fsync(f.fileno())
 write(run/'REQUESTS.json',dict(counts=Counter(r['kind'] for r in runtime.requests),details=runtime.requests,native_gates=runtime.native_gates))
 evidence=[p for p in run.rglob('*') if p.is_file() and p.name!='COMPILED.pt' and p.suffix not in ('.tmp','.partial')]
 # Includes the last private TRAIN checkpoint and all 19 first-gradient tensors.
 file_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in evidence}
 elapsed=time.monotonic()-started;assert elapsed<=BUDGET[case]
 complete=dict(complete=True,case=case,worker_pid=os.getpid(),ready_sha256=sha(REPORT/'READY.json'),model_RUN_seconds=elapsed,
  config=config,common_qv_digest=common_qv_digest,frozen_before_TRAIN=frozen_before,frozen_after_TRAIN=frozen_after,
  compiled_layout=compiled_layout,compiled_checkpoint=compiled_meta,final_reload=True,compiled_before_DEV=before,compiled_after_DEV=after,
  native_gates=runtime.native_gates,request_counts=dict(Counter(r['kind'] for r in runtime.requests)),trainable_names=train_names,
  file_sha256=file_hashes,memory_summary=memory,TEST_QA=0)
 write(report/'COMPLETE.json',complete);signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True,choices=CASES);args=ap.parse_args()
 try:main(args.case)
 except ResourceBusy as e:print(str(e),flush=True);sys.exit(75)
 except BaseException:
  write(REPORT/'cases'/args.case/'FAILURE.json',dict(traceback=traceback.format_exc(),no_retry=True,whole_pilot_stop=True));raise
