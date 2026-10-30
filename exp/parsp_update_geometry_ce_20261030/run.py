"""One original PaRSP T19 TRAIN update from full S18; train-mode paired TRAIN CE only; no method/TEST/F1."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,signal,traceback,importlib.util,math,random
CP_SHA='0b704968b1314595f5382464a1acc6cdaabc600da6a9e560692583040d992d06'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8*1024**2),b''):h.update(z)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
def scope(b):
 assert b['measurement_contract']=='native-step-geometry-plus-paired-CE-v1'
 assert b['schema']=='parsp-native-step-geometry-v1' and b['stage']==19 and b['effective_batch']==16 and b['micro_batch']==1
 P=Path(b['native_source_dir']);A=Path(b['load_run']);E=Path(__file__).resolve().parent
 assert E==Path(b['source_dir'])
 for n,h in b['sources'].items():assert sha(E/n)==h
 assert sha(b['protocol_path'])==b['protocol_sha256'] and sha(b['asset_receipt'])==b['asset_receipt_sha256']
 r=read(b['asset_receipt']);cfg=read(A/'config.json');meta=read(A/'checkpoints/latest.json');cp=(A/meta['path']).resolve()
 assert meta['stage']==18 and meta['sha256']==CP_SHA and cp.stat().st_size==5928091794
 assert cp.is_relative_to((A/'checkpoints').resolve()) and cp.stat().st_ino==Path(r['original_CP_resolved']).stat().st_ino and cp.stat().st_dev==Path(r['original_CP_resolved']).stat().st_dev
 for n,h in cfg['execution_source_sha256'].items():assert sha(P/n)==h
 assert cfg['method']=='PaRSP' and cfg['condition']=='pct10' and cfg['manifest_sha256']=='02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'
 root=P.parents[1];manifest=read(b['manifest_path']);assert sha(b['manifest_path'])==cfg['manifest_sha256']
 task=manifest['tasks'][18];f=task['files']['train'];path=root/f['path'];assert sha(path)==f['sha256']
 rows=[json.loads(z) for z in path.read_text().splitlines()];assert len(rows)==f['count']
 order=sorted(rows,key=lambda x:hashlib.sha256(f"42:19:{x['id']}".encode()).digest())
 assert hashlib.sha256(''.join(x['id']+chr(10) for x in order).encode()).hexdigest()==task['training_order_sha256']
 ids=[hashlib.sha256(x['id'].encode()).hexdigest() for x in order[:16]];assert ids==b['training_micro_ID_sha256']
 local=sorted(rows,key=lambda x:hashlib.sha256(('42:calibration:'+x['id']).encode()).digest())[:256]
 assert [hashlib.sha256(x['id'].encode()).hexdigest() for x in local]==b['localizer_ID_sha256']
 for item in r['original_protected_metadata']:assert sha(item['path'])==item['sha256']
 return P,A,cfg,manifest,task,rows,order,r

def object_signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def tensor_signature(value):
 import torch
 return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
def batch_signature(batch):
 import torch
 return object_signature({k:tensor_signature(v) if isinstance(v,torch.Tensor) else repr(v) for k,v in sorted(batch.items())})
def rng_signature(state):
 h=hashlib.sha256();h.update(repr(state['python']).encode());n=state['numpy'];h.update(repr((n[0],n[2:])).encode());h.update(n[1].tobytes());h.update(state['cpu'].numpy().tobytes())
 for value in state['cuda']:h.update(value.numpy().tobytes())
 return h.hexdigest()
def buffer_signature(model):return object_signature({n:tensor_signature(v) for n,v in model.named_buffers()})
def mode_signature(model):return object_signature({n:bool(v.training) for n,v in model.named_modules()})
def cache_signature(method):
 io=method.io.cache;visual=getattr(method.model,'_llmcl_visual_cache',{})
 return dict(io_entries=len(io),io_key_order_sha256=hashlib.sha256(repr(list(io)).encode()).hexdigest(),visual_entries=len(visual),visual_key_order_sha256=hashlib.sha256(repr(list(visual)).encode()).hexdigest(),cache_values_not_rehashed=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args();b=read(args.binding)
 P,A,cfg,manifest,task,rows,order,assets=scope(b)
 if not args.execute:print(json.dumps(dict(metadata_validated=True,model_loaded=False,stage18=True,history_groups=96,localizer_QA=256,train_micro_QA=16,optimizer_steps=1,planned_after_step_CE_forward=16,TEST_QA=0)));return
 assert b['admitted'] is True and os.getuid()==1084 and os.environ['CUDA_VISIBLE_DEVICES']==str(b['physical_GPU'])
 adm=read(b['admission_path']);assert adm['approved'] and adm['binding_sha256']==sha(args.binding)
 import subprocess
 uuid=subprocess.check_output(['nvidia-smi','--id='+str(b['physical_GPU']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==b['gpu_uuid']
 out=Path(b['output_root']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();write(out/'STARTED.json',dict(binding_sha256=sha(args.binding),PID=os.getpid(),UID=os.getuid(),PGID=os.getpgrp(),no_retry=True))
 def expired(*_):raise TimeoutError('fixed native one-step diagnostic deadline')
 signal.signal(signal.SIGALRM,expired);signal.alarm(b['limit_seconds']);captures={}
 try:
  import numpy as np,torch
  random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8);torch.cuda.set_device(0);assert torch.cuda.device_count()==1
  torch.cuda.reset_peak_memory_stats();sys.path.insert(0,str(P))
  import checkpoints,runtime_utils as rt,onevision_bridge as bridge
  from frozen_vision import prepare
  from training_validity import warmup_updates
  from transformers import get_cosine_schedule_with_warmup
  spec=importlib.util.spec_from_file_location('constraint_measure',Path(__file__).parent/'measure.py');measurement=importlib.util.module_from_spec(spec);spec.loader.exec_module(measurement)
  for mod in [checkpoints,rt,bridge]:assert sha(mod.__file__)==cfg['execution_source_sha256'][Path(mod.__file__).name]
  assert sys.flags.optimize==0
  obj=checkpoints.load(A,expected=cfg);assert obj['stage']==18 and obj['probe'] is None and obj['method'].stage==18
  method=obj['method'];model=method.model;processor=obj['processor'];assert method.name=='PaRSP' and len(method.history)==96 and not method.preflight
  assert len(method.io.cache)==0 and not hasattr(method,'after_update')
  bridge.bind(model);bridge.NATIVE_GATE_MODE=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
  method.bind_context(processor,manifest,{(19,'train'):rows},out,False)
  phase={'name':'begin_stage19'};forward_counts={'begin_stage19':0,'one_step':0,'after_step_CE':0}
  def count(*_):forward_counts[phase['name']]+=1
  decoder_count=model.model.register_forward_pre_hook(count)
  # Restore saved next-stage RNG after instrumentation, exactly as original resume.
  rt.restore_rng(obj['rng']);del obj
  method.begin_stage(19);assert method.stage==19 and method.selection['probe_count']==256
  layer_index=min(method.selection['ffn_layers']);mlp=model.model.layers[layer_index].mlp
  for name,axis,key in [('down_proj',1,'down'),('gate_proj',0,'gateup'),('up_proj',0,'gateup')]:
   module=getattr(mlp,name);captures[name]=measurement.NativeCapture(module.weight,method.projectors[f'{layer_index}:{key}'],axis)
  parameters=[p for p in model.parameters() if p.requires_grad];optimizer=rt.make_optimizer(method);assert all(g['weight_decay']==0 for g in optimizer.param_groups)
  assert sum(p.numel() for p in parameters)==134195328
  updates=math.ceil(len(order)/16);warmup=warmup_updates(updates);assert updates==31 and warmup==0
  scheduler=get_cosine_schedule_with_warmup(optimizer,warmup,updates);assert [g['lr'] for g in optimizer.param_groups]==[3e-5]
  scaler=torch.amp.GradScaler('cuda',init_scale=1,enabled=True);method.set_instruction(19);model.train();optimizer.zero_grad(set_to_none=True)
  before={n:c.parameter.detach().clone() for n,c in captures.items()};phase['name']='one_step';loss_value=0.;target_tokens=0;before_micro=[]
  for i,row in enumerate(order[:16]):
   micro_rng=rt.rng()
   for capture in captures.values():capture.micro=i
   batch,_=rt.input_batch(row,task,processor,model.device);prepare(batch,row,model,method.vision_original)
   signature=dict(rng_before_forward=rng_signature(rt.rng()),prepared_batch_sha256=batch_signature(batch),buffers_sha256=buffer_signature(model),mode_sha256=mode_signature(model),grad_enabled=torch.is_grad_enabled(),model_train=bool(model.training))
   with rt.activation_storage(method),torch.autocast('cuda',dtype=model.dtype):loss,length=rt.answer_loss(model,batch,row['answer'],processor)
   assert bool(torch.isfinite(loss));scaler.scale(loss/16).backward();loss_value+=float(loss.detach())/16;target_tokens+=length
   before_micro.append(dict(prepare_rng=micro_rng,signature=signature,CE=float(loss.detach()),target_tokens=length,qa_id_sha256=hashlib.sha256(row['id'].encode()).hexdigest()))
   del loss,batch
  scale=float(scaler.get_scale());assert scale==1.
  scaler.unscale_(optimizer);pre={n:c.parameter.grad.detach().clone() for n,c in captures.items()};norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert bool(torch.isfinite(norm))
  post={n:c.parameter.grad.detach().clone() for n,c in captures.items()};lr=[g['lr'] for g in optimizer.param_groups];scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=scale;scheduler.step()
  genuine_post_step_rng=rt.rng();genuine_post_step_rng_sha256=rng_signature(genuine_post_step_rng)
  report={n:c.report(before[n],c.parameter,pre[n],post[n],scale) for n,c in captures.items()}
  for c in captures.values():c.close()
  assert forward_counts['one_step']==16
  before_readout_versions={n:int(p._version) for n,p in model.named_parameters()}
  caches_before_readout=cache_signature(method);CE_pairs=[];phase['name']='after_step_CE'
  assert bool(model.training) and torch.is_grad_enabled()
  try:
   for i,row in enumerate(order[:16]):
    previous=before_micro[i];rt.restore_rng(previous['prepare_rng'])
    paired_batch,_=rt.input_batch(row,task,processor,model.device);prepare(paired_batch,row,model,method.vision_original)
    observed=dict(rng_before_forward=rng_signature(rt.rng()),prepared_batch_sha256=batch_signature(paired_batch),buffers_sha256=buffer_signature(model),mode_sha256=mode_signature(model),grad_enabled=torch.is_grad_enabled(),model_train=bool(model.training))
    with rt.activation_storage(method),torch.autocast('cuda',dtype=model.dtype):ce_after,length_after=rt.answer_loss(model,paired_batch,row['answer'],processor)
    assert bool(torch.isfinite(ce_after));value=float(ce_after.detach())
    guards={key:observed[key]==previous['signature'][key] for key in observed};guards['target_tokens']=length_after==previous['target_tokens'];guards['grad_enabled_true']=bool(observed['grad_enabled']);guards['model_train_true']=bool(observed['model_train'])
    CE_pairs.append(dict(qa_id_sha256=previous['qa_id_sha256'],CE_before=previous['CE'],CE_after=value,delta_after_minus_before=value-previous['CE'],target_tokens_before=previous['target_tokens'],target_tokens_after=length_after,pair_guards=guards,strict_pair=all(guards.values()),before_signature=previous['signature'],after_signature=observed))
    del ce_after,paired_batch
  finally:rt.restore_rng(genuine_post_step_rng)
  restored_post_step_rng=rng_signature(rt.rng())==genuine_post_step_rng_sha256
  after_readout_versions={n:int(p._version) for n,p in model.named_parameters()};versions_unchanged=after_readout_versions==before_readout_versions
  caches_after_readout=cache_signature(method);decoder_count.remove();assert forward_counts['after_step_CE']==16 and restored_post_step_rng
  assert all(c.calls==list(range(16)) for c in captures.values())
  paired_CE=dict(rows=CE_pairs,N=16,mean_before=math.fsum(x['CE_before'] for x in CE_pairs)/16,mean_after=math.fsum(x['CE_after'] for x in CE_pairs)/16,mean_delta=math.fsum(x['delta_after_minus_before'] for x in CE_pairs)/16,all_pairs_strict=all(x['strict_pair'] for x in CE_pairs) and versions_unchanged,parameters_version_unchanged=versions_unchanged,parameter_versions_sha256_before=object_signature(before_readout_versions),parameter_versions_sha256_after=object_signature(after_readout_versions),post_step_RNG_restored=restored_post_step_rng,post_step_RNG_sha256=genuine_post_step_rng_sha256,caches_before_readout=caches_before_readout,caches_after_readout=caches_after_readout,cache_membership_state_restored=False,cache_scope='NativeIO localizer cache is not used by the step/readout. Frozen visual LRU may have different hits/members; actual prepared tensors and forward-entry RNG must match for strict pairs.',forward_mode='train, grad-enabled, native activation_storage/autocast; one graph per QA released, no backward or optimizer step',causal_scope='Same16 TRAIN conditional CE only; no proof that any CE change is caused by the three measured targets or constraint leakage, no convergence/generalization/F1 claim.')
  assert all(x['accumulation']['elementwise_bound_violations']==0 for x in report.values())
  for item in assets['original_protected_metadata']:assert sha(item['path'])==item['sha256']
  torch.cuda.synchronize();result=dict(schema='parsp-native-step-geometry-v1',binding_sha256=sha(args.binding),checkpoint_source_sha256=CP_SHA,checkpoint_loads=1,starting_stage=18,observed_stage=19,conditioned_native_begin_stage=True,bitwise_original_stage19_replay_claimed=False,
   localizer_QA=256,localizer_calls=1,TRAIN_micro_QA=16,effective_batch=16,optimizer_steps=1,training_micro_backwards=16,TEST_QA=0,new_scores=0,generations=0,new_checkpoints=0,
   target_layer_index=layer_index,selection=method.selection,targets=report,forward_counts=forward_counts,gradient_scale=scale,global_gradient_l2_preclip=float(norm),LR_used=lr,warmup_updates=warmup,weight_decay=0,
   observed_original_micro_mean_TRAIN_CE_before=loss_value,TRAIN_target_tokens=target_tokens,after_step_CE_forward=16,paired_TRAIN_CE=paired_CE,
   registered_neural_parameters=sum(p.numel() for p in model.parameters()),registered_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),temporary_GPU_allocated_peak=int(torch.cuda.max_memory_allocated()),temporary_GPU_reserved_peak=int(torch.cuda.max_memory_reserved()),GPU_uuid=uuid,seconds=time.monotonic()-start,
   scope='Original native PaRSP conditional one-step replay. Geometry only; no causal learning damage, convergence, F1 or generic intersection-zero claim. Original checkpoint/source/benchmark unchanged; no post-step mask/projector added.')
  write(out/'RESULT.json',result);write(out/'COMPLETE.json',dict(binding_sha256=sha(args.binding),RESULT_sha256=sha(out/'RESULT.json'),seconds=time.monotonic()-start));signal.alarm(0)
 except BaseException:
  for c in captures.values():c.close()
  write(out/'FAILURE.json',dict(traceback=traceback.format_exc(),no_automatic_retry=True,seconds=time.monotonic()-start));raise
if __name__=='__main__':main()
