"""Fresh compact-first learning, followed by independent static-graph evaluation."""
from pathlib import Path
import ast,gc,hashlib,importlib.util,json,os,signal,sys,time,traceback
from types import SimpleNamespace
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN']);ASSET=Path(os.environ['LLMCL_ASSET'])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(name,value):
 with (RUN/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main(case):
 ready=read(RUN/'READY.json')
 for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
 for n,h in ready['runtime_files'].items():assert sha(ASSET/n)==h,n
 for n,h in ready['prior_files'].items():assert sha(RUN/n)==h,n
 spec=importlib.util.spec_from_file_location('ids_only',ASSET/'exp/paired_image_support_20261024/v1/common.py');ids=importlib.util.module_from_spec(spec);spec.loader.exec_module(ids)
 card=ids.inventory()[1];assert not card['owners'] and card['memory_MiB']<500 and card['utilization']<10
 who=ids.identity(os.getpid());assert who['pid']==who['pgid'];tick=time.monotonic()
 write(case+'_STARTED.json',dict(unix=time.time(),identity=who,GPU=1,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),ready_sha256=sha(RUN/'READY.json')))
 def expired(*args):raise TimeoutError('Frozen phase budget expired')
 signal.signal(signal.SIGALRM,expired);signal.alarm({'train':3000,'compact_eval':450,'parent_resource':150}[case])
 os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
 import random,numpy as np,torch
 from transformers import AutoProcessor
 sys.path.insert(0,str(ASSET/'exp/two_backbone_native_cl_20261017_v2'));sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
 import onevision_bridge as bridge,frozen_vision as fv,runtime_utils as rt,study
 import adaptation,vision_interface as vi
 spec=importlib.util.spec_from_file_location('storage_only',ROOT/'exp/ffn_static_delete_20261025/v1/core.py');storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
 assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
 torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV';processor=AutoProcessor.from_pretrained(base,local_files_only=True)
 path,manifest,original=study.original_train();split,fit,dev=study.split_train(manifest,original)
 assert study.sha(path)==ready['manifest_sha256'] and split==read(study.REPORT/'SPLIT.json') and study.sha(study.REPORT/'SPLIT.json')==ready['split_sha256']
 rows=[(t,row) for t in manifest['tasks'] for row in dev[t['task']]];assert len(rows)==100
 prior_predictions={c:{x['id']:x for x in read(RUN/('prior_'+c+'_predictions.json'))} for c in ['parent','candidate']}
 def buffer_digest(m):return storage.digest({n:v for n,v in m.core.named_buffers() if not n.startswith(('vision_tower.','multi_modal_projector.'))})
 def alias_check(m):
  assert m.visual.tower is m.core.vision_tower and m.visual.projector is m.core.multi_modal_projector
  assert storage._measure(m.named_parameters(remove_duplicate=False))==storage.layout(m)['weights']
 def prepare(m,cache_limit):
  bridge.bind(m);fv.LIMIT=cache_limit;fv.install(m);alias_check(m)
  assert sorted({type(layer.self_attn).__name__ for layer in m.model.layers})==read(RUN/'prior_parent_COMPLETE.json')['language_attention_classes']
  assert buffer_digest(m)==read(RUN/'prior_parent_COMPLETE.json')['preserved_nonvision_buffers_digest']
 def no_cache(m):
  m._llmcl_visual_cache.clear();m._llmcl_visual_cache_bytes=0;fv.LIMIT=0
 class FiniteScores:
  def __call__(self,input_ids,scores):
   assert not torch.isnan(scores).any() and not torch.isposinf(scores).any() and torch.isfinite(scores).any(dim=-1).all();return scores
 def generate(m,batch,t,fixed=False):
  kwargs=dict(do_sample=False,num_beams=1,max_new_tokens=32 if fixed else t['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[FiniteScores()])
  if fixed:kwargs['min_new_tokens']=32
  with torch.inference_mode(),torch.autocast('cuda',dtype=m.dtype):out=m.generate(**batch,**kwargs)
  return out[0,batch['input_ids'].shape[1]:].tolist()
 def evaluate(m,label,reference):
  m.eval();no_cache(m);before=storage.digest(m);values=[]
  for t,row in rows:
   batch,details=rt.input_batch(row,t,processor,m.device);old=prior_predictions[reference][row['id']]
   assert storage.digest(batch)==old['prepared_sha256']
   assert storage.digest({k:v for k,v in batch.items() if k!='pixel_values'})==old['transport_sha256'] and len(batch['pixel_values'])==old['packed_tokens']
   tokens=generate(m,batch,t);values.append(dict(id=row['id'],task=t['task'],domain=t['domain'],prepared_sha256=storage.digest(batch),transport_sha256=old['transport_sha256'],packed_tokens=len(batch['pixel_values']),prediction=processor.tokenizer.decode(tokens,skip_special_tokens=True),output_tokens=tokens,references=row['answers'],generation_cap=t['max_new_tokens']))
   del batch
  assert storage.digest(m)==before
  write(label+'_predictions.json',values)
 def resource_windows(m,reference):
  m.eval();no_cache(m);measurements=[]
  for domain in ['natural','medical','geography','finance','traffic']:
   t,row=min(((t,row) for t,row in rows if t['domain']==domain),key=lambda tr:hashlib.sha256(tr[1]['id'].encode()).digest())
   raw={}
   def capture(module,args):
    assert not raw;raw.update(raw_pixels_sha256=storage.digest({'pixels':args[0]}),tiles=args[0].shape[0])
   handle=m.core.vision_tower.register_forward_pre_hook(capture)
   gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=torch.cuda.memory_allocated()
   try:batch,_=rt.input_batch(row,t,processor,m.device)
   finally:handle.remove()
   old=prior_predictions[reference][row['id']];assert storage.digest(batch)==old['prepared_sha256'] and raw['raw_pixels_sha256']==old['raw_pixels_sha256']
   tokens=generate(m,batch,t,True);torch.cuda.synchronize();assert len(tokens)==32
   measurements.append(dict(id=row['id'],domain=domain,prepared_sha256=storage.digest(batch),transport_sha256=storage.digest({k:v for k,v in batch.items() if k!='pixel_values'}),text_sha256=storage.digest({'input_ids':batch['input_ids']}),input_shape=list(batch['input_ids'].shape),packed_tokens=len(batch['pixel_values']),new_tokens=32,baseline_allocated=start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),**raw))
   del batch,tokens
  write(case+'_memory.json',measurements)
 if case=='train':
  model=bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda();bridge.bind(model)
  base_language_digest=storage.digest(model.core.language_model)
  assert sha(RUN/'fixed_vision_only.pt')==ready['vision_asset']['sha256'];asset=torch.load(RUN/'fixed_vision_only.pt',map_location='cpu',weights_only=False)
  interface=vi.VisionInterface(vi.student(asset['student_config']),**asset['interface_config']).half().cuda();interface.load_state_dict(asset['state'],strict=True)
  student_config=asset['student_config'];assert storage.digest(interface)==asset['digest'];del asset
  # Match the original allocation run: seed reset immediately before BudgetQV.
  torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  method=adaptation.make_budget(model,ASSET,RUN/'inherited_budget_core.py');adapter_initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
  initial_adapter_digest=storage.digest(adapter_initial)
  receipt=adaptation.install_interface_preserving_adapters(model,interface,method);assert receipt['adapter_coordinates']==4399104 and len(method.layers)==48
  assert storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad})==initial_adapter_digest
  assert not any(isinstance(mod,torch.nn.Dropout) and mod.p>0 for mod in interface.modules()) and interface.student.config.attention_dropout==0
  alias_check(model);frozen=lambda:storage.digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
  frozen_before=frozen();initial_layout=storage.layout(model);assert initial_layout['trainable']==4399104 and initial_layout['weights']['coefficients']==585612160
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();prepare(model,0);method.vision_original=model._llmcl_visual_original
  write('INITIALIZATION.json',dict(base_language_digest=base_language_digest,adapter_initial_digest=initial_adapter_digest,rank_plan=method.rank_plan,initialization=method.initialization,receipt=receipt,layout=initial_layout,frozen_digest=frozen_before,vision_digest=storage.digest(interface),buffer_digest=buffer_digest(model),no_S19_language_loaded=True))
  evaluate(model,'initial','candidate');assert frozen()==frozen_before and storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad})==initial_adapter_digest
  # Initial evaluation and interface construction never set the training RNG.
  random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  fv.LIMIT=128*1024**2
  spec=importlib.util.spec_from_file_location('inherited_train',ASSET/'exp/two_backbone_native_cl_20261017_v2/run.py');formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal)
  training=RUN/'training';training.mkdir();(training/'logs').mkdir();monitor=formal.ResourceMonitor(training);monitor.snapshot('training_before',model=model)
  mapped={};plan=[]
  for t in manifest['tasks']:
   stage=t['task'];mapped[stage,'train']=fit[stage];mapped[stage,'dev']=dev[stage]
   order=sorted(fit[stage],key=lambda row:hashlib.sha256(f"42:{stage}:{row['id']}".encode()).digest());t['training_order_sha256']=hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
   plan.append(dict(stage=stage,QA=len(order),updates=(len(order)+15)//16,order_sha256=t['training_order_sha256']))
  assert sum(p['QA'] for p in plan)==1213 and sum(p['updates'] for p in plan)==87 and plan==ready['training_plan']
  namespace=dict(vars(formal),model=model,method=method,processor=processor,manifest=manifest,data=mapped,run=training,monitor=monitor,args=SimpleNamespace(method=method.name),resume_obj=None)
  counts={};raw_clip=torch.nn.utils.clip_grad_norm_
  def record_grad(parameters,*args,**kwargs):
   parameters=list(parameters);stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
   grads={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None};assert len(grads)==96 and all(torch.isfinite(g).all() for g in grads.values())
   write(f'grad_S{stage:02}_{counts[stage]:03}.json',dict(parameters=len(grads),gradient_bytes=sum(g.numel()*g.element_size() for g in grads.values()),finite=True))
   if counts[stage]==1:torch.save(grads,RUN/f'grad_S{stage:02}_first.pt')
   return raw_clip(parameters,*args,**kwargs)
  def save_stage(stage):
   assert frozen()==frozen_before
   adapter={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad};torch.save(adapter,RUN/f'adapter_S{stage:02}.pt')
   write(f'STAGE_{stage:02}_COMPLETE.json',dict(stage=stage,updates=counts[stage],adapter_digest=storage.digest(adapter),adapter_bytes=sum(t.numel()*t.element_size() for t in adapter.values()),frozen_digest=frozen_before,vision_digest=storage.digest(interface)))
  namespace['save_stage']=save_stage
  prefix,training_ast=study.training_ast();assert training_ast==ready['training_ast_sha256']
  node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_stage(stage)').body,orelse=[])
  random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  torch.nn.utils.clip_grad_norm_=record_grad
  try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<inherited-87-updates>','exec'),namespace)
  finally:torch.nn.utils.clip_grad_norm_=raw_clip
  assert sum(counts.values())==87 and len(counts)==19 and frozen()==frozen_before
  monitor.snapshot('training_after',model=model);memory=monitor.finish(model);model.core.language_model.disable_input_require_grads();model.core.gradient_checkpointing_disable()
  folded=adaptation.fold(model,method);assert len(folded)==48
  package=adaptation.export_package(model,student_config);package.update(digest=storage.digest(model),layout=storage.layout(model))
  assert package['layout']['weights']==dict(coefficients=581213056,bytes=1162426112) and package['layout']['trainable']==0
  cp=RUN/'compact_complete.pt';torch.save(package,cp)
  proof=dict(initial_adapter_digest=initial_adapter_digest,base_language_digest=base_language_digest,frozen_base_unchanged=True,vision_digest=storage.digest(interface),counts=counts,training_plan=plan,training_ast_sha256=training_ast,FIT_QA=1213,updates=87,new_generations=100,DEV_used_for_optimization=0,TEST_QA=0,checkpoint=dict(bytes=cp.stat().st_size,sha256=sha(cp)),digest=package['digest'],layout=package['layout'],fold=folded,training_memory=memory)
 elif case=='compact_eval':
  end=read(RUN/'train_COMPLETE.json');cp=RUN/'compact_complete.pt';assert cp.stat().st_size==end['checkpoint']['bytes'] and sha(cp)==end['checkpoint']['sha256']
  package=torch.load(cp,map_location='cpu',weights_only=False);model=adaptation.load_export(package,'cuda');assert storage.digest(model)==package['digest']==end['digest'] and storage.layout(model)==package['layout'];del package
  prepare(model,0);before=storage.digest(model);resource_windows(model,'candidate');evaluate(model,'final','candidate');assert storage.digest(model)==before
  proof=dict(digest=before,layout=storage.layout(model),new_generations=105,DEV_used_for_optimization=0,TEST_QA=0,strict_independent_static_reload=True,buffer_digest=buffer_digest(model))
 else:
  cp=RUN/'wide_parent_static.pt';assert cp.stat().st_size==ready['parent']['bytes'];package=torch.load(cp,map_location='cpu',weights_only=False)
  model=bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True);model.core.load_state_dict(package['state'],strict=True);model.cuda().requires_grad_(False);model.eval();assert storage.digest(model)==ready['parent']['digest'];del package
  prepare(model,0);before=storage.digest(model);resource_windows(model,'parent');assert storage.digest(model)==before
  proof=dict(digest=before,layout=storage.layout(model),new_generations=5,DEV_used_for_optimization=0,TEST_QA=0,buffer_digest=buffer_digest(model))
 import resource
 files=[p for p in RUN.iterdir() if p.is_file() and (p.name.startswith(('initial_predictions','final_predictions','INITIALIZATION','grad_','adapter_','STAGE_')) or p.name==case+'_memory.json')]
 if case=='train':files += [p for p in (RUN/'training').rglob('*') if p.is_file()]
 proof.update(identity=who,ready_sha256=sha(RUN/'READY.json'),files={p.relative_to(RUN).as_posix():sha(p) for p in files},process_RSS_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,last_reset_GPU_peak_allocated=torch.cuda.max_memory_allocated())
 proof['seconds']=time.monotonic()-tick
 write(case+'_COMPLETE.json',proof)
if __name__=='__main__':
 case=sys.argv[1];assert case in ['train','compact_eval','parent_resource']
 try:main(case)
 except Exception:write(case+'_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc()));raise
