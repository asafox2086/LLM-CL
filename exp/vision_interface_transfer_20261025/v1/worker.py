"""One fixed image-only fit, or one fresh-process full-graph evaluation."""
from pathlib import Path
import gc,hashlib,importlib.util,json,os,signal,sys,time,traceback
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN']);ASSET=ROOT/'.deployment/paired_image_support_20261024_view'
DOMAINS=('natural','medical','geography','finance','traffic')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(n,v):
 with (RUN/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main(case):
 ready=read(RUN/'READY.json')
 for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
 for n,h in ready['runtime_files'].items():assert sha(ASSET/n)==h,n
 assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
 spec=importlib.util.spec_from_file_location('identity_only',ASSET/'exp/paired_image_support_20261024/v1/common.py');ids=importlib.util.module_from_spec(spec);spec.loader.exec_module(ids)
 card=ids.inventory()[1];assert not card['owners'] and card['memory_MiB']<500 and card['utilization']<10
 assert not (RUN/(case+'_STARTED.json')).exists();tick=time.monotonic();who=ids.identity(os.getpid());assert who['pid']==who['pgid']
 write(case+'_STARTED.json',dict(unix=time.time(),identity=who,GPU=1,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),ready_sha256=sha(RUN/'READY.json')))
 def timeout(*args):raise TimeoutError('Fixed phase1200s budget')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
 os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
 import torch
 from transformers import AutoProcessor,LlavaOnevisionConfig,LlavaOnevisionForConditionalGeneration
 import vision_interface as vi
 sys.path.insert(0,str(ASSET/'exp/two_backbone_native_cl_20261017_v2'));sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
 import onevision_bridge as bridge,runtime_utils as rt,frozen_vision as fv,study
 spec=importlib.util.spec_from_file_location('static_storage',ROOT/'exp/ffn_static_delete_20261025/v1/core.py');storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
 torch.set_num_threads(8);torch.manual_seed(42);fv.LIMIT=0
 base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV';processor=AutoProcessor.from_pretrained(base,local_files_only=True)
 manifest_path,manifest,original=study.original_train();split,fit,dev=study.split_train(manifest,original)
 assert study.sha(manifest_path)==ready['manifest_sha256'] and study.sha(study.REPORT/'SPLIT.json')==ready['split_sha256'] and split==read(study.REPORT/'SPLIT.json')
 tasks={t['task']:t for t in manifest['tasks']};s1={r['id']:r for r in fit[1]}
 groups=[]
 for gid,g in split['groups'].items():
  selected=sorted(set(g['ids']).intersection(s1))
  if selected:assert not g['dev'];groups.append([gid,selected[0]])
 groups.sort(key=lambda g:hashlib.sha256(('vision-interface-20261025:'+g[0]).encode()).digest());assert groups==ready['groups'] and len(groups)==325
 def alias_check(m):
  assert m.visual.tower is m.core.vision_tower and m.visual.projector is m.core.multi_modal_projector
  assert storage._measure(m.named_parameters(remove_duplicate=False))==storage.layout(m)['weights']
 def parent():
  cp=RUN/'wide_parent_static.pt';assert cp.stat().st_size==ready['parent']['bytes']
  p=torch.load(cp,map_location='cpu',weights_only=False)
  m=bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True)
  m.core.load_state_dict(p['state'],strict=True);m.cuda().requires_grad_(False);m.eval()
  assert storage.digest(m)==ready['parent']['digest']==p['digest'] and storage.layout(m)==p['layout']
  assert m.core.language_model.config._attn_implementation=='sdpa'
  alias_check(m);bridge.bind(m);fv.install(m);return m
 def candidate():
  proof=read(RUN/'fit_COMPLETE.json');cp=RUN/'candidate_complete.pt';assert cp.stat().st_size==proof['checkpoint']['bytes'] and sha(cp)==proof['checkpoint']['sha256']
  p=torch.load(cp,map_location='cpu',weights_only=False)
  # Construct from config/state only; do not load original vision weights.
  cfg=LlavaOnevisionConfig(**p['base_config']);cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
  core=LlavaOnevisionForConditionalGeneration(cfg);m=bridge.OneVisionBridge(core)
  interface=vi.VisionInterface(vi.student(p['student_config']));vi.install(m,interface)
  # Keep derived FP32 language buffers (notably nonpersistent RoPE) intact.
  with torch.no_grad():
   for parameter in m.core.parameters():parameter.data=parameter.data.to(dtype=torch.float16)
  m.core.load_state_dict(p['state'],strict=True);m.cuda().requires_grad_(False);m.eval()
  assert storage.digest(m)==p['digest']==proof['digest'] and storage.layout(m)==p['layout']==proof['layout']
  assert m.core.config.vision_config.image_size==384 and m.core.config.vision_config.patch_size==14
  assert m.core.language_model.config._attn_implementation=='sdpa'
  assert sorted({type(layer.self_attn).__name__ for layer in m.model.layers})==read(RUN/'parent_COMPLETE.json')['language_attention_classes']
  assert m.core.vision_tower.student.config._attn_implementation=='eager'
  alias_check(m);bridge.bind(m);fv.install(m);return m
 checked=set()
 def inputs(m,t,row):
  bridge.bind(m)
  for n in row['images']:
   if n not in checked:assert sha(ASSET/n)==split['image_sha256'][n];checked.add(n)
  batch,_=rt.input_batch(row,t,processor,m.device);assert m._llmcl_visual_cache_bytes==0;return batch
 if case=='fit':
  assert sha(RUN/'wide_parent_static.pt')==ready['parent']['sha256'] and sha(RUN/'student_vision_fp16.pt')==ready['student']['sha256']
  model=parent();before=storage.digest(model)
  def preserved_digest():return storage.digest({n:v for n,v in model.core.state_dict().items() if not n.startswith(('vision_tower.','multi_modal_projector.'))})
  preserved_before=preserved_digest();p=torch.load(RUN/'student_vision_fp16.pt',map_location='cpu',weights_only=False)
  interface=vi.VisionInterface(vi.student(p['config'])).half().cuda().eval().requires_grad_(False);interface.student.load_state_dict(p['state'],strict=True)
  assert sum(t.numel() for t in interface.student.parameters())==86433024
  student_before=storage.digest(interface.student);moments=vi.Moments(768,896);pending={};records=[];mode='fit'
  def pre(module,args):
   assert not pending
   pixels=args[0];pending.update(x=vi.spatial_features(interface.student,pixels),raw_pixels_sha256=storage.digest({'pixels':pixels}),tiles=pixels.shape[0])
  def post(module,args,y):
   x=pending['x'];assert x.shape[:2]==y.shape[:2] and y.shape[-1]==896
   item=dict(raw_pixels_sha256=pending['raw_pixels_sha256'],tiles=pending['tiles'],all_crop_positions=x.shape[0]*x.shape[1],mode=mode)
   if mode=='fit':moments.add(x,y)
   else:
    pred=interface.affine(x);error=(pred.float()-y.float()).square().mean();energy=y.float().square().mean()
    assert torch.isfinite(error) and torch.isfinite(energy) and energy>0
    item.update(MSE=float(error),target_energy=float(energy),ratio=float(error/energy))
   records.append(item);pending.clear()
  handles=[model.core.vision_tower.register_forward_pre_hook(pre),model.core.multi_modal_projector.register_forward_hook(post)]
  try:
   for i,(gid,rid) in enumerate(groups[:293]):
    row=s1[rid];count=len(records)
    with torch.inference_mode():batch=inputs(model,tasks[1],row)
    assert len(records)==count+1 and not pending
    records[-1].update(id=rid,group=gid,packed_tokens=len(batch['pixel_values']))
    del batch
   assert moments.count==293
   weight,bias,ridge=moments.solve()
   with torch.no_grad():interface.affine.weight.copy_(weight.to(interface.affine.weight));interface.affine.bias.copy_(bias.to(interface.affine.bias))
   mode='diagnostic'
   for gid,rid in groups[293:]:
    row=s1[rid];count=len(records)
    with torch.inference_mode():batch=inputs(model,tasks[1],row)
    assert len(records)==count+1 and not pending
    records[-1].update(id=rid,group=gid,packed_tokens=len(batch['pixel_values']));del batch
  finally:
   for h in handles:h.remove()
  assert len(records)==325 and storage.digest(model)==before and storage.digest(interface.student)==student_before
  assert all(p.grad is None and not p.requires_grad for p in model.parameters()) and all(p.grad is None for p in interface.parameters())
  write('feature_records.json',records)
  # Remove the frozen_vision bound hook before changing the producer's graph.
  model.visual.forward=model._llmcl_visual_original;del model._llmcl_visual_original
  vi.install(model,interface);alias_check(model)
  assert preserved_digest()==preserved_before
  layout=storage.layout(model);assert layout['weights']['coefficients']==581213056 and layout['weights']['bytes']==1162426112 and layout['trainable']==0
  assert not any(n.startswith('multi_modal_projector.') for n in model.core.state_dict())
  assert all(n.startswith('vision_tower.student.') or n.startswith('vision_tower.affine.') for n in model.core.state_dict() if n.startswith('vision_tower.'))
  package=dict(state=storage.state_cpu(model),digest=storage.digest(model),layout=layout,base_config=model.core.config.to_dict(),student_config=p['config'])
  cp=RUN/'candidate_complete.pt';torch.save(package,cp);assert cp.stat().st_size<ready['parent']['bytes']
  proof=dict(identity=who,seconds=time.monotonic()-tick,teacher_vision_QA=325,student_vision_QA=325,teacher_projector_calls=325,decoder_requests=0,calibration_QA=293,diagnostic_QA=32,backwards=0,optimizer_updates=0,ridge=ridge,layout=layout,digest=package['digest'],student_digest=student_before,language_unchanged=True,preserved_nonvision_digest=preserved_before,checkpoint=dict(bytes=cp.stat().st_size,sha256=sha(cp)),feature_records_sha256=sha(RUN/'feature_records.json'))
 else:
  model=parent() if case=='parent' else candidate();before=storage.digest(model);raw={}
  def capture(module,args):
   assert not raw;raw.update(raw_pixels_sha256=storage.digest({'pixels':args[0]}),tiles=args[0].shape[0])
  handle=model.core.vision_tower.register_forward_pre_hook(capture)
  class FiniteScores:
   def __call__(self,input_ids,scores):
    assert not torch.isnan(scores).any() and not torch.isposinf(scores).any() and torch.isfinite(scores).any(dim=-1).all();return scores
  rows=[(t,row) for t in manifest['tasks'] for row in dev[t['task']]];assert len(rows)==100
  predictions=[];measurements=[]
  try:
   for t,row in rows:
    raw.clear();batch=inputs(model,t,row);assert raw
    with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
     loss,n=rt.answer_loss(model,batch,row['answer'],processor);assert torch.isfinite(loss)
     out=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=t['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[FiniteScores()])
    tokens=out[0,batch['input_ids'].shape[1]:].tolist()
    predictions.append(dict(id=row['id'],task=t['task'],domain=t['domain'],transport_sha256=storage.digest({k:v for k,v in batch.items() if k!='pixel_values'}),prepared_sha256=storage.digest(batch),packed_tokens=len(batch['pixel_values']),NLL=float(loss),target_tokens=n,output_tokens=tokens,prediction=processor.tokenizer.decode(tokens,skip_special_tokens=True),references=row['answers'],generation_cap=t['max_new_tokens'],**raw))
    with (RUN/(case+'_events.jsonl')).open('a') as f:f.write(json.dumps(dict(completed_DEV=len(predictions)))+'\n');f.flush()
    del batch,out,loss
   write(case+'_predictions.json',predictions)
   for d in DOMAINS:
    t,row=min(((t,row) for t,row in rows if t['domain']==d),key=lambda tr:hashlib.sha256(tr[1]['id'].encode()).digest())
    raw.clear();bridge.LAST_NATIVE=None;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
    with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
     batch=inputs(model,t,row)
     out=model.generate(**batch,do_sample=False,num_beams=1,min_new_tokens=32,max_new_tokens=32,use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[FiniteScores()])
    torch.cuda.synchronize();assert out.shape[1]-batch['input_ids'].shape[1]==32 and raw
    measurements.append(dict(id=row['id'],domain=d,transport_sha256=storage.digest({k:v for k,v in batch.items() if k!='pixel_values'}),new_tokens=32,baseline_allocated=baseline,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),**raw))
    del batch,out
  finally:handle.remove()
  assert storage.digest(model)==before and all(p.grad is None and not p.requires_grad for p in model.parameters())
  write(case+'_memory.json',measurements)
  proof=dict(identity=who,seconds=time.monotonic()-tick,NLL_requests=100,new_generations=105,backwards=0,optimizer_updates=0,digest=before,layout=storage.layout(model),files={n:sha(RUN/n) for n in [case+'_predictions.json',case+'_memory.json']})
 import resource
 proof.update(ready_sha256=sha(RUN/'READY.json'),language_attention_classes=sorted({type(layer.self_attn).__name__ for layer in model.model.layers}),language_attention_backend=model.core.language_model.config._attn_implementation,preserved_nonvision_buffers_digest=storage.digest({n:v for n,v in model.core.named_buffers() if not n.startswith(('vision_tower.','multi_modal_projector.'))}),process_RSS_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 if case=='fit':proof['whole_fit_GPU_peak_allocated']=torch.cuda.max_memory_allocated()
 write(case+'_COMPLETE.json',proof)
if __name__=='__main__':
 case=sys.argv[1];assert case in ('fit','parent','candidate')
 try:main(case)
 except Exception:write(case+'_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc()));raise
