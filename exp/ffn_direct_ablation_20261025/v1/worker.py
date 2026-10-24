"""Finite, forward-only FFN interventions on one legal stronger static parent."""
from pathlib import Path
import collections,gc,hashlib,importlib.util,json,os,signal,sys,time,traceback
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['LLMCL_ROOT'])
RUN=Path(os.environ['LLMCL_RUN'])
ASSET=ROOT/'.deployment/paired_image_support_20261024_view'
PORT=ASSET/'exp/two_backbone_native_cl_20261017_v2'
DOMAINS=('natural','medical','geography','finance','traffic')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main(gpu):
 ready=json.loads((RUN/'READY.json').read_bytes())
 assert ready['admitted'] and sha(HERE/'worker.py')==ready['worker_sha256']
 assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
 assert sha(RUN/'PROTOCOL.md')==ready['protocol_sha256']
 sys.path.insert(0,str(ASSET/'exp/paired_image_support_20261024/v1'))
 spec=importlib.util.spec_from_file_location('identity_only',ASSET/'exp/paired_image_support_20261024/v1/common.py')
 ids=importlib.util.module_from_spec(spec);spec.loader.exec_module(ids)
 cards=ids.inventory();assert not cards[gpu]['owners'] and cards[gpu]['memory_MiB']<500 and cards[gpu]['utilization']<10
 assert not (RUN/'STARTED.json').exists()
 started=time.monotonic();who=ids.identity(os.getpid());write(RUN/'STARTED.json',dict(unix=time.time(),identity=who,GPU=gpu,ready_sha256=sha(RUN/'READY.json')))
 def timeout(*a):raise TimeoutError('Fixed5400s diagnostic limit')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(5400)
 os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
 import torch
 from transformers import AutoProcessor
 sys.path.insert(0,str(PORT));sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
 import onevision_bridge as bridge,runtime_utils as rt,frozen_vision as fv,study
 spec=importlib.util.spec_from_file_location('static_storage',ROOT/'exp/ffn_static_delete_20261025/v1/core.py');storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
 torch.set_num_threads(8);torch.manual_seed(42);fv.LIMIT=0
 base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV';processor=AutoProcessor.from_pretrained(base,local_files_only=True)
 _,manifest,original=study.original_train();split,fit,dev=study.split_train(manifest,original)
 assert sha(study.REPORT/'SPLIT.json')==ready['split_sha256'] and split==json.loads((study.REPORT/'SPLIT.json').read_bytes())
 tasks={t['task']:t for t in manifest['tasks']}
 calibration=[(t,r) for k,t in tasks.items() for r in sorted(fit[k],key=lambda r:hashlib.sha256(('FFN-delete-20261025:'+r['id']).encode()).digest())[:2]]
 assert len(calibration)==37
 assert {str(k):[r['id'] for t,r in calibration if t['task']==k] for k in tasks}==ready['calibration_ids']
 devrows=[(t,r) for k,t in tasks.items() for r in dev[k]];assert len(devrows)==100
 path=RUN/'wide_parent_static.pt';assert path.stat().st_size==ready['parent']['bytes'] and sha(path)==ready['parent']['sha256']
 def release():
  bridge.ACTIVE=None;bridge.LAST_NATIVE=None;gc.collect();torch.cuda.empty_cache()
 def fresh():
  m=bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
  m.requires_grad_(False);m.eval();bridge.bind(m);fv.install(m);return m
 def load(p):
  package=torch.load(p,map_location='cpu',weights_only=False);m=fresh()
  if package['indexes']:storage.delete(m,package['indexes'])
  m.core.load_state_dict(package['state'],strict=True)
  assert storage.digest(m)==package['digest'] and storage.layout(m)==package['layout']
  for i in package['indexes']:assert not any(f'model.layers.{i}.mlp.' in n for n in m.core.state_dict())
  return m
 checked=set()
 def inputs(m,t,r):
  bridge.bind(m)
  for name in r['images']:
   if name not in checked:assert sha(ASSET/name)==split['image_sha256'][name];checked.add(name)
  batch,_=rt.input_batch(r,t,processor,m.device);assert m._llmcl_visual_cache_bytes==0
  return batch
 requests=0;phase='locator';qa_id=None;layer_index=None
 def event(v):
  with (RUN/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
 def loss(m,batch,row):
  nonlocal requests
  with torch.inference_mode(),torch.autocast('cuda',dtype=m.dtype):v,n=rt.answer_loss(m,batch,row['answer'],processor)
  requests+=1
  if not torch.isfinite(v):raise FloatingPointError(f'Nonfinite forward NLL phase={phase} id={row["id"]} layer={layer_index}')
  return float(v),n
 model=load(path);before=storage.digest(model);assert storage.layout(model)['weights']['coefficients']==893675552
 matrix=[]
 for t,row in calibration:
  qa_id=row['id'];batch=inputs(model,t,row);batch_digest=storage.digest(batch)
  layer_index=None;full,n=loss(model,batch,row);values=[]
  for i in range(24):
   layer_index=i;original_mlp=model.model.layers[i].mlp
   try:
    model.model.layers[i].mlp=storage.ZeroFFN()
    value,nn=loss(model,batch,row);assert nn==n;values.append(value)
   finally:model.model.layers[i].mlp=original_mlp
   del original_mlp
  assert storage.digest(batch)==batch_digest
  record=dict(id=qa_id,task=t['task'],domain=t['domain'],input_sha256=batch_digest,full_NLL=full,target_tokens=n,single_deleted_NLL=values,delta_NLL=[v-full for v in values])
  matrix.append(record);event(dict(phase='locator',completed_QA=len(matrix),forward_NLL_requests=requests))
  del batch
 assert requests==925 and storage.digest(model)==before and not any(p.grad is not None for p in model.parameters())
 costs={d:[sum(max(0.,x['delta_NLL'][i]) for x in matrix if x['domain']==d)/sum(x['domain']==d for x in matrix) for i in range(24)] for d in DOMAINS}
 available=set(range(24));selected=[];cumulative={d:0. for d in DOMAINS}
 for _ in range(8):
  i=min(available,key=lambda i:(max(cumulative[d]+costs[d][i] for d in DOMAINS),i));selected.append(i);available.remove(i)
  for d in DOMAINS:cumulative[d]+=costs[d][i]
 selected.sort();write(RUN/'selection.json',dict(selected=selected,matrix=matrix,domain_costs=costs,cumulative_proxy=cumulative,single_effects_not_additive_joint_proof=True))
 removed=storage.delete(model,selected);assert removed['coefficients']==104595456 and removed['bytes']==209190912
 assert storage.layout(model)['weights']['coefficients']==789080096
 state=storage.state_cpu(model);pruned=RUN/'pruned_complete.pt'
 package=dict(state=state,indexes=selected,digest=storage.digest(state),layout=storage.layout(model));torch.save(package,pruned)
 pruned_meta=dict(path=pruned.name,bytes=pruned.stat().st_size,sha256=sha(pruned),layout=package['layout'])
 del model,state,package;release();assert torch.cuda.memory_allocated()<1024**2
 generations=0;bindings={};metadata={};memory={}
 class FiniteScores:
  def __call__(self,input_ids,scores):
   assert not torch.isnan(scores).any() and not torch.isposinf(scores).any() and torch.isfinite(scores).any(dim=-1).all()
   return scores
 for name,p in [('parent',path),('pruned',pruned)]:
  phase=name;model=load(p);before=storage.digest(model);records=[]
  for t,row in devrows:
   qa_id=row['id'];layer_index=None;batch=inputs(model,t,row);h=storage.digest(batch)
   if qa_id in bindings:assert bindings[qa_id]==h
   else:bindings[qa_id]=h
   nll,n=loss(model,batch,row)
   with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
    out=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=t['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[FiniteScores()])
   tokens=out[0,batch['input_ids'].shape[1]:].tolist()
   records.append(dict(id=qa_id,task=t['task'],domain=t['domain'],input_sha256=h,NLL=nll,target_tokens=n,output_tokens=tokens,prediction=processor.tokenizer.decode(tokens,skip_special_tokens=True),references=row['answers'],generation_cap=t['max_new_tokens']))
   generations+=1;event(dict(phase=phase,completed_DEV=len(records),new_generations=generations,NLL_requests=requests));del batch,out
  write(RUN/(name+'_predictions.json'),records)
  measurements=[]
  for d in DOMAINS:
   t,row=min(((t,row) for t,row in devrows if t['domain']==d),key=lambda tr:hashlib.sha256(tr[1]['id'].encode()).digest())
   release();bridge.bind(model);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
   with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
    batch=inputs(model,t,row)
    out=model.generate(**batch,do_sample=False,num_beams=1,min_new_tokens=32,max_new_tokens=32,use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[FiniteScores()])
   torch.cuda.synchronize();assert out.shape[1]-batch['input_ids'].shape[1]==32
   measurements.append(dict(id=row['id'],domain=d,input_sha256=storage.digest(batch),prompt_tokens=batch['input_ids'].shape[1],new_tokens=32,baseline_allocated=baseline,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
   generations+=1;del batch,out
  memory[name]=measurements;write(RUN/(name+'_memory.json'),measurements)
  assert storage.digest(model)==before and not any(p.grad is not None for p in model.parameters())
  metadata[name]=dict(layout=storage.layout(model),digest=before,DEV=100)
  del model;release();assert torch.cuda.memory_allocated()<1024**2
 assert requests==1125 and generations==210
 assert [(x['id'],x['input_sha256'],x['prompt_tokens']) for x in memory['parent']]==[(x['id'],x['input_sha256'],x['prompt_tokens']) for x in memory['pruned']]
 write(RUN/'COMPLETE.json',dict(complete=True,identity=who,seconds=time.monotonic()-started,ready_sha256=sha(RUN/'READY.json'),selected=selected,parent=ready['parent'],pruned=pruned_meta,metadata=metadata,NLL_requests=requests,new_generations=generations,backwards=0,training_updates=0,TEST_QA=0,files={p.name:sha(p) for p in RUN.glob('*.json') if p.name not in ['READY.json','PARENT_EXPORT.json']}))
 signal.alarm(0)
if __name__=='__main__':
 try:main(int(sys.argv[1]))
 except Exception:write(RUN/'FAILURE.json',dict(traceback=traceback.format_exc(),no_quality_claim=True));raise
