"""One fixed FIT-located physical FFN deletion diagnostic, never TEST or training."""
from pathlib import Path
import ast,collections,gc,hashlib,importlib.util,json,os,pickle,signal,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REPORT=ROOT/'summary/ffn_static_delete_20261025/v1'
WORK=HERE/'private'
ASSET=ROOT/'.deployment/paired_image_support_20261024_view'
PORT=ASSET/'exp/two_backbone_native_cl_20261017_v2'
BASE=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
DOMAINS=('natural','medical','geography','finance','traffic')
BUDGET=10800
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main(gpu):
 ready=json.loads((REPORT/'READY.json').read_bytes())
 assert ready['admitted'] and all(sha(ROOT/n)==h for n,h in ready['sources'].items())
 assert sha(REPORT/'PROTOCOL.md')==ready['protocol_sha256']
 sys.path.insert(0,str(ASSET/'exp/paired_image_support_20261024/v1'))
 spec=importlib.util.spec_from_file_location('old_pair_common',ASSET/'exp/paired_image_support_20261024/v1/common.py')
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 cards=old.inventory();assert not cards[gpu]['owners'] and cards[gpu]['memory_MiB']<500 and cards[gpu]['utilization']<10
 assert not (REPORT/'STARTED.json').exists() and not WORK.exists()
 start=time.monotonic();who=old.identity(os.getpid());assert who['pid']==who['pgid']
 write(REPORT/'STARTED.json',dict(identity=who,GPU=gpu,unix=time.time(),ready_sha256=sha(REPORT/'READY.json'),inventory=cards))
 def timeout(*a):raise TimeoutError('Fixed 10800s model-work limit; no restart')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(BUDGET)
 os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
 import torch,numpy as np
 from transformers import AutoProcessor
 from core import GatedFFN,ZeroFFN,delete,digest,layout,state_cpu
 sys.path.insert(0,str(PORT));sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
 import onevision_bridge as bridge,runtime_utils as rt,frozen_vision as fv,study
 assert study.ROOT==ASSET and rt.ROOT==ASSET
 torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42)
 processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
 _,manifest,original=study.original_train();split,fit,dev=study.split_train(manifest,original)
 assert sha(study.REPORT/'SPLIT.json')==ready['split_sha256'] and split==json.loads((study.REPORT/'SPLIT.json').read_bytes())
 tasks={t['task']:t for t in manifest['tasks']}
 calibration=[(t,sorted(fit[k],key=lambda r:hashlib.sha256(('FFN-delete-20261025:'+r['id']).encode()).digest())[:2]) for k,t in tasks.items()]
 assert sum(len(rows) for _,rows in calibration)==38 and all(len(rows)==2 for _,rows in calibration)
 devrows=[(t,r) for k,t in tasks.items() for r in dev[k]];assert len(devrows)==100
 WORK.mkdir();write(WORK/'selection_ids.json',dict(calibration={str(t['task']):[r['id'] for r in rr] for t,rr in calibration},DEV=[r['id'] for _,r in devrows]))
 fv.LIMIT=0 # No feature cache in any condition, including matched GPU-memory workloads.
 def clear():
  bridge.ACTIVE=None;bridge.LAST_NATIVE=None;gc.collect();torch.cuda.empty_cache()
 def fresh():
  model=bridge.Factory.from_pretrained(BASE,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
  model.requires_grad_(False);model.eval();bridge.bind(model);fv.install(model);return model
 def save(model,path,selected):
  state=state_cpu(model);v=dict(state=state,indexes=selected,digest=digest(state),layout=layout(model))
  torch.save(v,path);return dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path),digest=v['digest'],layout=v['layout'])
 def load(path):
  p=torch.load(path,map_location='cpu',weights_only=False);model=fresh()
  if p['indexes']:delete(model,p['indexes'])
  model.core.load_state_dict(p['state'],strict=True)
  assert digest(model)==p['digest'] and layout(model)==p['layout']
  assert not any(f'model.layers.{i}.mlp.' in n for n in model.core.state_dict() for i in p['indexes'])
  return model
 def inputs(model,t,r):
  bridge.bind(model)
  for name in r['images']:assert sha(ASSET/name)==split['image_sha256'][name]
  batch,details=rt.input_batch(r,t,processor,model.device)
  assert model._llmcl_visual_cache_bytes==0
  return batch,dict(input_sha256=digest(batch),**details)
 # Bind existing legal checkpoint by original small proof/metadata, never rehash the old large object.
 screen=study.REPORT;complete=json.loads((screen/'LoRA_qv_COMPLETE.json').read_bytes())
 assert sha(screen/'LoRA_qv_COMPLETE.json')==ready['parent_complete_sha256']
 proof=screen/'sealed_case_reports_20261022/LoRA_qv_VERIFIED.json'
 assert sha(proof)==ready['parent_proof_sha256']
 pv=json.loads(proof.read_bytes())
 assert pv['case']=='LoRA_qv' and pv['original_independent_verifier_passed']
 assert pv['checkpoint']==complete['checkpoint'] and pv['complete_marker_sha256']==sha(screen/'LoRA_qv_COMPLETE.json')
 assert pv['checkpoint']['stage']==19 and pv['checkpoint']['method']=='LoRA_qv'
 assert complete['checkpoint_root']=='exp/osft_increment_screen_20261019/v1/private/LoRA_qv'
 cp=ASSET/complete['checkpoint_root']/complete['checkpoint']['path'];assert cp.stat().st_size==complete['checkpoint']['bytes']
 obj=torch.load(cp,map_location='cpu',pickle_module=pickle,weights_only=False)
 assert obj['stage']==19 and obj['format']=='llmcl-method-graph-v1'
 assert obj['config']['FIT_QA']==1213 and obj['config']['DEV_QA']==100 and obj['config']['TEST_QA']==0
 assert obj['config']['split_sha256']==ready['split_sha256'] and obj['config']['method']=='LoRA_qv'
 assert obj['config']['manifest_sha256']==complete['checkpoint']['manifest_sha256']
 source=obj['method'].model;source.requires_grad_(False);source.cuda();source.eval()
 # Common ordinary FP16 adapter merge; compare every deletion to this same static parent.
 from methods import SeqLinear
 folded=[]
 for name,module in list(source.named_modules()):
  if not isinstance(module,SeqLinear):continue
  n,m=module.weight.shape
  linear=torch.nn.Linear(m,n,bias=module.bias is not None,device=module.weight.device,dtype=module.weight.dtype)
  with torch.no_grad():
   linear.weight.copy_((module.weight.float()+module.scale*(module.B.weight.float()@module.A.weight.float())).to(module.weight.dtype))
   if module.bias is not None:linear.bias.copy_(module.bias)
  prefix,child=name.rsplit('.',1);setattr(source.get_submodule(prefix),child,linear);folded.append(name)
 assert len(folded)==48
 state=state_cpu(source);del source,obj,module,linear;clear()
 model=fresh();model.core.load_state_dict(state,strict=True);del state;gc.collect()
 assert layout(model)['weights']['coefficients']==893675552
 s19=save(model,WORK/'S19_full.pt',[])
 # Locate two fixed deletion sets, without changing any original weight.
 base_digest=digest(model);gates=[];handles=[];residual={};bi={};grads=[];scores=[]
 for i,layer in enumerate(model.model.layers):
  layer.mlp=GatedFFN(layer.mlp);gates.append(layer.mlp.gate)
  def pre(module,args,i=i):residual[i]=args[0].detach()
  def post(module,args,out,i=i):
   raw=residual.pop(i);y=(raw+out.detach()).float();x=raw.float()
   bi[i]=float((1-torch.nn.functional.cosine_similarity(x,y,dim=-1,eps=1e-8)).mean())
  handles += [layer.post_attention_layernorm.register_forward_pre_hook(pre),layer.mlp.register_forward_hook(post)]
 for t,rows in calibration:
  for r in rows:
   for g in gates:g.grad=None
   batch,_=inputs(model,t,r);bi.clear()
   with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast('cuda',dtype=model.dtype):loss,n=rt.answer_loss(model,batch,r['answer'],processor)
   loss.backward();gg=[float(g.grad) for g in gates]
   assert len(bi)==24 and all(np.isfinite(x) for x in gg) and torch.isfinite(loss)
   assert all(p.grad is None for name,p in model.named_parameters() if not name.endswith('.gate'))
   grads.append(dict(id=r['id'],task=t['task'],domain=t['domain'],NLL=float(loss),target_tokens=n,gate_gradient=gg,BI=[bi[i] for i in range(24)]))
   del batch,loss
 for h in handles:h.remove()
 for layer in model.model.layers:layer.mlp=layer.mlp.original
 del layer
 del gates,handles,residual,bi,g;assert digest(model)==base_digest
 domains={d:[x for x in grads if x['domain']==d] for d in DOMAINS};assert all(domains.values())
 costs={d:[sum(max(0.,-r['gate_gradient'][i]) for r in rr)/len(rr) for i in range(24)] for d,rr in domains.items()}
 remaining=set(range(24));chosen=[];cumulative={d:0. for d in DOMAINS}
 for _ in range(8):
  i=min(remaining,key=lambda i:(max(cumulative[d]+costs[d][i] for d in DOMAINS),i))
  chosen.append(i);remaining.remove(i)
  for d in DOMAINS:cumulative[d]+=costs[d][i]
 bival=[sum(sum(r['BI'][i] for r in rr)/len(rr) for rr in domains.values())/5 for i in range(24)]
 selected={'worst':sorted(chosen),'BI':sorted(sorted(range(24),key=lambda i:(bival[i],i))[:8])}
 write(WORK/'selection.json',dict(sets=selected,calibration=grads,domain_costs=costs,BI_macro=bival,cumulative_proxy=cumulative,not_deletion_evidence=True))
 del model,grads,domains;clear()
 assert torch.cuda.memory_allocated()<1024**2,'Unexpected retained model CUDA tensors after locator release'
 model=fresh();s0=save(model,WORK/'S0_full.pt',[]);del model;clear()
 assert torch.cuda.memory_allocated()<1024**2
 results={};packages={};binding={};fixed_memory={};new_generations=0
 for stage in ('S0','S19'):
  mappings={'full':[]};mappings.update(selected)
  seen={}
  for condition,indexes in mappings.items():
   key=stage+'_'+condition
   if tuple(indexes) in seen:
    results[key]=dict(reused_identical_condition=seen[tuple(indexes)]);continue
   seen[tuple(indexes)]=key
   model=load(WORK/(stage+'_full.pt'))
   if indexes:
    removed=delete(model,indexes);assert removed['coefficients']==104595456 and removed['bytes']==209190912
    path=WORK/(key+'.pt');packages[key]=save(model,path,indexes);del model;clear();model=load(path)
   else:packages[key]=s0 if stage=='S0' else s19
   before=digest(model);records=[];fixed=[]
   for t,r in devrows:
    with torch.inference_mode():
     batch,detail=inputs(model,t,r)
     identity=digest(batch)
     if r['id'] in binding:assert identity==binding[r['id']]
     else:binding[r['id']]=identity
     with torch.autocast('cuda',dtype=model.dtype):
      loss,n=rt.answer_loss(model,batch,r['answer'],processor)
      assert torch.isfinite(loss)
      output=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=t['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
     ids=output[0,batch['input_ids'].shape[1]:].tolist();prediction=processor.tokenizer.decode(ids,skip_special_tokens=True)
     records.append(dict(id=r['id'],task=t['task'],domain=t['domain'],NLL=float(loss),target_tokens=n,prediction=prediction,references=r['answers'],input_sha256=identity,output_tokens=ids,F1=rt.score(prediction,r['answers'])['token_f1']))
     new_generations+=1;del batch,output,loss
   # Matched batch1, one fixed hash-selected DEV input/domain, exactly32 generated tokens.
   for d in DOMAINS:
    t,r=min(((t,r) for t,r in devrows if t['domain']==d),key=lambda tr:hashlib.sha256(tr[1]['id'].encode()).digest())
    clear();bridge.bind(model);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    baseline=torch.cuda.memory_allocated()
    with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
     batch,details=inputs(model,t,r)
     output=model.generate(**batch,do_sample=False,num_beams=1,min_new_tokens=32,max_new_tokens=32,use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
    torch.cuda.synchronize();assert output.shape[1]-batch['input_ids'].shape[1]==32
    fixed.append(dict(id=r['id'],domain=d,prompt_tokens=batch['input_ids'].shape[1],input_sha256=digest(batch),new_tokens=32,baseline_allocated=baseline,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    new_generations+=1;del batch,output
   assert digest(model)==before and not any(p.grad is not None for p in model.parameters())
   write(WORK/(key+'_predictions.json'),records);write(WORK/(key+'_memory.json'),fixed)
   results[key]={d:dict(F1=sum(x['F1'] for x in records if x['domain']==d)/20,NLL=sum(x['NLL'] for x in records if x['domain']==d)/20) for d in DOMAINS}
   fixed_memory[key]=fixed;del model;clear()
   assert torch.cuda.memory_allocated()<1024**2,'Unexpected retained graph after inference release'
 write(REPORT/'COMPLETE.json',dict(complete=True,identity=who,ready_sha256=sha(REPORT/'READY.json'),seconds=time.monotonic()-start,selection=selected,results=results,packages=packages,fixed_memory=fixed_memory,new_generations=new_generations,locator_backwards=38,training_updates=0,TEST_QA=0,not_framework_success=True,not_historical_S0_bitwise=True,files={p.name:sha(p) for p in WORK.glob('*.json')}))
 signal.alarm(0)
if __name__=='__main__':
 try:main(int(sys.argv[1]))
 except Exception:
  write(REPORT/'FAILURE.json',dict(traceback=traceback.format_exc(),no_retry=True));raise
