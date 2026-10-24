"""Same frozen evaluation in one fresh process per physical graph."""
from pathlib import Path
import gc,hashlib,importlib.util,json,os,signal,sys,time,traceback
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN'])
ASSET=ROOT/'.deployment/paired_image_support_20261024_view';PORT=ASSET/'exp/two_backbone_native_cl_20261017_v2'
DOMAINS=('natural','medical','geography','finance','traffic')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main(name):
 start=time.monotonic();repair=json.loads((RUN/'EVAL_REPAIR.json').read_bytes());assert time.time()<repair['absolute_deadline_unix']
 assert sha(Path(__file__))==repair['evaluation_source_sha256']
 signal.alarm(max(1,int(repair['absolute_deadline_unix']-time.time())))
 os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
 import torch
 from transformers import AutoProcessor
 sys.path.insert(0,str(PORT));sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
 import onevision_bridge as bridge,runtime_utils as rt,frozen_vision as fv,study
 spec=importlib.util.spec_from_file_location('static_storage',ROOT/'exp/ffn_static_delete_20261025/v1/core.py');storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
 torch.set_num_threads(8);torch.manual_seed(42);fv.LIMIT=0
 pid=os.getpid();stat=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
 identity=dict(pid=pid,startticks=stat[19],uid=os.getuid(),pgid=int(stat[2]));assert identity['pid']==identity['pgid']
 write(RUN/(name+'_EVAL_STARTED.json'),dict(unix=time.time(),identity=identity))
 path=RUN/repair['packages'][name]['path'];assert sha(path)==repair['packages'][name]['sha256']
 p=torch.load(path,map_location='cpu',weights_only=False);base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
 processor=AutoProcessor.from_pretrained(base,local_files_only=True)
 model=bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
 model.requires_grad_(False);model.eval();bridge.bind(model);fv.install(model)
 if p['indexes']:storage.delete(model,p['indexes'])
 model.core.load_state_dict(p['state'],strict=True)
 assert storage.digest(model)==p['digest'] and storage.layout(model)==p['layout']
 assert p['indexes']==([] if name=='parent' else repair['selected'])
 for i in p['indexes']:assert not any(f'model.layers.{i}.mlp.' in n for n in model.core.state_dict())
 del p;gc.collect();torch.cuda.empty_cache()
 before=storage.digest(model)
 _,manifest,original=study.original_train();split,fit,dev=study.split_train(manifest,original)
 ready=json.loads((RUN/'READY.json').read_bytes());assert sha(study.REPORT/'SPLIT.json')==ready['split_sha256']
 devrows=[(t,r) for t in manifest['tasks'] for r in dev[t['task']]];assert len(devrows)==100
 def inputs(t,r):
  for n in r['images']:assert sha(ASSET/n)==split['image_sha256'][n]
  batch,_=rt.input_batch(r,t,processor,model.device);assert model._llmcl_visual_cache_bytes==0;return batch
 class FiniteScores:
  def __call__(self,input_ids,scores):
   assert not torch.isnan(scores).any() and not torch.isposinf(scores).any() and torch.isfinite(scores).any(dim=-1).all();return scores
 records=[];measurements=[]
 for t,row in devrows:
  batch=inputs(t,row)
  with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
   loss,n=rt.answer_loss(model,batch,row['answer'],processor);assert torch.isfinite(loss)
   out=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=t['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[FiniteScores()])
  tokens=out[0,batch['input_ids'].shape[1]:].tolist()
  records.append(dict(id=row['id'],task=t['task'],domain=t['domain'],input_sha256=storage.digest(batch),NLL=float(loss),target_tokens=n,output_tokens=tokens,prediction=processor.tokenizer.decode(tokens,skip_special_tokens=True),references=row['answers'],generation_cap=t['max_new_tokens']))
  with (RUN/(name+'_eval_events.jsonl')).open('a') as f:f.write(json.dumps(dict(completed_DEV=len(records)))+'\n')
  del batch,out,loss
 write(RUN/(name+'_predictions.json'),records)
 for d in DOMAINS:
  t,row=min(((t,row) for t,row in devrows if t['domain']==d),key=lambda tr:hashlib.sha256(tr[1]['id'].encode()).digest())
  bridge.LAST_NATIVE=None;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
  with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
   batch=inputs(t,row)
   out=model.generate(**batch,do_sample=False,num_beams=1,min_new_tokens=32,max_new_tokens=32,use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[FiniteScores()])
  torch.cuda.synchronize();assert out.shape[1]-batch['input_ids'].shape[1]==32
  measurements.append(dict(id=row['id'],domain=d,input_sha256=storage.digest(batch),prompt_tokens=batch['input_ids'].shape[1],new_tokens=32,baseline_allocated=baseline,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
  del batch,out
 write(RUN/(name+'_memory.json'),measurements)
 assert storage.digest(model)==before and not any(p.grad is not None for p in model.parameters())
 write(RUN/(name+'_EVAL_COMPLETE.json'),dict(identity=identity,seconds=time.monotonic()-start,NLL_requests=100,new_generations=105,layout=storage.layout(model),digest=before,files={n:sha(RUN/n) for n in [name+'_predictions.json',name+'_memory.json']},backwards=0,training_updates=0))
if __name__=='__main__':
 name=sys.argv[1];assert name in ('parent','pruned')
 try:main(name)
 except Exception:write(RUN/(name+'_EVAL_FAILURE.json'),dict(traceback=traceback.format_exc()));raise
