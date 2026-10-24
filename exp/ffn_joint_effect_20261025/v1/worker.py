"""37 fixed-input NLLs; existing failed graph, no selection or optimization."""
from pathlib import Path
import gc,hashlib,importlib.util,json,os,signal,sys,time,traceback
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN'])
ASSET=ROOT/'.deployment/paired_image_support_20261024_view'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 ready=json.loads((RUN/'READY.json').read_text())
 assert sha(__file__)==ready['worker_sha256'] and sha(RUN/'PROTOCOL.md')==ready['protocol_sha256']
 for n,h in ready['source_files'].items():assert sha(Path(ready['source_run'])/n)==h,n
 for n,h in ready['runtime_files'].items():assert sha(ASSET/n)==h,n
 assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
 sys.path.insert(0,str(ASSET/'exp/paired_image_support_20261024/v1'))
 spec=importlib.util.spec_from_file_location('identity_only',ASSET/'exp/paired_image_support_20261024/v1/common.py');ids=importlib.util.module_from_spec(spec);spec.loader.exec_module(ids)
 cards=ids.inventory();assert not cards[1]['owners'] and cards[1]['memory_MiB']<500 and cards[1]['utilization']<10
 assert not (RUN/'STARTED.json').exists()
 start=time.monotonic();who=ids.identity(os.getpid());assert who['pid']==who['pgid']
 write(RUN/'STARTED.json',dict(unix=time.time(),identity=who,GPU=1,ready_sha256=sha(RUN/'READY.json')))
 def timeout(*args):raise TimeoutError('600 second whole diagnostic budget')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
 os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
 import torch
 from transformers import AutoProcessor
 sys.path.insert(0,str(ASSET/'exp/two_backbone_native_cl_20261017_v2'));sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
 import onevision_bridge as bridge,runtime_utils as rt,frozen_vision as fv,study
 spec=importlib.util.spec_from_file_location('static_storage',ROOT/'exp/ffn_static_delete_20261025/v1/core.py');storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
 torch.set_num_threads(8);torch.manual_seed(42);fv.LIMIT=0
 source=Path(ready['source_run']);selection=json.loads((source/'selection.json').read_text());oldready=json.loads((source/'READY.json').read_text())
 assert selection['selected']==ready['selected']
 cp=source/'pruned_complete.pt';assert cp.stat().st_size==ready['checkpoint']['bytes'] and sha(cp)==ready['checkpoint']['sha256']
 pack=torch.load(cp,map_location='cpu',weights_only=False);base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
 assert pack['indexes']==selection['selected']
 processor=AutoProcessor.from_pretrained(base,local_files_only=True)
 model=bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
 model.requires_grad_(False);model.eval();bridge.bind(model);fv.install(model);storage.delete(model,pack['indexes']);model.core.load_state_dict(pack['state'],strict=True)
 before=storage.digest(model);layout=storage.layout(model);assert before==pack['digest'] and layout==pack['layout']
 for i in pack['indexes']:assert not any(f'model.layers.{i}.mlp.' in n for n in model.core.state_dict())
 del pack;gc.collect();torch.cuda.empty_cache()
 manifest_path,manifest,original=study.original_train();split,fit,dev=study.split_train(manifest,original)
 assert sha(manifest_path)==oldready['parent']['source_checkpoint']['manifest_sha256']
 assert sha(study.REPORT/'SPLIT.json')==oldready['split_sha256']
 assert split==json.loads((study.REPORT/'SPLIT.json').read_bytes())
 rows=[(t,r) for t in manifest['tasks'] for r in sorted(fit[t['task']],key=lambda r:hashlib.sha256(('FFN-delete-20261025:'+r['id']).encode()).digest())[:2]]
 assert len(rows)==37 and [r['id'] for t,r in rows]==[x['id'] for x in selection['matrix']]
 records=[]
 for (t,row),old in zip(rows,selection['matrix']):
  assert (t['task'],t['domain'])==(old['task'],old['domain'])
  for n in row['images']:assert sha(ASSET/n)==split['image_sha256'][n]
  batch,_=rt.input_batch(row,t,processor,model.device);assert model._llmcl_visual_cache_bytes==0
  input_digest=storage.digest(batch);assert input_digest==old['input_sha256']
  with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):loss,n=rt.answer_loss(model,batch,row['answer'],processor)
  assert torch.isfinite(loss) and n==old['target_tokens']
  records.append(dict(id=row['id'],task=t['task'],domain=t['domain'],input_sha256=input_digest,target_tokens=n,answer_sha256=hashlib.sha256(row['answer'].encode()).hexdigest(),joint_NLL=float(loss)))
  with (RUN/'events.jsonl').open('a') as f:f.write(json.dumps(dict(completed_NLL=len(records)))+'\n');f.flush()
  del batch,loss
 assert storage.digest(model)==before and storage.layout(model)==layout and not any(p.grad is not None for p in model.parameters())
 write(RUN/'joint_NLL.json',records)
 write(RUN/'COMPLETE.json',dict(identity=who,seconds=time.monotonic()-start,requests=37,new_generations=0,backwards=0,training_updates=0,digest=before,layout=layout,ready_sha256=sha(RUN/'READY.json'),joint_NLL_sha256=sha(RUN/'joint_NLL.json')))
if __name__=='__main__':
 try:main()
 except Exception:
  write(RUN/'FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc()));raise
