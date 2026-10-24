"""Read fixed domain-end snapshots; no optimization, teacher or candidate."""
from pathlib import Path
import hashlib,importlib.util,json,os,signal,sys,time,traceback
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN']);ASSET=Path(os.environ['LLMCL_ASSET'])
CASES={'control19':(19,None),'natural1':(1,'natural'),'medical11':(11,'medical'),'geography15':(15,'geography'),'finance16':(16,'finance')}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
 with (RUN/n).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main(case):
 ready=read(RUN/'READY.json');stage,domain=CASES[case];gpu=int(os.environ['LLMCL_GPU'])
 for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
 for n,h in ready['runtime_files'].items():assert sha(ASSET/n)==h,n
 for n,h in ready['prior_files'].items():assert sha(RUN/n)==h,n
 spec=importlib.util.spec_from_file_location('identity_only',ASSET/'exp/paired_image_support_20261024/v1/common.py');ids=importlib.util.module_from_spec(spec);spec.loader.exec_module(ids)
 card=ids.inventory()[gpu];assert not card['owners'] and card['memory_MiB']<500 and card['utilization']<10
 who=ids.identity(os.getpid());assert who['pid']==who['pgid'];tick=time.monotonic()
 write(case+'_STARTED.json',dict(identity=who,unix=time.time(),GPU=gpu,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),ready_sha256=sha(RUN/'READY.json'),activation_sha256=sha(RUN/'ACTIVATION.json')))
 def expired(*args):raise TimeoutError('Frozen 180 second readout phase expired')
 signal.signal(signal.SIGALRM,expired);signal.alarm(180)
 os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
 import random,numpy as np,torch
 from transformers import AutoProcessor
 sys.path.insert(0,str(ASSET/'exp/two_backbone_native_cl_20261017_v2'));sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
 import onevision_bridge as bridge,frozen_vision as fv,adaptation,vision_interface as vi
 from snapshot_restore import restore_snapshot
 spec=importlib.util.spec_from_file_location('storage_only',ROOT/'exp/ffn_static_delete_20261025/v1/core.py');storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
 assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
 torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 init=read(RUN/'cf_INITIALIZATION.json');train=read(RUN/'cf_train_COMPLETE.json');record=read(RUN/f'STAGE_{stage:02}_COMPLETE.json');assert record['stage']==stage
 base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
 model=bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda();bridge.bind(model)
 assert storage.digest(model.core.language_model)==init['base_language_digest']
 assert sha(RUN/'fixed_vision_only.pt')==ready['vision_asset']['sha256'];asset=torch.load(RUN/'fixed_vision_only.pt',map_location='cpu',weights_only=False)
 interface=vi.VisionInterface(vi.student(asset['student_config']),**asset['interface_config']).half().cuda();interface.load_state_dict(asset['state'],strict=True)
 assert storage.digest(interface)==asset['digest']==init['vision_digest'];del asset
 torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 method=adaptation.make_budget(model,ASSET,RUN/'inherited_budget_core.py')
 adaptation.install_interface_preserving_adapters(model,interface,method)
 snapshot_path=RUN/f'adapter_S{stage:02}.pt';assert sha(snapshot_path)==ready['snapshot_files'][snapshot_path.name]['sha256'];snapshot=torch.load(snapshot_path,map_location='cpu',weights_only=True)
 receipt=restore_snapshot(model,method,snapshot,record,init,storage);del snapshot
 fold=adaptation.fold(model,method);model.eval()
 assert all(p.grad is None and not p.requires_grad for p in model.parameters())
 layout=storage.layout(model);assert layout==train['layout']
 assert model.visual.tower is model.core.vision_tower and model.visual.projector is model.core.multi_modal_projector
 assert sorted({type(l.self_attn).__name__ for l in model.model.layers})==read(RUN/'teacher_parent_COMPLETE.json')['language_attention_classes']
 assert model.core.language_model.config._attn_implementation=='sdpa' and interface.student.config._attn_implementation=='eager'
 buffer_digest=storage.digest({n:v for n,v in model.core.named_buffers() if not n.startswith(('vision_tower.','multi_modal_projector.'))})
 assert buffer_digest==init['buffer_digest']
 before=storage.digest(model);assert storage.digest(interface)==init['vision_digest']
 predictions=[]
 if domain is None:
  assert before==train['digest']==read(RUN/'cf_compact_eval_COMPLETE.json')['digest']
  assert record['stage']==19
 else:
  control=read(RUN/'control19_COMPLETE.json');assert control['digest_before']==control['digest_after']==train['digest'] and control['new_generations']==0
  fv.LIMIT=0;fv.install(model)
  from runtime_utils import input_batch
  processor=AutoProcessor.from_pretrained(base,local_files_only=True)
  policy=read(ASSET/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json');manifest_path=ASSET/policy['conditions']['pct01']['manifest'];assert sha(manifest_path)==ready['manifest_sha256'];manifest=read(manifest_path)
  split_path=ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json';assert sha(split_path)==ready['split_sha256'];split=read(split_path)
  bounds={d:max(t['task'] for t in manifest['tasks'] if t['domain']==d) for d in dict.fromkeys(t['domain'] for t in manifest['tasks'])}
  assert bounds=={'natural':1,'medical':11,'geography':15,'finance':16,'traffic':19} and stage==bounds[domain]
  final={p['id']:p for p in read(RUN/'cf_final_predictions.json')};rows=[]
  for t in manifest['tasks']:
   if t['domain']!=domain:continue
   source=ASSET/t['files']['train']['path'];assert sha(source)==t['files']['train']['sha256']
   wanted=set(split['dev_ids'][str(t['task'])]);rows.extend((t,r) for r in [json.loads(x) for x in source.read_bytes().splitlines()] if r['id'] in wanted)
  assert len(rows)==20 and {r['id'] for t,r in rows}=={rid for rid,p in final.items() if p['domain']==domain}
  class FiniteScores:
   def __call__(self,input_ids,scores):
    assert not torch.isnan(scores).any() and not torch.isposinf(scores).any() and torch.isfinite(scores).any(dim=-1).all();return scores
  for t,row in rows:
   for name in row['images']:assert sha(ASSET/name)==split['image_sha256'][name]
   batch,_=input_batch(row,t,processor,model.device);old=final[row['id']]
   prepared=storage.digest(batch);transport=storage.digest({k:v for k,v in batch.items() if k!='pixel_values'})
   assert prepared==old['prepared_sha256'] and transport==old['transport_sha256'] and len(batch['pixel_values'])==old['packed_tokens']
   assert row['answers']==old['references'] and t['max_new_tokens']==old['generation_cap']
   with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
    out=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=t['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[FiniteScores()])
   tokens=out[0,batch['input_ids'].shape[1]:].tolist()
   predictions.append(dict(id=row['id'],task=t['task'],domain=domain,stage=stage,prepared_sha256=prepared,transport_sha256=transport,packed_tokens=len(batch['pixel_values']),prediction=processor.tokenizer.decode(tokens,skip_special_tokens=True),output_tokens=tokens,references=row['answers'],generation_cap=t['max_new_tokens']))
   del batch,out
  assert model._llmcl_visual_cache_bytes==0
 after=storage.digest(model);assert before==after and buffer_digest==init['buffer_digest'] and storage.digest(interface)==init['vision_digest']
 assert all(p.grad is None and not p.requires_grad for p in model.parameters())
 name=case+'_predictions.json';write(name,predictions)
 proof=dict(identity=who,stage=stage,domain=domain,reconstruction=receipt,fold=fold,digest_before=before,digest_after=after,layout=layout,vision_digest=storage.digest(interface),buffer_digest=buffer_digest,new_generations=len(predictions),backwards=0,optimizer_updates=0,teacher_requests=0,FIT_model_requests=0,NLL_requests=0,TEST_QA=0,files={name:sha(RUN/name)},ready_sha256=sha(RUN/'READY.json'))
 proof['seconds']=time.monotonic()-tick;write(case+'_COMPLETE.json',proof)
if __name__=='__main__':
 case=sys.argv[1];assert case in CASES
 try:main(case)
 except Exception:write(case+'_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc()));raise
