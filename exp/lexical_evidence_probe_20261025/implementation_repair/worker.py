"""One frozen graph, twenty paired lexical-evidence interventions, no adaptation."""
from pathlib import Path
import hashlib,importlib.util,json,os,signal,sys,time,traceback
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN']);ASSET=ROOT/'.deployment/paired_image_support_20261024_view'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(n,v):
 with (RUN/n).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main(case):
 ready=read(RUN/'READY.json')
 for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
 for n,h in ready['prior_files'].items():assert sha(RUN/n)==h,n
 prior=read(RUN/'prior_READY.json')
 for n,h in prior['runtime_files'].items():assert sha(ASSET/n)==h,n
 assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==prior['storage_helper_sha256']
 spec=importlib.util.spec_from_file_location('identity_only',ASSET/'exp/paired_image_support_20261024/v1/common.py');ids=importlib.util.module_from_spec(spec);spec.loader.exec_module(ids)
 card=ids.inventory()[1];assert not card['owners'] and card['memory_MiB']<500 and card['utilization']<10
 tick=time.monotonic();who=ids.identity(os.getpid());assert who['pid']==who['pgid']
 write(case+'_STARTED.json',dict(unix=time.time(),identity=who,GPU=1,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),ready_sha256=sha(RUN/'READY.json')))
 def timeout(*args):raise TimeoutError('Fixed phase900s budget')
 remaining=(ready['original_parent_deadline_unix']-time.time()) if case=='parent' else 900
 assert remaining>0
 signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(remaining)))
 os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
 import torch
 from transformers import AutoProcessor,LlavaOnevisionConfig,LlavaOnevisionForConditionalGeneration
 import evidence,vision_interface as vi
 sys.path.insert(0,str(ASSET/'exp/two_backbone_native_cl_20261017_v2'));sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
 import onevision_bridge as bridge,runtime_utils as rt,frozen_vision as fv,study
 spec=importlib.util.spec_from_file_location('static_storage',ROOT/'exp/ffn_static_delete_20261025/v1/core.py');storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
 torch.set_num_threads(8);torch.manual_seed(42);fv.LIMIT=0
 base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV';processor=AutoProcessor.from_pretrained(base,local_files_only=True)
 manifest_path,manifest,original=study.original_train();split,fit,dev=study.split_train(manifest,original)
 assert study.sha(manifest_path)==prior['manifest_sha256'] and study.sha(study.REPORT/'SPLIT.json')==prior['split_sha256'] and split==read(study.REPORT/'SPLIT.json')
 assert sha(RUN/'official_ocr_projection.json')==ready['projection_sha256']
 ocr={r['id']:r for r in read(RUN/'official_ocr_projection.json')};wanted=set(ocr)
 group_by_id={rid:gid for gid,g in split['groups'].items() for rid in g['ids'] if rid in wanted}
 assert len(group_by_id)==20
 donors=evidence.pair_sources(list(ocr.values()),group_by_id);bodies={rid:evidence.body_ids(processor.tokenizer,row) for rid,row in ocr.items()}
 write(case+'_INTERVENTION_PLAN.json',dict(donors=donors,bodies=bodies,body_token_counts={rid:len(x) for rid,x in bodies.items()},groups=group_by_id))
 # Same already-verified independent strict loading route; no teacher weights in candidate.
 if case=='parent':
  cp=RUN/'wide_parent_static.pt';assert cp.stat().st_size==prior['parent']['bytes'];p=torch.load(cp,map_location='cpu',weights_only=False)
  model=bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True)
  model.core.load_state_dict(p['state'],strict=True);expected=prior['parent']['digest']
 else:
  proof=read(RUN/'prior_fit_COMPLETE.json');cp=RUN/'candidate_complete.pt';assert cp.stat().st_size==proof['checkpoint']['bytes'];p=torch.load(cp,map_location='cpu',weights_only=False)
  cfg=LlavaOnevisionConfig(**p['base_config']);cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
  core=LlavaOnevisionForConditionalGeneration(cfg);model=bridge.OneVisionBridge(core);interface=vi.VisionInterface(vi.student(p['student_config']));vi.install(model,interface)
  with torch.no_grad():
   for parameter in model.core.parameters():parameter.data=parameter.data.to(dtype=torch.float16)
  model.core.load_state_dict(p['state'],strict=True);expected=proof['digest']
 model.cuda().requires_grad_(False);model.eval();assert storage.digest(model)==expected==p['digest'] and storage.layout(model)==p['layout']
 assert model.visual.tower is model.core.vision_tower and model.visual.projector is model.core.multi_modal_projector
 assert storage._measure(model.named_parameters(remove_duplicate=False))==storage.layout(model)['weights']
 assert model.core.language_model.config._attn_implementation=='sdpa'
 assert sorted({type(layer.self_attn).__name__ for layer in model.model.layers})==read(RUN/('prior_'+case+'_COMPLETE.json'))['language_attention_classes']
 buffer_digest=lambda:storage.digest({n:v for n,v in model.core.named_buffers() if not n.startswith(('vision_tower.','multi_modal_projector.'))})
 buffers_before=buffer_digest();assert buffers_before==read(RUN/('prior_'+case+'_COMPLETE.json'))['preserved_nonvision_buffers_digest']
 bridge.bind(model);fv.install(model);before=storage.digest(model);del p
 previous={r['id']:r for r in read(RUN/('prior_'+case+'_predictions.json')) if r['id'] in wanted};assert len(previous)==20
 rows=[(t,row) for t in manifest['tasks'] for row in dev[t['task']] if row['id'] in wanted];assert len(rows)==20 and all(t['domain']=='natural' for t,row in rows)
 captures={};visual_requests=0
 def capture(module,args):
  nonlocal visual_requests
  assert not captures;visual_requests+=1;captures.update(raw_pixels_sha256=storage.digest({'pixels':args[0]}),tiles=args[0].shape[0])
 handle=model.core.vision_tower.register_forward_pre_hook(capture)
 class FiniteScores:
  def __call__(self,input_ids,scores):
   assert not torch.isnan(scores).any() and not torch.isposinf(scores).any() and torch.isfinite(scores).any(dim=-1).all();return scores
 records=[]
 try:
  for task,row in rows:
   rid=row['id'];assert str(row['metadata']['original_image_id'])==str(ocr[rid]['image_id'])
   for name in row['images']:assert sha(ASSET/name)==split['image_sha256'][name]
   captures.clear();batch,details=rt.input_batch(row,task,processor,model.device);assert captures and model._llmcl_visual_cache_bytes==0
   old=previous[rid];prepared=storage.digest(batch);assert prepared==old['prepared_sha256']
   assert captures['raw_pixels_sha256']==old['raw_pixels_sha256'] and len(batch['pixel_values'])==old['packed_tokens']
   assert storage.digest({k:v for k,v in batch.items() if k!='pixel_values'})==old['transport_sha256']
   assert row['answers']==old['references'] and task['max_new_tokens']==old['generation_cap']
   original_ids=batch['input_ids'][0].tolist();assert bool((batch['attention_mask']==1).all())
   donor=donors[rid];item=evidence.intervention(original_ids,bodies[rid],bodies[donor] if donor else [],processor.tokenizer)
   assert item['correct'].count(model.config.image_token_id)==original_ids.count(model.config.image_token_id)==len(batch['pixel_values'])
   image_digest=storage.digest({'features':batch['pixel_values']});outer_digest=storage.digest({k:v for k,v in batch.items() if k not in ['input_ids','attention_mask']})
   order=['correct','wrong'] if int(hashlib.sha256(rid.encode()).hexdigest(),16)%2==0 else ['wrong','correct']
   for condition in order:
    altered=dict(batch,input_ids=torch.tensor([item[condition]],dtype=batch['input_ids'].dtype,device=model.device))
    altered['attention_mask']=torch.ones_like(altered['input_ids'])
    with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
     loss,n=rt.answer_loss(model,altered,row['answer'],processor);assert torch.isfinite(loss)
     output=model.generate(**altered,do_sample=False,num_beams=1,max_new_tokens=task['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[FiniteScores()])
    tokens=output[0,altered['input_ids'].shape[1]:].tolist()
    records.append(dict(id=rid,condition=condition,donor=donor,original_prepared_sha256=prepared,raw_pixels_sha256=captures['raw_pixels_sha256'],image_features_sha256=image_digest,nontext_sha256=outer_digest,original_input_ids=original_ids,input_ids=item[condition],body_start=item['start'],body_stop=item['stop'],packed_tokens=len(batch['pixel_values']),target_tokens=n,NLL=float(loss),prediction=processor.tokenizer.decode(tokens,skip_special_tokens=True),output_tokens=tokens,references=row['answers'],generation_cap=task['max_new_tokens']))
    assert storage.digest({'features':batch['pixel_values']})==image_digest
    del altered,loss,output
   assert len(records)%2==0
   with (RUN/(case+'_events.jsonl')).open('a') as f:f.write(json.dumps(dict(completed_QA=len(records)//2))+'\n');f.flush()
   del batch
 finally:handle.remove()
 assert len(records)==40 and visual_requests==20 and storage.digest(model)==before and buffer_digest()==buffers_before
 assert all(p.grad is None and not p.requires_grad for p in model.parameters())
 write(case+'_predictions.json',records)
 import resource
 write(case+'_COMPLETE.json',dict(identity=who,seconds=(time.time()-ready['original_parent_started_unix']) if case=='parent' else time.monotonic()-tick,attempt_seconds=time.monotonic()-tick,completed_unix=time.time(),ready_sha256=sha(RUN/'READY.json'),digest=before,layout=storage.layout(model),preserved_nonvision_buffers_digest=buffers_before,visual_requests=visual_requests,NLL_requests=40,new_generations=40,backwards=0,optimizer_updates=0,process_RSS_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,whole_phase_GPU_peak_allocated=torch.cuda.max_memory_allocated(),files={n:sha(RUN/n) for n in [case+'_predictions.json',case+'_INTERVENTION_PLAN.json']}))
if __name__=='__main__':
 case=sys.argv[1];assert case in ['parent','candidate']
 try:main(case)
 except Exception:write(case+'_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc()));raise
