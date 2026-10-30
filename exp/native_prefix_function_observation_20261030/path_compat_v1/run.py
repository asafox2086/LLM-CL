"""Original-checkpoint native-prefix functional observation; one model per process.
Default is metadata-only. Original asset scope is reused as a pure validator, never main.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,time,traceback
SCHEMA='native-prefix-function-observation-v1'
OLD_BINDING_SHA='fa32a3e584e5c8aaf030931928873be4a3f2761d744bd8656c40f00c88eeea90'
OLD_SCOPE_SHA='1f3eb8ef68d6ac9f3c9a9961cc1b2f2f9964ba16370b0a79e5020c4e6bddc6db'
READOUT_SHA='55e95fd9dde9a76b34aa02bdd12b507e03506feb2137a706ba31067c37da6ef1'
REF_SHA='11ee3cff9c09534d67f35a09662b1131426e55d570cc92ab5ca3b42012b8673c'
METHODS=('PASs_MoE','CL_MoE')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,j):
 with Path(p).open('x') as f:json.dump(j,f,indent=2,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
 fd=os.open(Path(p).parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def scope(b,method):
 assert b['schema']==SCHEMA and tuple(b['methods'])==METHODS
 assert b['condition']=='pct01' and b['backbone']=='onevision' and b['prediction_prefix_limit']==32
 assert b['source_root']=='/data2/liyapeng_grp/program/LLMCL'
 here=Path(__file__).resolve().parent
 assert here==Path(b['observation_source_dir']).resolve()
 assert set(b['scientific_sources'])=={'run.py','observe_qa.py','native_hooks.py','gpu_statistics.py'}
 for n,h in b['scientific_sources'].items():assert sha(here/n)==h,n
 assert sha(b['protocol_path'])==b['protocol_sha256']
 assert sha(b['original_binding_path'])==OLD_BINDING_SHA and sha(b['original_scope_path'])==OLD_SCOPE_SHA
 old=read(b['original_binding_path']);original=load('native_prefix_original_asset_validator',b['original_scope_path'])
 # This function only reads original source/config/TRAIN/image metadata; no old CLI or model.
 root,port,run,cfg,rows,ids_sha=original.scope(old,method)
 assert b['calibration_ids_sha256']==ids_sha and len(rows)==80
 assert b['native_source_dir']==str(port)
 assert sha(b['old_tf_c_path'])==REF_SHA==b['old_tf_c_sha256']
 ref=read(b['old_tf_c_path']);assert ref['source_READOUT_sha256']==READOUT_SHA
 assert set(ref['methods'])==set(METHODS) and len(ref['methods'][method])==168
 assert sha(b['prefix_proof_path'])==b['prefix_proof_sha256']
 for name,h in b['prefix_runtime_files'].items():assert sha(name)==h,name
 assert b['limit_seconds']==7200 and b['cuda_visible_devices']=='1' and b['gpu_uuid']=='GPU-3c72e4ce-0896-0b40-d610-be585d2bb51e'
 assert b['reference_policy']=='old_TF_global_and_native_pooled_global_no_phase_refit'
 assert b['output_root'].startswith(str(here)+'/runs/') and Path(b['admission_path']).is_absolute()
 return old,original,root,port,run,cfg,rows,ids_sha,ref['methods'][method]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--method',choices=METHODS,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 b=read(a.binding);old,original,root,port,run,cfg,rows,ids_sha,c_ref=scope(b,a.method)
 out=Path(b['output_root'])/a.method
 if not a.execute:
  print(json.dumps(dict(metadata_validated=True,original_model_loads=0,new_generations=0,TRAIN_queries=80,TEST=0,output_exists=out.exists())));return
 assert b['admitted'] is True and os.getuid()==1084
 adm=read(b['admission_path']);assert adm['approved'] is True and adm['binding_sha256']==sha(a.binding)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
 uuid=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==b['gpu_uuid']
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 stat=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split();who=dict(pid=os.getpid(),pgid=os.getpgrp(),uid=os.getuid(),startticks=stat[19]);assert who['pid']==who['pgid']
 write(out/'STARTED.json',dict(schema=SCHEMA,identity=who,method=a.method,binding_sha256=sha(a.binding),limit_seconds=7200,no_automatic_retry=True))
 def expire(*unused):raise TimeoutError('native prefix observation deadline7200')
 signal.signal(signal.SIGALRM,expire);signal.alarm(7200)
 try:
  import torch
  torch.set_num_threads(8);torch.cuda.set_device(0);assert torch.cuda.device_count()==1
  sys.path.insert(0,str(port))
  import checkpoints,runtime_utils as rt,onevision_bridge as bridge,frozen_vision as fv
  for m in (checkpoints,rt,bridge,fv):assert sha(m.__file__)==cfg['execution_source_sha256'][Path(m.__file__).name]
  assert rt.ROOT.resolve()==root.resolve()
  storage=load('native_prefix_storage',old['storage_helper'])
  hooks=load('native_prefix_hooks',Path(__file__).parent/'native_hooks.py')
  stats=load('native_prefix_gpu_statistics',Path(__file__).parent/'gpu_statistics.py')
  observer=load('native_prefix_observe_qa',Path(__file__).parent/'observe_qa.py')
  import transformers.generation.utils as gu,transformers.generation.stopping_criteria as gs
  canonical_runtime={str(Path(p).resolve()):h for p,h in b['prefix_runtime_files'].items()}
  assert len(canonical_runtime)==len(b['prefix_runtime_files']),'Duplicate canonical runtime bindings'
  for mod in (gu,gs):assert sha(mod.__file__)==canonical_runtime[str(Path(mod.__file__).resolve())]
  cp=run/'checkpoints/stage_19.pt';assert sha(cp)==original.METHODS[a.method][0]
  payload=checkpoints.load(run,expected=cfg)
  assert payload['format']=='llmcl-method-graph-v1' and payload['stage']==19 and payload['probe'] is None
  method=payload['method'];model=method.model.eval();processor=payload['processor'];del payload
  assert method.name==a.method;model.requires_grad_(False);bridge.bind(model);bridge.NATIVE_GATE_MODE=False
  model._llmcl_visual_cache.clear();model._llmcl_visual_cache_bytes=0;fv.LIMIT=0
  before=storage.digest(model.core);buffers=storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
  gen_before=model.core.language_model.generation_config.to_dict()
  counts=dict(original_checkpoint_loads=1,TRAIN_queries=0,generations=0,native_prediction_forwards=0,question_feature_decoder_forwards=0,feature_cache_hits=0,TEST_queries=0,CE=0,F1=0,backwards=0,optimizer_updates=0,new_trained_models=0,new_checkpoints=0)
  manifests=[]
  write(out/'MODEL_LOADED.json',dict(method=a.method,original_checkpoint_sha256=original.METHODS[a.method][0],original_core_digest=before,calibration_ids_sha256=ids_sha,gpu_uuid=uuid,neural_layout=storage.layout(model.core)))
  with torch.inference_mode():
   for index,(task,row) in enumerate(rows):
    qa=out/('QA_%03d'%index)
    meta=observer.observe_qa(method=method,processor=processor,task=task,row=row,rt=rt,fv=fv,hooks_module=hooks,statistics=stats,old_tf_c=c_ref,qa_dir=qa)
    assert meta['new_F1']==meta['new_CE']==meta['new_scores']==meta['new_checkpoints']==0
    counts['TRAIN_queries']+=1;counts['generations']+=1
    counts['native_prediction_forwards']+=meta['generation_decoder_forwards']
    counts['question_feature_decoder_forwards']+=meta['question_feature_decoder_forwards']
    counts['feature_cache_hits']+=int(bool(meta['question_feature_cache_hit']))
    manifests.append(dict(index=index,task=task['task'],domain=task['domain'],qa_id_sha256=meta['qa_id_sha256'],prefix_positions=meta['actual_prediction_positions'],decode_eligible=meta['decode_eligible'],QA_COMPLETE_sha256=sha(qa/'COMPLETE.json'),moments_sha256=meta['moments_sha256'],position_records_sha256=meta['position_records_sha256']))
    print(json.dumps(dict(event='QA_COMPLETE',index=index,positions=meta['actual_prediction_positions'],method=a.method)),flush=True)
  assert counts['TRAIN_queries']==counts['generations']==80
  assert storage.digest(model.core)==before and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==buffers
  assert model.core.language_model.generation_config.to_dict()==gen_before
  write(out/'RESULT.json',dict(schema=SCHEMA,method=a.method,binding_sha256=sha(a.binding),counts=counts,QA=manifests,original_core_digest_before_and_after=before,weights_and_buffers_unchanged=True,generation_config_unchanged=True,old_TF_c_sha256=REF_SHA,no_reference_injected_into_model=True,no_mean_fit_in_this_worker=True,not_F1_or_deployment_resource_verification=True,native_generation_with_readonly_hooks=True))
  write(out/'COMPLETE.json',dict(schema=SCHEMA,identity=who,binding_sha256=sha(a.binding),RESULT_sha256=sha(out/'RESULT.json'),seconds=time.monotonic()-start));signal.alarm(0)
 except BaseException:
  write(out/'FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time(),no_automatic_retry=True));raise
if __name__=='__main__':main()
