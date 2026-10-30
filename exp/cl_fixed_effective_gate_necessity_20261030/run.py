"""DRAFT one-CP final TEST intervention. No new training, folding, c fitting or scheduling.
Default metadata-only; execution needs independent binding/admission and exclusive output.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,signal,subprocess,sys,time,traceback
SCHEMA='cl-fixed-effective-gate-v1';CP_SHA='8550fdeb886a0e942ee9038ab0aeae7e5933b943094ee3af40c0816df592ffa4';CONFIG_SHA='125d1f9ead67d5e4d222f5fcfc05897b582f90bbe2bdbd1603e12534aa91dc02'
OLD_BINDING_SHA='fa32a3e584e5c8aaf030931928873be4a3f2761d744bd8656c40f00c88eeea90';OLD_SCOPE_SHA='1f3eb8ef68d6ac9f3c9a9961cc1b2f2f9964ba16370b0a79e5020c4e6bddc6db';REF_SHA='11ee3cff9c09534d67f35a09662b1131426e55d570cc92ab5ca3b42012b8673c';READOUT_SHA='55e95fd9dde9a76b34aa02bdd12b507e03506feb2137a706ba31067c37da6ef1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,j):
 with Path(p).open('x') as f:json.dump(j,f,indent=2,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def scope(b):
 assert b['schema']==SCHEMA and b['method']=='CL_MoE' and b['condition']=='pct01' and b['backbone']=='onevision'
 assert Path(__file__).resolve().parent==Path(b['source_dir']).resolve()
 assert set(b['scientific_sources'])=={'run.py','fixed_gate.py','controls.py','verify.py'}
 for n,h in b['scientific_sources'].items():assert sha(Path(b['source_dir'])/n)==h
 assert sha(b['protocol_path'])==b['protocol_sha256'] and sha(b['original_binding_path'])==OLD_BINDING_SHA and sha(b['original_scope_path'])==OLD_SCOPE_SHA
 old=read(b['original_binding_path']);orig=load('cl_fixed_original_asset_scope',b['original_scope_path']);root,port,run,cfg,rows,ids=orig.scope(old,'CL_MoE')
 assert orig.METHODS['CL_MoE'][0]==CP_SHA and orig.METHODS['CL_MoE'][1]==2100250526 and sha(run/'config.json')==CONFIG_SHA
 assert b['calibration_ids_sha256']==ids and b['control_indices']==[0,16,32,48,64]
 assert sha(b['old_TF_c_path'])==REF_SHA;ref=read(b['old_TF_c_path']);assert ref['source_READOUT_sha256']==READOUT_SHA
 c=ref['methods']['CL_MoE'];assert len(c)==168 and all(len(v)==8 and all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in v) for v in c.values())
 manifest=read(old['manifest']);assert len(manifest['tasks'])==19 and sum(t['files']['test']['count'] for t in manifest['tasks'])==21454
 assert b['reference_policy']=='frozen_55e_old_TF_effective_mean_no_renorm' and b['expert_scale']==2 and b['limit_seconds']==86400
 assert sha(b['baseline_complete_proof_path'])==b['baseline_complete_proof_sha256'];assert isinstance(read(b['baseline_complete_proof_path']),dict)
 assert sha(b['baseline_reference_path'])==b['baseline_reference_sha256']=='f31cb5e6f0ecbcb67abd3b2abbb0edc38423686cb3fe43ab784a1b5e031f148f'
 br=read(b['baseline_reference_path']);assert br['original_run']==str(run) and br['original_config_sha256']==CONFIG_SHA and br['checkpoint_payload_sha256']==CP_SHA
 assert br['checkpoint_latest_json_sha256']==sha(run/'checkpoints/latest.json') and br['manifest_sha256']==cfg['manifest_sha256'] and br['original_execution_source_sha256']==cfg['execution_source_sha256']
 assert br['original_formal_proof_sha256']==b['baseline_complete_proof_sha256'] and br['full_TEST_count']==21454 and len(br['final19'])==19
 assert sha(br['metric_lock_path'])==br['metric_lock_sha256'] and sha(root/'analysis/answer_f1.py')==br['answer_f1_source_sha256']
 for task,record in zip(manifest['tasks'],br['final19']):
  assert task['task']==record['task'] and task['max_new_tokens']==record['max_new_tokens'] and task['instruction']==record['instruction'] and task['files']['test']==record['TEST_descriptor']
  assert b['baseline_score_sha256'][str(task['task'])]==record['sha256']
 assert sha(b['baseline_prediction_hashes_path'])==b['baseline_prediction_hashes_sha256']=='72dcbe7581d04fecd4aa82826ce0e7b0a5b0e8de88a4d23c30e66edd3bf06b5b'
 rawref=read(b['baseline_prediction_hashes_path']);assert rawref['baseline_reference_sha256']==b['baseline_reference_sha256'] and rawref['checkpoint_payload_sha256']==CP_SHA and rawref['formal_proof_sha256']==b['baseline_complete_proof_sha256'] and rawref['original_run']==str(run)
 assert len(rawref['records'])==19
 for task,entry in zip(manifest['tasks'],rawref['records']):
  assert entry['task']==task['task'] and entry['TEST_count']==task['files']['test']['count']
  assert Path(entry['path'])==run/'evaluations/stage_19'/f"T{task['task']:02d}_test_predictions.jsonl"
 assert set(b['baseline_score_sha256'])=={str(i) for i in range(1,20)}
 baseline=[]
 for task in manifest['tasks']:
  i=task['task'];p=run/'evaluations'/f'stage_19'/f'T{i:02d}_test_predictions.scores.json';assert sha(p)==b['baseline_score_sha256'][str(i)]
  v=read(p);assert v['count']==task['files']['test']['count'] and 0<=v['token_f1']<=100
  baseline.append(dict(task=i,domain=task['domain'],count=v['count'],token_f1=v['token_f1'],path=str(p),sha256=sha(p)))
 assert b['cuda_visible_devices']=='1' and b['gpu_uuid']=='GPU-3c72e4ce-0896-0b40-d610-be585d2bb51e'
 assert Path(b['output_root']).is_relative_to(Path(b['source_dir'])/'runs')
 return old,orig,root,port,run,cfg,rows,manifest,c,baseline

def tests(root,manifest):
 data={};seen=set()
 for task in manifest['tasks']:
  f=task['files']['test'];p=root/f['path'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==f['sha256'];rows=[json.loads(x) for x in raw.splitlines()]
  assert len(rows)==f['count'];ids=[r['id'] for r in rows];assert len(set(ids))==len(ids) and not seen.intersection(ids);seen.update(ids);data[(task['task'],'test')]=rows
 assert len(seen)==21454;return data

def prior_tensors(method,layers):
 result=[('anchor_'+str(k),v) for k,v in method.anchors.items()]
 for n,m in layers.items():result += [(n+'.task_weight_'+str(k),v) for k,v in m.task_weights.items()]
 return result

def cache_state(method):
 m=method.model
 return dict(question_feature_entries=len(method.feature_cache),visual_entries=len(m._llmcl_visual_cache),visual_bytes=m._llmcl_visual_cache_bytes,
  selected_global_priors_are_mutable_runtime_state_not_parameters=True,control_warmup_and_full_TEST_preserve_original_cache_updates=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args();b=read(a.binding)
 old,orig,root,port,original_run,cfg,rows,manifest,c,baseline=scope(b);out=Path(b['output_root'])
 if not a.execute:print(json.dumps(dict(metadata_validated=True,model_loads=0,TEST_generation_calls=0,planned_TEST=21454,controls=5,output_exists=out.exists())));return
 assert b['admitted'] and os.getuid()==1084;ad=read(b['admission_path']);assert ad['approved'] and ad['binding_sha256']==sha(a.binding)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==b['gpu_uuid']
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stat=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split();identity=dict(pid=os.getpid(),pgid=os.getpgrp(),uid=os.getuid(),startticks=stat[19]);assert identity['pid']==identity['pgid']
 write(out/'STARTED.json',dict(schema=SCHEMA,binding_sha256=sha(a.binding),identity=identity,no_retry=True,limit_seconds=86400))
 def expire(*unused):raise TimeoutError('CL fixed-G full-test hard limit86400')
 signal.signal(signal.SIGALRM,expire);signal.alarm(86400);controller=None
 try:
  import torch
  torch.set_num_threads(8);torch.cuda.set_device(0);assert torch.cuda.device_count()==1;torch.cuda.reset_peak_memory_stats()
  sys.path.insert(0,str(port));import checkpoints,runtime_utils as rt,onevision_bridge as bridge,frozen_vision as fv
  for m in (checkpoints,rt,bridge,fv):assert sha(m.__file__)==cfg['execution_source_sha256'][Path(m.__file__).name]
  import transformers.generation.utils as gu,transformers.generation.stopping_criteria as gs
  canonical={str(Path(p).resolve()):h for p,h in b['prefix_runtime_files'].items()};assert len(canonical)==len(b['prefix_runtime_files'])
  for m in (gu,gs):assert sha(m.__file__)==canonical[str(Path(m.__file__).resolve())]
  assert sha(original_run/'checkpoints/stage_19.pt')==CP_SHA
  payload=checkpoints.load(original_run,expected=cfg);assert payload['format']=='llmcl-method-graph-v1' and payload['stage']==19 and payload['probe'] is None
  method=payload['method'];processor=payload['processor'];model=method.model.eval();del payload
  assert method.name=='CL_MoE';model.requires_grad_(False);bridge.bind(model);bridge.NATIVE_GATE_MODE=False
  storage=load('cl_fixed_storage',old['storage_helper']);gate=load('cl_fixed_gate',Path(__file__).parent/'fixed_gate.py');control=load('cl_fixed_controls',Path(__file__).parent/'controls.py')
  before=storage.digest(model.core);buffers=storage.digest(list(model.core.named_buffers(remove_duplicate=False)));generation=model.core.language_model.generation_config.to_dict()
  controller=gate.GateIntervention(method,c,port/'clmoe_adapter.py',cfg['execution_source_sha256']['clmoe_adapter.py'])
  priors=storage.digest(prior_tensors(method,controller.layers));cache_before=cache_state(method)
  write(out/'MODEL_LOADED.json',dict(payload_sha256=CP_SHA,config_sha256=CONFIG_SHA,original_parameter_digest=before,original_buffer_digest=buffers,neural_layout=storage.layout(model.core),new_trainable_parameters=0,original_bank_retained=True,cache_before=cache_before))
  with torch.inference_mode():controls=control.run_controls(controller=controller,method=method,processor=processor,rows=rows,rt=rt,fv=fv)
  assert len(controls)==5 and storage.digest(model.core)==before and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==buffers
  write(out/'CONTROLS_COMPLETE.json',dict(controls=controls,controller_report=controller.report(),new_quality_scores=0))
  data=tests(root,manifest);controller.install();controller.begin('fixed')
  with torch.inference_mode():rt.evaluate(method,processor,manifest['tasks'],data,out,19)
  report=controller.report();assert report['fixed_main_coverage_complete'] and report['counts']['fixed_main_calls']>0
  assert storage.digest(model.core)==before and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==buffers and storage.digest(prior_tensors(method,controller.layers))==priors
  assert model.core.language_model.generation_config.to_dict()==generation
  scores=[]
  for task in manifest['tasks']:
   p=out/'evaluations/stage_19'/f"T{task['task']:02d}_test_predictions.scores.json";v=read(p);assert v['count']==task['files']['test']['count']
   predictions=p.with_name(p.name.replace('.scores.json','.jsonl'));assert predictions.exists() and not predictions.with_suffix(predictions.suffix+'.partial').exists()
   scores.append(dict(task=task['task'],domain=task['domain'],count=v['count'],token_f1=v['token_f1'],score_path=str(p),score_sha256=sha(p),predictions_path=str(predictions),predictions_sha256=sha(predictions)))
  torch.cuda.synchronize()
  write(out/'RESULT.json',dict(schema=SCHEMA,binding_sha256=sha(a.binding),counts=dict(original_checkpoint_loads=1,TEST_generations=21454,control_queries=5,control_GENERATE_calls=10,backwards=0,optimizer_updates=0,new_checkpoints=0),
   fixed_effective_gate_report=report,scores=scores,baseline=baseline,original_parameters_and_buffers_unchanged=True,anchors_task_weights_unchanged=True,generation_config_unchanged=True,
   cache_before=cache_before,cache_after=cache_state(method),constant_tensor_bytes=report['constant_tensor_bytes'],no_bank_removed_or_folded=True,
   whole_worker_GPU_peak_allocated=torch.cuda.max_memory_allocated(),whole_worker_GPU_peak_reserved=torch.cuda.max_memory_reserved(),resource_scope='Load+dynamic_controls+full_TEST overhead; not an optimized deployment resource comparison'))
  write(out/'COMPLETE.json',dict(schema=SCHEMA,identity=identity,binding_sha256=sha(a.binding),RESULT_sha256=sha(out/'RESULT.json'),seconds=time.monotonic()-start));signal.alarm(0)
 except BaseException:
  write(out/'FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time(),no_retry=True));raise
 finally:
  if controller is not None:controller.close()
if __name__=='__main__':main()
