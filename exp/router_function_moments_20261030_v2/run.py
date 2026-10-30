"""Independent one-model TRAIN-only observation worker. No automatic retries.
Two methods require two distinct processes; external reviewed controller runs serially.
--execute requires a final binding plus independent admission of its exact digest.
"""
import argparse,hashlib,importlib.util,json,os,signal,sys,time,traceback,gc,subprocess
from pathlib import Path
SCHEMA='router-function-moments-v1'
MANIFEST_SHA='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
METHODS={'PASs_MoE':('583ab5d02e187fbd6a54bd4a05a019e33101deb8af2d9533795253140721441a',3051292733,'79edcca96dd315260816932438b7c0ec57835f5761d2ec0b88d8ecdb5c837fb6'),
'CL_MoE':('8550fdeb886a0e942ee9038ab0aeae7e5933b943094ee3af40c0816df592ffa4',2100250526,'125d1f9ead67d5e4d222f5fcfc05897b582f90bbe2bdbd1603e12534aa91dc02')}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def durable(p,obj):
 with Path(p).open('x',encoding='utf8') as f:
  json.dump(obj,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def scope(b,method):
 assert b['schema']==SCHEMA and method in METHODS
 root=Path(b['source_root']);port=Path(b['native_source_dir']);run=Path(b['methods'][method]['original_run'])
 assert str(root)=='/data2/liyapeng_grp/program/LLMCL'
 assert all(Path(b[k]).is_absolute() for k in ('manifest','split','selection_path','protocol_path','output_root','admission_path','storage_helper','native_source_dir'))
 assert Path(__file__).resolve().parent==Path(b['observation_source_dir']).resolve()
 for n,h in b['observation_sources'].items():assert sha(Path(__file__).parent/n)==h,n
 assert sha(b['protocol_path'])==b['protocol_sha256']
 assert sha(b['storage_helper'])==b['storage_helper_sha256']
 assert sha(b['manifest'])==MANIFEST_SHA and sha(b['split'])==b['split_sha256']
 assert sha(b['selection_path'])==b['selection_sha256']
 sel=read(b['selection_path']);assert sel['schema']=='pas-static-selection-v1' and sel['calibration_QA']==80 and sel['TEST_QA']==0
 assert sel['manifest_sha256']==MANIFEST_SHA and sel['split_sha256']==b['split_sha256']
 cpbytes=METHODS[method][1];cpsha=METHODS[method][0];cfgsha=METHODS[method][2]
 assert sha(run/'config.json')==cfgsha==b['methods'][method]['config_sha256']
 cfg=read(run/'config.json');assert cfg['method']==method and cfg['condition']=='pct01' and cfg['backbone']=='onevision'
 meta=read(run/'checkpoints/latest.json')
 assert meta['path']=='checkpoints/stage_19.pt' and meta['stage']==19 and meta['bytes']==cpbytes and meta['sha256']==cpsha
 assert b['methods'][method]['checkpoint_sha256']==cpsha and (run/meta['path']).stat().st_size==cpbytes
 for n,h in cfg['execution_source_sha256'].items():assert sha(port/n)==sha(run/'runner_source'/n)==h,n
 for rel,h in b['runtime_files'].items():assert sha(root/rel)==h,rel
 manifest=read(b['manifest']);split=read(b['split']);assert len(manifest['tasks'])==19 and manifest['seed']==42
 assert split['FIT_QA']==1213 and split['DEV_QA']==100 and split['TEST_rows_opened']==0
 domains=list(dict.fromkeys(t['domain'] for t in manifest['tasks']));assert len(domains)==5
 fit=[];allids=set()
 # Read only original TRAIN files, never any TEST/readout predictions or scores.
 for task in manifest['tasks']:
  f=task['files']['train'];p=root/f['path'];assert sha(p)==f['sha256']
  rows=[json.loads(s) for s in p.read_text().splitlines()];assert len(rows)==f['count']
  byid={r['id']:r for r in rows};assert len(byid)==len(rows) and not allids.intersection(byid);allids.update(byid)
  fit.extend((task,byid[i]) for i in split['fit_ids'][str(task['task'])])
 assert len(allids)==1313 and len(fit)==1213
 selected=[]
 for domain in domains:
  pool=[p for p in fit if p[0]['domain']==domain]
  pool.sort(key=lambda p:hashlib.sha256(('pas-static-cal-v1:'+p[1]['id']).encode()).digest())
  assert len(pool)>=16;selected.extend(pool[:16])
 summary=[dict(task=t['task'],domain=t['domain'],id_sha256=hashlib.sha256(r['id'].encode()).hexdigest()) for t,r in selected]
 assert summary==sel['calibration'] and len(selected)==80
 ids_sha=hashlib.sha256(''.join(str(t['task'])+':'+r['id']+'\n' for t,r in selected).encode()).hexdigest()
 assert ids_sha==b['calibration_ids_sha256']
 for _,row in selected:
  for image in row['images']:assert sha(root/image)==split['image_sha256'][image]
 assert b['limit_seconds']==7200 and b['cuda_visible_devices']=='1'
 assert b['closure_atol']==0.0 and b['closure_rtol']==0.0 # Native reconstruction expected bitwise output closure.
 return root,port,run,cfg,selected,ids_sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--method',choices=tuple(METHODS),required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 b=read(a.binding);root,port,run,cfg,rows,ids_sha=scope(b,a.method)
 output=Path(b['output_root'])/a.method
 if not a.execute:
  print(json.dumps(dict(metadata_validated=True,method=a.method,model_loaded=False,TRAIN_forward_QA=80,TEST_QA=0,output_exists=output.exists())));return
 assert b['admitted'] is True
 admission=read(b['admission_path']);assert admission['approved'] is True and admission['binding_sha256']==sha(a.binding)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
 uuid=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 assert uuid==b['gpu_uuid'] and uuid.startswith('GPU-')
 # This exact output directory is a no_retry claim. Any prior directory blocks execution.
 output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 proc=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
 ident=dict(pid=os.getpid(),pgid=os.getpgrp(),uid=os.getuid(),startticks=proc[19]);assert ident['pid']==ident['pgid'] and ident['uid']==1084
 durable(output/'STARTED.json',dict(schema=SCHEMA,method=a.method,identity=ident,binding_sha256=sha(a.binding),unix=time.time(),limit_seconds=7200,no_retry=True))
 def expire(*args):raise TimeoutError('one-model 7200s fixed deadline')
 signal.signal(signal.SIGALRM,expire);signal.alarm(7200)
 try:
  import torch
  torch.set_num_threads(8);torch.cuda.set_device(0);assert torch.cuda.device_count()==1
  sys.path.insert(0,str(port))
  import checkpoints,runtime_utils as rt,onevision_bridge as bridge,frozen_vision as fv
  for mod in (checkpoints,rt,bridge,fv):assert sha(mod.__file__)==cfg['execution_source_sha256'][Path(mod.__file__).name]
  assert rt.ROOT.resolve()==root.resolve()
  storage=load('router_moment_storage',b['storage_helper'])
  helper=load('router_moment_native_hooks',Path(__file__).parent/'worker_draft.py')
  stats=load('router_moment_gpu_stats',Path(__file__).parent/'gpu_statistics.py')
  cp=run/'checkpoints/stage_19.pt';assert sha(cp)==METHODS[a.method][0]
  payload=checkpoints.load(run,expected=cfg)
  assert payload['format']=='llmcl-method-graph-v1' and payload['stage']==19 and payload['probe'] is None
  method=payload['method'];model=method.model.eval();processor=payload['processor'];del payload
  assert method.name==a.method;model.requires_grad_(False);bridge.bind(model);bridge.NATIVE_GATE_MODE=False
  model._llmcl_visual_cache.clear();model._llmcl_visual_cache_bytes=0;fv.LIMIT=0
  before=storage.digest(model.core);buffers=storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
  generation_before=model.core.language_model.generation_config.to_dict()
  counts=dict(checkpoint_loads=1,TRAIN_QA=0,answer_decoder_forward=0,feature_decoder_forward=0,feature_cache_hit=0,generations=0,CE=0,F1=0,TEST_QA=0,backwards=0,optimizer_updates=0,new_checkpoints=0)
  decoder_counts={'value':0}
  def decpre(*args):decoder_counts['value']+=1
  dec_handle=model.core.language_model.model.register_forward_pre_hook(decpre)
  pending=[]
  def emit(row):pending.append(row)
  hooks=helper.NativeMomentHooks(model=model,method_name=a.method,statistics=stats,emit=emit,closure_atol=b['closure_atol'],closure_rtol=b['closure_rtol'])
  durable(output/'MODEL_LOADED.json',dict(method=a.method,checkpoint_sha256=METHODS[a.method][0],initial_core_digest=before,layers=list(hooks.layers),calibration_ids_sha256=ids_sha,gpu_uuid=uuid,raw_neural_layout=storage.layout(model.core)))
  with (output/'QA_MOMENTS.jsonl').open('x',encoding='utf8') as stream:
   for index,(task,row) in enumerate(rows):
    assert not pending
    began=time.monotonic();method.set_instruction(task['task'])
    batch,_=rt.input_batch(row,task,processor,model.device)
    feature_before=decoder_counts['value']
    if a.method=='CL_MoE':
     hit=row['question'] in method.feature_cache
     method.prepare_row(row,batch)
     counts['feature_cache_hit']+=int(hit)
    counts['feature_decoder_forward']+=decoder_counts['value']-feature_before
    target=processor.tokenizer(row['answer'],add_special_tokens=False,truncation=True,max_length=512)['input_ids'];assert target
    hooks.begin_qa(qa_id=row['id'],domain=task['domain'],prompt_length=batch['input_ids'].shape[1],answer_tokens=len(target))
    dec_before=decoder_counts['value']
    with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype),hooks.dispatch():
     result=helper.forward_hidden_only(model=model,batch=batch,answer=row['answer'],processor=processor)
    hooks.end_qa()
    assert decoder_counts['value']-dec_before==1
    assert len(pending)==168 and {r['layer_id'] for r in pending}==set(hooks.layers)
    assert all(r['qa_id']==row['id'] and r['domain']==task['domain'] and r['closure']['all_elements_within_tolerance'] and r['actual_native_layer_output_closure']['bitwise_equal'] for r in pending)
    # One complete QA is a durable unit; no retry consumes this directory.
    for r in pending:
     stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    stream.flush();os.fsync(stream.fileno())
    durable(output/f'QA_{index:03d}_COMPLETE.json',dict(qa_index=index,qa_id_sha256=hashlib.sha256(row['id'].encode()).hexdigest(),task=task['task'],domain=task['domain'],layers=168,target_tokens=result['target_tokens_including_eos'],seconds=time.monotonic()-began))
    pending.clear();counts['TRAIN_QA']+=1;counts['answer_decoder_forward']+=1
  hooks.close();dec_handle.remove()
  assert counts['TRAIN_QA']==counts['answer_decoder_forward']==80
  assert storage.digest(model.core)==before and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==buffers
  assert model.core.language_model.generation_config.to_dict()==generation_before
  moments_sha=sha(output/'QA_MOMENTS.jsonl')
  durable(output/'RESULT.json',dict(schema=SCHEMA,method=a.method,counts=counts,QA_moments_sha256=moments_sha,raw_rows=168*80,calibration_ids_sha256=ids_sha,checkpoint_sha256=METHODS[a.method][0],initial_and_final_core_digest=before,weights_and_buffers_unchanged=True,mean_gate_fitted=False,model_replaced=False,uniform_new_forward=False,not_quality_verification=True))
  durable(output/'COMPLETE.json',dict(schema=SCHEMA,identity=ident,RESULT_sha256=sha(output/'RESULT.json'),binding_sha256=sha(a.binding),seconds=time.monotonic()-started))
  signal.alarm(0)
 except BaseException as exc:
  if 'pending' in locals() and pending:
   durable(output/'FAILED_QA_PARTIAL.json',dict(complete=False,rows=pending,not_for_aggregate=True))
  durable(output/'FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time(),capture_metadata=getattr(exc,'capture_metadata',None)));raise
if __name__=='__main__':main()
