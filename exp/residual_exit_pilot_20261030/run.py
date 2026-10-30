"""Fixed block12 exit pilot. Independent source/admission required; no auto retry.
No TEST, score, generation, optimizer, export or hyperparameter search.
"""
from pathlib import Path
import argparse,hashlib,json,importlib.util,sys,os,time,signal,traceback,subprocess,resource
ROOT=Path('/data1/liyapeng/program/LLMCL');ARMS=('A','B','C','D')
CP={'PASs_MoE':('583ab5d02e187fbd6a54bd4a05a019e33101deb8af2d9533795253140721441a',3051292733),'CL_MoE':('8550fdeb886a0e942ee9038ab0aeae7e5933b943094ee3af40c0816df592ffa4',2100250526)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8*1024**2),b''):h.update(z)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def load(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def durable(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def data_scope(b):
 A=Path(b['asset_root']);md=read(A/'DATA_SELECTION_METADATA.json')
 assert sha(A/'DATA_SELECTION_METADATA.json')==b['data_metadata_sha256']
 manifest=read(ROOT/md['manifest']);assert sha(ROOT/md['manifest'])==md['manifest_sha256']
 split=read(A/'split.json');assert sha(A/'split.json')==md['split_sha256']
 sel=read(A/'selection.json');assert sha(A/'selection.json')==md['selection_sha256']
 pairs=[];allids=set()
 for task in manifest['tasks']:
  item=task['files']['train'];p=ROOT/item['path'];assert sha(p)==item['sha256']
  rows=[json.loads(z) for z in p.read_text().splitlines()];assert len(rows)==item['count']
  byid={r['id']:r for r in rows};assert len(byid)==len(rows) and not allids.intersection(byid);allids.update(byid)
  pairs.extend((task,byid[i]) for i in split['fit_ids'][str(task['task'])])
 assert len(pairs)==1213 and len(allids)==1313
 domains=list(dict.fromkeys(t['domain'] for t in manifest['tasks']));cal=[]
 for d in domains:
  pool=[x for x in pairs if x[0]['domain']==d];pool.sort(key=lambda x:hashlib.sha256(('pas-static-cal-v1:'+x[1]['id']).encode()).digest());cal.extend(pool[:16])
 def descriptors(xs):return [dict(task=t['task'],domain=t['domain'],id_sha256=hashlib.sha256(r['id'].encode()).hexdigest()) for t,r in xs]
 assert descriptors(cal)==sel['calibration']==md['calibration']
 used={r['id'] for _,r in cal};olddev={i for ids in split['dev_ids'].values() for i in ids}
 pool=[x for x in pairs if x[1]['id'] not in used];assert len(pool)==1133
 out=[]
 for d in domains:
  xs=[x for x in pool if x[0]['domain']==d];xs.sort(key=lambda x:hashlib.sha256(('residual-exit-readout-v1:'+x[1]['id']).encode()).digest());out.extend(xs[:4])
 assert len(cal)==80 and len(out)==20 and descriptors(out)==md['readout']
 assert not {r['id'] for _,r in out}&(used|olddev)
 for _,r in cal+out:
  for image in r['images']:assert sha(ROOT/image)==split['image_sha256'][image]
 return cal,out

def scope(b,method):
 assert b['schema']=='residual-exit-pilot-v1' and method in CP and b['block_index']==11 and b['lambda_factor']==.01
 E=Path(b['source_dir']);P=Path(b['native_source_dir']);A=Path(b['asset_root']);run=A/method
 assert E.resolve()==Path(__file__).resolve().parent and P.resolve().parents[1]==ROOT.resolve()
 for n,h in b['sources'].items():assert sha(E/n)==h,n
 assert sha(b['protocol_path'])==b['protocol_sha256']
 assert sha(b['asset_receipt'])==b['asset_receipt_sha256'];receipt=read(b['asset_receipt'])
 assert receipt['new_persistent_CP_payload_bytes']==0 and receipt['native_source_dir']==str(P)
 cfg=read(run/'config.json');meta=read(run/'checkpoints/latest.json')
 assert cfg['method']==method and cfg['backbone']=='onevision' and cfg['condition']=='pct01'
 assert (meta['sha256'],meta['bytes'])==CP[method] and meta['stage']==19
 cp=(run/meta['path']).resolve();assert cp.is_relative_to((run/'checkpoints').resolve()) and cp.stat().st_size==CP[method][1]
 assert str(cp).startswith('/dev/shm/llmcl-residual-exit-pilot-1004-20261030/')
 for n,h in cfg['execution_source_sha256'].items():assert sha(P/n)==sha(run/'runner_source'/n)==h,n
 assert cfg['execution_source_sha256']==receipt['native_source_sha256']
 for rel,h in receipt['runtime_files'].items():assert sha(ROOT/rel)==h,rel
 assert sha(A/'storage_helper.py')==b['storage_helper_sha256']
 assert sha(b['readout_path'])==b['readout_sha256']=='55e95fd9dde9a76b34aa02bdd12b507e03506feb2137a706ba31067c37da6ef1'
 cal,out=data_scope(b)
 return E,P,A,run,cfg,cal,out

def signature(core,excluded):
 rows={};groups={}
 for n,p in core.named_parameters(remove_duplicate=False):
  if any(n==z or n.startswith(z+'.') for z in excluded):continue
  rows[n]=(tuple(p.shape),str(p.dtype));groups.setdefault(id(p),[]).append(n)
 return rows,sorted(sorted(v) for v in groups.values())

def forward(model,batch,answer,processor,capture,with_logits):
 import torch
 target=processor.tokenizer(answer,add_special_tokens=False,truncation=True,max_length=512)['input_ids'];assert target
 P=batch['input_ids'].shape[1];capture.begin(P,len(target))
 ids=torch.cat([batch['input_ids'],torch.tensor([target+[processor.tokenizer.eos_token_id]],device=model.device)],1)
 emb=model.get_input_embeddings()(ids);grid=batch.get('image_grid_thw')
 if 'pixel_values' in batch:
  visual=model.visual(batch['pixel_values'].to(model.visual.get_dtype()),grid_thw=grid)
  mask=(ids==model.config.image_token_id).unsqueeze(-1).expand_as(emb);assert mask.sum().item()==visual.numel();emb=emb.masked_scatter(mask,visual.to(emb.dtype))
 pos,_=model.get_rope_index(ids,grid,attention_mask=torch.ones_like(ids))
 result=model.model(inputs_embeds=emb,attention_mask=torch.ones_like(ids),position_ids=pos,use_cache=False,return_dict=True)
 logits=model.lm_head(result.last_hidden_state[:,P-1:-1,:])[0] if with_logits else None
 return capture.take(),logits

def main():
 worker_started=time.monotonic()
 ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--method',choices=tuple(CP),required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 b=read(args.binding);E,P,A,run,cfg,cal,out=scope(b,args.method)
 folder=Path(b['output_root'])/args.method
 if not args.execute:print(json.dumps(dict(metadata_validated=True,model_loaded=False,calibration_QA=80,readout_QA=20,TEST_QA=0,method=args.method,output_exists=folder.exists())));return
 assert b['admitted'] is True and os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
 adm=read(b['admission_path']);assert adm['approved'] is True and adm['binding_sha256']==sha(args.binding)
 assert b['limit_seconds']==7200
 uuid=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==b['gpu_uuid'] and uuid.startswith('GPU-')
 folder.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 ps=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split();who=dict(pid=os.getpid(),pgid=os.getpgrp(),uid=os.getuid(),startticks=ps[19]);assert who['pid']==who['pgid'] and who['uid']==1004
 durable(folder/'STARTED.json',dict(binding_sha256=sha(args.binding),identity=who,method=args.method,unix=time.time(),limit_seconds=7200,no_retry=True))
 def expired(*_):raise TimeoutError('fixed 7200s one-teacher pilot deadline')
 signal.signal(signal.SIGALRM,expired);signal.alarm(7200)
 try:
  import torch
  torch.set_num_threads(8);torch.cuda.set_device(0);assert torch.cuda.device_count()==1
  torch.cuda.reset_peak_memory_stats(0) # Before native imports / original CP load.
  gpu_name=torch.cuda.get_device_name(0)
  sys.path.insert(0,str(P));import checkpoints,runtime_utils as rt,onevision_bridge as bridge,frozen_vision as fv
  for mod in (checkpoints,rt,bridge,fv):assert sha(mod.__file__)==cfg['execution_source_sha256'][Path(mod.__file__).name]
  assert rt.ROOT.resolve()==ROOT.resolve()
  probe=load('residual_exit_probe',E/'probe.py');ridge=load('residual_exit_ridge',E/'ridge.py');storage=load('residual_exit_storage',A/'storage_helper.py')
  assert sys.flags.optimize==0,'original loader integrity assertions must be active'
  load_started=time.monotonic()
  payload=checkpoints.load(run,expected=cfg);assert payload['format']=='llmcl-method-graph-v1' and payload['stage']==19 and payload['probe'] is None
  checkpoint_integrity=dict(original_loader_sha256=sha(checkpoints.__file__),
   full_payload_SHA_verified_inside_original_loader=True,expected_checkpoint_sha256=CP[args.method][0],
   actual_temporary_cp_path=str((run/'checkpoints/stage_19.pt').resolve()),
   resolve_containment_guard_preserved=True,assertions_active=True,load_seconds=time.monotonic()-load_started)
  method=payload['method'];teacher=method.model.eval();processor=payload['processor'];del payload
  assert method.name==args.method;teacher.requires_grad_(False);bridge.bind(teacher);bridge.NATIVE_GATE_MODE=False;fv.LIMIT=0
  teacher._llmcl_visual_cache.clear();teacher._llmcl_visual_cache_bytes=0
  before=storage.digest(teacher.core);bufbefore=storage.digest(list(teacher.core.named_buffers(remove_duplicate=False)))
  descriptions=read(b['readout_path'])['methods'][args.method]['layers']
  original_signature=signature(teacher.core,list(descriptions))
  student,folding=probe.make_mean_student(teacher,args.method,descriptions)
  assert signature(student.core,list(descriptions))==original_signature
  registered=list(student.core.named_parameters(remove_duplicate=True))
  student_layout=dict(registered_neural_parameters=int(sum(p.numel() for _,p in registered)),
   registered_neural_weight_bytes=int(sum(p.numel()*p.element_size() for _,p in registered)),
   registration_deduplicated=True,additional_deployment_parameters=0,all_arms_same_registered_graph=True)
  assert student_layout['registered_neural_parameters']==893675552
  dtype_layout={}
  for _,param in registered:
   entry=dtype_layout.setdefault(str(param.dtype),dict(parameters=0,bytes=0));entry['parameters']+=int(param.numel());entry['bytes']+=int(param.numel()*param.element_size())
  student_layout['dtypes']=dtype_layout
  target=student.core.language_model.model.layers[11].mlp.down_proj;assert type(target) is torch.nn.Linear
  prior=target.weight.detach().clone();bias_before=None if target.bias is None else target.bias.detach().clone()
  all_student_params=list(student.core.named_parameters(remove_duplicate=False));target_names=[n for n,p in all_student_params if p is target.weight];assert len(target_names)==1
  def other_digest():return storage.digest([(n,p) for n,p in all_student_params if p is not target.weight])
  others=other_digest();student_buffers=storage.digest(list(student.core.named_buffers(remove_duplicate=False)))
  tc=probe.BlockCapture(teacher);sc=probe.BlockCapture(student)
  fits=ridge.ExitRidge(prior,target.bias,[r['id'] for _,r in cal])
  counts=dict(checkpoint_loads=1,teacher_main_forward=0,student_calibration_forward=0,student_readout_forward=0,teacher_feature_forward=0,generations=0,CE=0,F1=0,TEST_QA=0,backwards=0,optimizer_updates=0,new_checkpoints=0)
  decoder_calls={'n':0}
  def decoder_counter(*_):decoder_calls['n']+=1
  handle=teacher.core.language_model.model.register_forward_pre_hook(decoder_counter)
  def prepare(task,row):
   bridge.bind(teacher);method.set_instruction(task['task']);batch,_=rt.input_batch(row,task,processor,teacher.device)
   n=decoder_calls['n']
   if args.method=='CL_MoE':method.prepare_row(row,batch)
   counts['teacher_feature_forward']+=decoder_calls['n']-n
   return batch
  durable(folder/'MODEL_LOADED.json',dict(checkpoint_sha256=CP[args.method][0],teacher_digest=before,student_parameters=sum(p.numel() for p in student.core.parameters()),additional_deployment_parameters=0,block_index=11,folded_layers=folding,student_other_parameter_signature_preserved=True,weight_prior_dtype=str(prior.dtype),construction_rng=student._llmcl_exit_construction_rng_proof,forward_integrity=student._llmcl_exit_forward_integrity_proof,student_registered_layout=student_layout,checkpoint_integrity=checkpoint_integrity))
  for i,(task,row) in enumerate(cal):
   batch=prepare(task,row)
   with torch.inference_mode(),torch.autocast('cuda',dtype=teacher.dtype):
    t,_=forward(teacher,batch,row['answer'],processor,tc,False)
    s,_=forward(student,batch,row['answer'],processor,sc,False)
   fits.add_qa(row['id'],t,s);counts['teacher_main_forward']+=1;counts['student_calibration_forward']+=1
   durable(folder/f'CAL_{i:03d}_COMPLETE.json',dict(qa_id_sha256=hashlib.sha256(row['id'].encode()).hexdigest(),domain=task['domain'],positions=len(t['positions']),skip_add_closed=True))
   del t,s,batch
  candidates,fitproof=fits.finish();durable(folder/'FIT.json',fitproof)
  records=[]
  with (folder/'READOUT.jsonl').open('x') as stream:
   for i,(task,row) in enumerate(out):
    batch=prepare(task,row)
    with torch.inference_mode(),torch.autocast('cuda',dtype=teacher.dtype):t,tlogits=forward(teacher,batch,row['answer'],processor,tc,True)
    counts['teacher_main_forward']+=1;anchor=None;baseline_metrics=None
    for arm in ARMS:
     with torch.no_grad():target.weight.copy_(candidates[arm])
     with torch.inference_mode(),torch.autocast('cuda',dtype=student.dtype):s,slogits=forward(student,batch,row['answer'],processor,sc,True)
     counts['student_readout_forward']+=1
     if anchor is None:anchor=(s['down_input'].clone(),s['skip'].clone())
     else:assert torch.equal(anchor[0],s['down_input']) and torch.equal(anchor[1],s['skip'])
     metric=probe.activation_metrics(t,s);metric.update(probe.kl_and_agreement(tlogits,slogits))
     if arm=='A':baseline_metrics=(metric['block_output']['mean_squared_vector_error'],metric['teacher_to_student_KL'])
     metric['block_MSE_delta_vs_A']=metric['block_output']['mean_squared_vector_error']-baseline_metrics[0]
     metric['block_MSE_ratio_vs_A']=metric['block_output']['mean_squared_vector_error']/baseline_metrics[0] if baseline_metrics[0]>0 else None
     metric['KL_delta_vs_A']=metric['teacher_to_student_KL']-baseline_metrics[1]
     rec=dict(qa_id_sha256=hashlib.sha256(row['id'].encode()).hexdigest(),domain=task['domain'],task=task['task'],arm=arm,metrics=metric)
     records.append(rec);stream.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n');del s,slogits
    stream.flush();os.fsync(stream.fileno());durable(folder/f'READOUT_{i:03d}_COMPLETE.json',dict(qa_index=i,arms=list(ARMS),qa_id_sha256=hashlib.sha256(row['id'].encode()).hexdigest()))
    del t,tlogits,batch,anchor
  with torch.no_grad():target.weight.copy_(prior)
  if bias_before is None:assert target.bias is None
  else:assert torch.equal(target.bias,bias_before)
  assert other_digest()==others and storage.digest(list(student.core.named_buffers(remove_duplicate=False)))==student_buffers
  assert storage.digest(teacher.core)==before and storage.digest(list(teacher.core.named_buffers(remove_duplicate=False)))==bufbefore
  tc.close();sc.close();handle.remove()
  assert counts['teacher_main_forward']==100 and counts['student_calibration_forward']==80 and counts['student_readout_forward']==80 and len(records)==80
  summary={}
  for arm in ARMS:
   xs=[r for r in records if r['arm']==arm];assert len(xs)==20
   summary[arm]=dict(raw_block_MSE=sum(r['metrics']['block_output']['mean_squared_vector_error'] for r in xs)/20,block_MSE_delta_vs_A=sum(r['metrics']['block_MSE_delta_vs_A'] for r in xs)/20,KL_delta_vs_A=sum(r['metrics']['KL_delta_vs_A'] for r in xs)/20,KL=sum(r['metrics']['teacher_to_student_KL'] for r in xs)/20,top1_agreement=sum(r['metrics']['top1_agreement'] for r in xs)/20,
    block_relative_squared_error=sum(r['metrics']['block_output']['relative_squared_error'] for r in xs)/20,
    domains={d:dict(KL=sum(r['metrics']['teacher_to_student_KL'] for r in xs if r['domain']==d)/4,block_relative_squared_error=sum(r['metrics']['block_output']['relative_squared_error'] for r in xs if r['domain']==d)/4) for d in sorted({r['domain'] for r in xs})})
  assert int(sum(p.numel() for _,p in student.core.named_parameters(remove_duplicate=True)))==student_layout['registered_neural_parameters']
  assert int(sum(p.numel()*p.element_size() for _,p in student.core.named_parameters(remove_duplicate=True)))==student_layout['registered_neural_weight_bytes']
  torch.cuda.synchronize(0)
  temporary_resources=dict(gpu_uuid=uuid,gpu_name=gpu_name,
   peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
   peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
   CPU_process_max_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
   whole_worker_wall_seconds=time.monotonic()-worker_started,
   peak_scope='pilot temporary teacher+student+calibration+FP64 solve+hash checks; reset before load',
   is_deployment_inference_peak=False,is_speedup_measurement=False)
  durable(folder/'TEMPORARY_RESOURCES.json',dict(resources=temporary_resources,student_registered_layout=student_layout))
  durable(folder/'RESULT.json',dict(schema='residual-exit-pilot-v1',method=args.method,counts=counts,block_index=11,arms=summary,readout_sha256=sha(folder/'READOUT.jsonl'),fit_sha256=sha(folder/'FIT.json'),teacher_unchanged=True,student_only_existing_down_weight_changed=True,weights_restored_to_A=True,final_added_parameters=0,not_full_24block_validation=True,not_quality_verification=True,student_registered_layout=student_layout,temporary_resources=temporary_resources,checkpoint_integrity=checkpoint_integrity,temporary_resources_sha256=sha(folder/'TEMPORARY_RESOURCES.json')))
  durable(folder/'COMPLETE.json',dict(identity=who,binding_sha256=sha(args.binding),RESULT_sha256=sha(folder/'RESULT.json'),seconds=time.monotonic()-started));signal.alarm(0)
 except BaseException:durable(folder/'FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time(),no_automatic_retry=True));raise
if __name__=='__main__':main()
