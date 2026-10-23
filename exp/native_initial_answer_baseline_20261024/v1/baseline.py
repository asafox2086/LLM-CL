"""One unadapted native answer baseline. No model/data work on import."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'summary/final_weight_reduction_20261023/native_initial_answer_baseline_v1'
WORK=Path(__file__).parent/'private'
OLD=ROOT/'exp/native_response_reuse_20261024/v2'
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
PRIOR=REPORT.parent/'native_response_reuse_pilot_v2'
BASE=ROOT/'model/SmolVLM-500M-Instruct_a7da5b9'
BUDGET=3600
DOMAINS=('natural','medical','geography','finance','traffic')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)
 if os.name!='nt':
  fd=os.open(path.parent,os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def metadata():
 policy=json.loads((PORT/'execution_policy.json').read_text());v=policy['conditions']['pct01']
 path=ROOT/v['manifest'];assert sha(path)==v['manifest_sha256']
 screen=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'
 anchor=json.loads((screen/'READY.json').read_text());assert sha(path)==anchor['manifest_sha256']
 assert sha(screen/'SPLIT.json')==anchor['split_sha256']
 manifest=json.loads(path.read_text());split=json.loads((screen/'SPLIT.json').read_text())
 assert len(manifest['tasks'])==19 and manifest['seed']==42 and split['DEV_QA']==100
 ids=[i for task in manifest['tasks'] for i in split['dev_ids'][str(task['task'])]]
 assert len(ids)==len(set(ids))==100
 return path,manifest,split,hashlib.sha256(''.join(i+'\n' for i in ids).encode()).hexdigest()

def data():
 """Read original TRAIN containers only to select the already frozen DEV IDs."""
 path,manifest,split,_=metadata()
 dev={};seen=[]
 for task in manifest['tasks']:
  k=task['task'];wanted=split['dev_ids'][str(k)];wanted_set=set(wanted)
  source=ROOT/task['files']['train']['path'];assert sha(source)==task['files']['train']['sha256']
  selected={}
  for line in source.read_text().splitlines():
   row=json.loads(line)
   if row['id'] in wanted_set:
    assert row['id'] not in selected;selected[row['id']]=row
  assert set(selected)==wanted_set
  dev[k]=[selected[i] for i in wanted];seen.extend(wanted)
 assert len(seen)==len(set(seen))==100
 assert Counter(t['domain'] for t in manifest['tasks'] for r in dev[t['task']])==Counter({d:20 for d in DOMAINS})
 return path,manifest,split,dev

def freeze():
 """Metadata/source binding only. No Torch import, model or CPU gradient gate."""
 assert 'torch' not in sys.modules and not (REPORT/'READY.json').exists()
 review=json.loads((REPORT/'SOURCE_REVIEW.json').read_text())
 assert review['decision']=='accept_fixed_unadapted_forward_baseline_source_only'
 assert review['protocol_sha256']==sha(REPORT/'PROTOCOL.md')
 assert review['necessity_review_sha256']==sha(REPORT/'NECESSITY_REVIEW.json')
 assert review['previous_source_review_sha256']==sha(REPORT/'SOURCE_REVIEW_PRE_METADATA_FIX.json')
 assert review['metadata_failure_sha256']==sha(REPORT/'METADATA_FREEZE_FAILURE.json')
 assert set(review['sources'])=={(Path(__file__).parent/n).relative_to(ROOT).as_posix() for n in ('baseline.py','dispatch.py')}
 for n,h in review['sources'].items():assert sha(ROOT/n)==h
 necessary=json.loads((REPORT/'NECESSITY_REVIEW.json').read_text())
 assert necessary['protocol_sha256']==review['protocol_sha256']
 assets=json.loads((PRIOR/'ASSET_RECEIPT.json').read_text())
 deps=[OLD/'core.py',OLD/'runtime.py',PORT/'runtime_utils.py',PORT/'retrieval.py',PORT/'resource_monitor.py',PORT/'training_validity.py',
  ROOT/'analysis/onevision_research_device_entry.py',PORT/'execution_policy.json',
  ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json',
  ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/READY.json']
 deps.extend(PRIOR/n for n in ('ASSET_RECEIPT.json','ASSET_TRANSPORT.json','CPU_GATE_FAILURE.json','FAILURE_REVIEW.json','SOURCE_REVIEW.json'))
 deps.extend(REPORT/n for n in ('SOURCE_REVIEW_PRE_METADATA_FIX.json','METADATA_FREEZE_FAILURE.json'))
 # Bind installed native/generation implementation without importing those modules.
 import importlib.metadata
 relative=['transformers/modeling_utils.py','transformers/generation/utils.py','transformers/generation/configuration_utils.py',
  'transformers/models/idefics3/modeling_idefics3.py','transformers/models/idefics3/processing_idefics3.py',
  'transformers/models/idefics3/configuration_idefics3.py','transformers/models/idefics3/image_processing_idefics3.py',
  'transformers/models/idefics2/image_processing_idefics2.py',
  'transformers/models/llama/modeling_llama.py','transformers/models/siglip/modeling_siglip.py','torch/__init__.py']
 environment={};packages={}
 for name in relative:
  package,inside=name.split('/',1)
  spec=importlib.util.find_spec(package);assert spec and spec.submodule_search_locations
  locations=list(spec.submodule_search_locations);assert len(locations)==1 and spec.origin
  package_root=Path(locations[0]).resolve()
  packages[package]=dict(origin=str(Path(spec.origin).resolve()),root=str(package_root))
  actual=(package_root/inside).resolve();assert actual.is_file() and actual.is_relative_to(package_root)
  environment[str(actual)]=sha(actual)
 evidence=[REPORT/n for n in ('PROTOCOL.md','NECESSITY_REVIEW.json','SOURCE_REVIEW.json')]
 manifest_path,_,_,dev_order_sha=metadata()
 # Source-approved asset receipt is reused, not reacquired or relabelled as a new download.
 value=dict(kind='fixed_unadapted_native_baseline_v1',budget_seconds=BUDGET,DEV_QA=100,FIT_updates=0,TEST_QA=0,
  sources=review['sources'],evidence={p.relative_to(ROOT).as_posix():sha(p) for p in evidence},
  dependencies={p.relative_to(ROOT).as_posix():sha(p) for p in deps},environment=environment,
  package_locations=packages,
  base_files=assets['base_files'],expected_coefficients=assets['serialized_weight_coefficients'],
  manifest_sha256=sha(manifest_path),DEV_ID_order_sha256=dev_order_sha,
  interpreter=str(Path(sys.executable).resolve()),versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers')},
  no_model_or_QA_or_Torch_execution=True,no_CPU_gradient_gate=True,no_automatic_training=True)
 assert 'torch' not in sys.modules
 write(REPORT/'READY.json',value)

def check_ready(assets=False):
 ready=json.loads((REPORT/'READY.json').read_text());assert ready['kind']=='fixed_unadapted_native_baseline_v1'
 assert ready['budget_seconds']==BUDGET and ready['DEV_QA']==100 and ready['FIT_updates']==ready['TEST_QA']==0
 assert str(Path(sys.executable).resolve())==ready['interpreter']
 for field in ('sources','evidence','dependencies'):
  for n,h in ready[field].items():assert sha(ROOT/n)==h,n
 for n,h in ready['environment'].items():assert sha(n)==h,n
 manifest_path,_,_,dev_order_sha=metadata()
 assert sha(manifest_path)==ready['manifest_sha256'] and dev_order_sha==ready['DEV_ID_order_sha256']
 if assets:
  for n,h in ready['base_files'].items():assert sha(ROOT/n)==h,n
 return ready

def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
  uid=int(next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')))
  return dict(pid=pid,startticks=s[19],uid=uid,pgid=int(s[2]))
 except FileNotFoundError:return None

def available():
 dev=load(ROOT/'analysis/onevision_research_device_entry.py','baseline_device')
 cards=dev.inventory();ours={g for g,c in cards.items() if any(p['uid']==os.getuid() for p in c['owners'])};c=cards[1]
 return not c['owners'] and c['memory_MiB']<500 and c['utilization']<10 and not (ours&{0,1}) and len(ours|{1})<=3

def worker():
 check_ready();assert not (REPORT/'STARTED.json').exists() and not WORK.exists()
 if not available():return 75
 tick=time.monotonic();who=identity(os.getpid());assert who['uid']==os.getuid() and who['pgid']==who['pid']
 write(REPORT/'STARTED.json',dict(**who,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
  started_monotonic=tick,started_unix=time.time(),budget_seconds=BUDGET,ready_sha256=sha(REPORT/'READY.json'),only_GPU=1,TEST_QA=0))
 def expired(*unused):raise TimeoutError('Fixed unadapted baseline model budget expired; no repeat')
 signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,BUDGET-(time.monotonic()-tick)))
 os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
 ready=check_ready(assets=True);manifest_path,manifest,split,dev=data()
 import torch
 from transformers import AutoProcessor,Idefics3ForConditionalGeneration,GenerationConfig
 sys.path.insert(0,str(OLD));sys.path.insert(0,str(PORT))
 from core import layout,digest
 from runtime import NativeRuntime,policy
 from resource_monitor import ResourceMonitor
 assert torch.cuda.is_bf16_supported(),'No dtype fallback'
 torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
 WORK.mkdir();monitor=ResourceMonitor(WORK);monitor.snapshot('process_before_model')
 processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
 native_policy=policy(processor,GenerationConfig.from_pretrained(BASE,local_files_only=True))
 import inspect
 assert str(Path(inspect.getfile(type(processor.image_processor))).resolve()) in ready['environment']
 model,loading=Idefics3ForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,
  attn_implementation={'vision_config':'eager','text_config':'sdpa'},local_files_only=True,output_loading_info=True)
 assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),loading
 assert model.model.vision_model.config._attn_implementation=='eager' and model.model.text_model.config._attn_implementation=='sdpa'
 model.requires_grad_(False);model=model.cuda().eval()
 before=layout(model);assert before['trainable_coordinates']==0 and before['weight_coefficients']==ready['expected_coefficients']
 assert all(p.dtype==torch.bfloat16 for p in model.parameters())
 initial=digest(model.state_dict().values());monitor.snapshot('inference_before',model=model)
 class ObservedRuntime(NativeRuntime):
  def generate(self,batch,cap):
   assert batch['input_ids'].shape[1]+cap<=self.context_limit,'Generation cap would exceed native context'
   kwargs={k:v for k,v in batch.items() if not k.startswith('_')}
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    out=self.model.generate(**kwargs,do_sample=False,num_beams=1,max_new_tokens=cap,use_cache=True,
     eos_token_id=self.policy['generation_stop_ids'],pad_token_id=self.policy['pad_id'])
   response=out[0,batch['input_ids'].shape[1]:]
   assert len(response)>0
   last=int(response[-1]);terminal='eos' if last in self.policy['generation_stop_ids'] else 'cap'
   assert terminal=='eos' or len(response)==cap
   self.last_generation_evidence=dict(final_token_id=last,terminal_reason=terminal)
   self.requests.append(dict(kind='generation',output_tokens=len(response),cap=cap))
   return self.processor.tokenizer.decode(response,skip_special_tokens=True),len(response)
 runtime=ObservedRuntime(processor,model,WORK,native_policy,image_SHA256=split['image_sha256']);runtime.phase='DEV'
 temporary=WORK/'predictions.jsonl.partial'
 with temporary.open('x',encoding='utf-8') as f:
  for task in manifest['tasks']:
   for row in dev[task['task']]:
    batch,details=runtime.input_batch(row,task,processor,model.device)
    started=time.monotonic();prediction,n=runtime.generate(batch,task['max_new_tokens'])
    output=dict(id=row['id'],task=task['task'],domain=task['domain'],prediction=prediction,references=row['answers'],
     generation_cap=task['max_new_tokens'],output_tokens=n,generation_seconds=time.monotonic()-started,**details)
    # Record exact final generated token and terminal reason without another model call.
    output.update(runtime.last_generation_evidence)
    f.write(json.dumps(output,ensure_ascii=False)+'\n');f.flush()
   os.fsync(f.fileno())
 temporary.replace(WORK/'predictions.jsonl')
 assert len(runtime.requests)==100 and all(x['kind']=='generation' for x in runtime.requests)
 assert len(runtime.input_records)==100 and all(x['phase']=='DEV' for x in runtime.input_records)
 final=digest(model.state_dict().values());after=layout(model);assert initial==final and before==after
 write(WORK/'INPUT_RECORDS.json',runtime.input_records);write(WORK/'REQUESTS.json',runtime.requests)
 write(WORK/'MODEL.json',dict(before=before,after=after,state_before=initial,state_after=final,policy=native_policy,
  asset_files=ready['base_files'],manifest_sha256=sha(manifest_path),local_FIT_updates=0,model_eval=True,loading_info=loading))
 monitor.done.set();monitor.thread.join(timeout=6);monitor.snapshot('inference_after',model=model);monitor.snapshot('run_finished',model=model)
 rows=[json.loads(x) for x in (WORK/'resources.jsonl').read_text().splitlines()]
 keys=('VmRSS_bytes','Pss_bytes','cpu_uss_bytes','gpu_allocated_bytes','gpu_reserved_bytes','gpu_process_nvml_bytes')
 write(WORK/'RESOURCES.json',dict(sampling_seconds=5,observed_peak_bytes={k:max((x[k] for x in rows if x.get(k) is not None),default=None) for k in keys},
  note='Sampled full-process inference baseline; not a controlled speed/memory comparison or parameter-count proxy.'))
 elapsed=time.monotonic()-tick;assert elapsed<BUDGET
 files={p.relative_to(ROOT).as_posix():sha(p) for p in WORK.iterdir() if p.is_file()}
 elapsed=time.monotonic()-tick;assert elapsed<BUDGET
 write(REPORT/'COMPLETE.json',dict(complete=True,worker=who,ready_sha256=sha(REPORT/'READY.json'),model_RUN_seconds=elapsed,
  DEV_QA=100,local_FIT_updates=0,TEST_QA=0,requests=100,file_sha256=files))
 signal.setitimer(signal.ITIMER_REAL,0)
 return 0

def verify():
 assert not (REPORT/'VERIFY_STARTED.json').exists() and not (REPORT/'VERIFIED.json').exists()
 ready=check_ready();complete=json.loads((REPORT/'COMPLETE.json').read_text());start=json.loads((REPORT/'STARTED.json').read_text())
 exited=json.loads((REPORT/'ACTUAL_EXIT.json').read_text())
 write(REPORT/'VERIFY_STARTED.json',dict(complete_sha256=sha(REPORT/'COMPLETE.json'),single_score_pass=True))
 assert complete['complete'] and complete['DEV_QA']==100 and complete['local_FIT_updates']==complete['TEST_QA']==0
 assert complete['ready_sha256']==sha(REPORT/'READY.json')==start['ready_sha256']
 assert exited['identity']==complete['worker']=={k:start[k] for k in ('pid','startticks','uid','pgid')} and exited['code']==0
 assert 0<complete['model_RUN_seconds']<=exited['observed_elapsed_bound_seconds']<=BUDGET
 actual={p.relative_to(ROOT).as_posix() for p in WORK.iterdir() if p.is_file()}
 assert actual==set(complete['file_sha256'])
 for n,h in complete['file_sha256'].items():assert sha(ROOT/n)==h
 _,manifest,split,dev=data();expected=[(t,r) for t in manifest['tasks'] for r in dev[t['task']]]
 predictions=[json.loads(x) for x in (WORK/'predictions.jsonl').read_text().splitlines()]
 inputs=json.loads((WORK/'INPUT_RECORDS.json').read_text());requests=json.loads((WORK/'REQUESTS.json').read_text())
 assert len(predictions)==len(inputs)==len(requests)==100
 assert [r['id'] for _,r in expected]==[r['id'] for r in predictions]==[r['id'] for r in inputs]
 assert all(r['phase']=='DEV' for r in inputs) and all(r['kind']=='generation' for r in requests)
 model=json.loads((WORK/'MODEL.json').read_text())
 assert model['state_before']==model['state_after'] and model['before']==model['after'] and model['before']['trainable_coordinates']==0
 assert model['asset_files']==ready['base_files'] and model['before']['weight_coefficients']==ready['expected_coefficients']
 tree=ast.parse((PORT/'runtime_utils.py').read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('score','tokens')]
 assert len(defs)==2;namespace=dict(re=re,Counter=Counter);exec(compile(ast.Module(body=defs,type_ignores=[]),'<original-score>', 'exec'),namespace)
 domain_scores={d:[] for d in DOMAINS};scores={}
 for (task,row),pred,inp,req in zip(expected,predictions,inputs,requests):
  assert pred['references']==row['answers'] and pred['task']==task['task'] and pred['domain']==task['domain']
  assert pred['generation_cap']==req['cap']==task['max_new_tokens'] and pred['output_tokens']==req['output_tokens']
  assert pred['prepared_input_SHA256']==inp['prepared_input_SHA256']
  assert 0<pred['output_tokens']<=pred['generation_cap'] and pred['terminal_reason'] in ('eos','cap')
  assert (pred['final_token_id'] in model['policy']['generation_stop_ids'])==(pred['terminal_reason']=='eos')
  assert pred['terminal_reason']=='eos' or pred['output_tokens']==pred['generation_cap']
  assert len(pred['image_SHA256'])==len(row['images'])
  for image,observed in zip(row['images'],pred['image_SHA256']):assert observed==dict(path=image,sha256=split['image_sha256'][image])
  value=float(namespace['score'](pred['prediction'],row['answers'])['token_f1']);assert math.isfinite(value)
  scores[row['id']]=dict(domain=task['domain'],score=value);domain_scores[task['domain']].append(value)
 resources=[json.loads(x) for x in (WORK/'resources.jsonl').read_text().splitlines()]
 assert resources and all(r['pid']==start['pid'] and r['visible_gpu']=='1' for r in resources)
 assert any(r['phase']=='run_finished' for r in resources)
 means={d:sum(v)/len(v) for d,v in domain_scores.items()};assert all(len(v)==20 for v in domain_scores.values())
 write(REPORT/'VERIFIED.json',dict(kind=ready['kind'],complete_sha256=sha(REPORT/'COMPLETE.json'),ready_sha256=sha(REPORT/'READY.json'),
  input_records_sha256=sha(WORK/'INPUT_RECORDS.json'),predictions_sha256=sha(WORK/'predictions.jsonl'),
  domains=means,macro=sum(means.values())/5,scores=scores,layout=model['before'],model_RUN_seconds=complete['model_RUN_seconds'],
  local_FIT_updates=0,TEST_QA=0,no_method_or_parameter_avoidability_claim=True,no_automatic_training=True))

if __name__=='__main__':
 import argparse,traceback
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','worker','verify']);args=ap.parse_args()
 try:
  if args.action=='worker':raise SystemExit(worker())
  elif args.action=='freeze':freeze()
  else:verify()
 except Exception:
  if args.action=='worker':write(REPORT/'FAILURE.json',dict(traceback=traceback.format_exc(),no_repeat=True,not_quality_negative=True))
  raise
