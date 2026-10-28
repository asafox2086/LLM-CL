"""Prepare two committed-CP IO recovery branches. Default inspect; --execute writes
only exclusive NEW branch directories/report, never starts a GPU/coordinator,
changes a source/job/request, or changes an original failed RUN. No scorer/load.
Old Qwen SMo has no CP and is deliberately a separate fresh-run follow-up.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,time,traceback
ROOT=Path('/data1/liyapeng/program/LLMCL')
FRAME=ROOT/'exp/two_backbone_native_checkpoint_20261027'
REPORT=ROOT/'summary/storage_recovery_20261028'
OUT=REPORT/'BRANCH_PREPARATION'
MANIFEST='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
JOBS=[dict(key='pct01:OSFT',host='new3090',backbone='qwen',method='OSFT',gpu=2,stage=2,
 original='runs/qwen/pct01/OSFT/training_20261027_234229',branch='runs/qwen/pct01/OSFT/recovery_io_20261028_S02',
 bytes=8516943791,sha256='ba9bf9583ed2bd81a882f0b6eeb46531a1fcd25d3d8314dc869ad50c1c1075d6'),
 dict(key='pct01:SMoLoRA',host='new3090_onevision',backbone='onevision',method='SMoLoRA',gpu=1,stage=13,
 original='runs/onevision/pct01/SMoLoRA/training_20261027_121553',branch='runs/onevision/pct01/SMoLoRA/recovery_io_20261028_S13',
 bytes=2219762819,sha256='9d74ed4971158090b3e028677da83a759c1ff27a807d723aeb74f66a77f4efcb')]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def statkey(p):
 s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
def safe(p):
 assert p.resolve()==p and p.is_relative_to(ROOT) and p.stat().st_uid==1004
 q=p
 while q!=ROOT:
  assert not q.is_symlink(),str(q);q=q.parent
 return p
def dead_sources():
 for pid,start in ((8265,'60882'),(2840219,'45864072'),(2154888,'41744412'),(122374,None)):
  p=Path('/proc')/str(pid)/'stat'
  if p.exists():
   f=p.read_text().rsplit(') ',1)[1].split()
   assert f[0]=='Z' or (start is not None and f[19]!=start),(pid,'original process still exists')
def no_run_process():
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=1004:continue
   args=(proc/'cmdline').read_bytes().split(b'\0')
   for j in JOBS:
    original=str(FRAME/j['original']).encode();branch=str(FRAME/j['branch']).encode()
    assert not any(original==x or branch==x for x in args),('RUN process active',proc.name)
  except (FileNotFoundError,ProcessLookupError):continue
def inspect(job):
 run=safe(FRAME/job['original']);dst=FRAME/job['branch'];assert not dst.exists()
 c=read(run/'config.json');m=read(run/'checkpoints/latest.json');stage=job['stage']
 assert c['method']==job['method'] and c['backbone']==job['backbone'] and c['condition']=='pct01'
 assert c['manifest_sha256']==MANIFEST and c['seed']==42 and c['epochs']==1 and c['batch_size']==16 and c['micro_batch_size']==1
 assert c['checkpoint_all_stages'] and not c['preflight_only']
 assert (run/'resources.jsonl').is_file()
 sources={p.name:sha(p) for p in FRAME.glob('*.py')}
 assert sources==c['execution_source_sha256']
 assert m['stage']==stage and m['bytes']==job['bytes'] and m['sha256']==job['sha256']
 assert m['path']==f'checkpoints/stage_{stage:02d}.pt' and m['full_architecture_and_runtime'] and m['manifest_sha256']==MANIFEST
 cp=safe(run/m['path']);assert cp.is_file() and cp.stat().st_size==m['bytes']
 assert read(cp.with_suffix('.json'))==m
 state=read(run/'run_status.json');assert state['status']=='evaluation' and state['completed_stage']==stage and state['active_stage']==stage+1
 for name in ('PAUSE_REQUEST.json','RESUME_REQUEST.json'):assert not (run/name).exists(),name
 policy=read(FRAME/'execution_policy.json');manifest=ROOT/policy['conditions']['pct01']['manifest']
 assert sha(manifest)==MANIFEST
 tasks=read(manifest)['tasks'];files=[run/'config.json',run/'checkpoints/latest.json',safe(run/'resources.jsonl')]
 for name,h in sources.items():
  p=safe(run/'runner_source'/name);assert sha(p)==h;files.append(p)
 p=run/'runner_source/execution_policy.json'
 if p.exists():files.append(safe(p))
 for k in range(1,stage+1):
  sm=safe(run/f'stages/stage_{k:02d}.json');cm=safe(run/f'checkpoints/stage_{k:02d}.json');v=safe(run/f'validity/stage_{k:02d}.json')
  a,b,d=read(sm),read(cm),read(v)
  assert a['stage']==b['stage']==k and a['complete'] and a['weights_saved'] and b['full_architecture_and_runtime']
  assert b['method']==job['method'] and b['manifest_sha256']==MANIFEST and d['passed']
  files.extend((sm,cm,v))
 # Copy only fully submitted evaluation cells, as opaque original bytes.
 for k in range(stage+1):
  ids=range(1,20) if k==0 else (k,k+1)
  for tid in ids:
   base=run/f'evaluations/stage_{k:02d}/T{tid:02d}_test_predictions'
   for suffix in ('.jsonl','.scores.json','.datasets.json'):
    p=safe(Path(str(base)+suffix));assert p.is_file();files.append(p)
 raw=(run/'logs/events.jsonl').read_bytes();keep=[];positions={k:[] for k in range(1,stage+1)};discard=0
 for line in raw.splitlines(keepends=True):
  event=json.loads(line);assert event['event']=='train_update'
  if event['stage']<=stage:
   k=event['stage'];assert k in positions;positions[k].append(event);keep.append(line)
  else:discard+=1
 for k,events in positions.items():
  n=tasks[k-1]['files']['train']['count'];expected=math.ceil(n/16)
  assert len(events)==expected and [e['step'] for e in events]==list(range(1,expected+1))
  assert all(e['qa_seen']==min(16*e['step'],n) and e['total_qa']==n for e in events)
 assert len(files)==len(set(files))
 return dict(run=run,dst=dst,config=c,meta=m,cp=cp,cp_stat=statkey(cp),files=files,events=b''.join(keep),
  source_sha256=sources,original_events_sha256=hashlib.sha256(raw).hexdigest(),committed_event_count=len(keep),omitted_uncommitted_event_count=discard,
  original_status_sha256=sha(run/'run_status.json'))
def copy_exact(src,dst):
 before=statkey(src);h=sha(src);dst.parent.mkdir(parents=True,exist_ok=True)
 with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b,8*1024**2);b.flush();os.fsync(b.fileno())
 assert statkey(src)==before and sha(dst)==h
 return h
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 assert os.name=='posix' and os.getuid()==1004 and ROOT.resolve()==ROOT
 dead_sources();no_run_process();assert not OUT.exists();items=[inspect(j) for j in JOBS]
 summary=[dict(host=j['host'],key=j['key'],original=str(x['run']),branch=str(x['dst']),checkpoint=x['meta'],
  committed_event_count=x['committed_event_count'],omitted_uncommitted_event_count=x['omitted_uncommitted_event_count'],
  checkpoint_full_SHA_checked=args.execute) for j,x in zip(JOBS,items)]
 if not args.execute:print(json.dumps(dict(inspection_only=True,writes=0,branches=summary)));return
 assert shutil.disk_usage(ROOT).free>=20*1024**3
 # Full source CP verification BEFORE any branch/output creation; no torch/load.
 for j,x in zip(JOBS,items):assert sha(x['cp'])==j['sha256'] and statkey(x['cp'])==x['cp_stat']
 dead_sources();no_run_process();OUT.mkdir(exist_ok=False);done=[]
 try:
  write(OUT/'INTENT.json',dict(unix=time.time(),script_sha256=sha(__file__),branches=summary,no_GPU_launch=True))
  for j,x in zip(JOBS,items):
   assert inspect(j)['cp_stat']==x['cp_stat'];dst=x['dst'];dst.mkdir(exist_ok=False);inventory={}
   for p in x['files']:
    rel=p.relative_to(x['run']);inventory[str(rel)]=copy_exact(p,dst/rel)
   target=dst/x['meta']['path'];target.parent.mkdir(parents=True,exist_ok=True)
   os.link(x['cp'],target,follow_symlinks=False);assert statkey(target)==statkey(x['cp'])==x['cp_stat']
   inventory[x['meta']['path']]=j['sha256']
   (dst/'logs').mkdir(exist_ok=True)
   with (dst/'logs/events.jsonl').open('xb') as f:f.write(x['events']);f.flush();os.fsync(f.fileno())
   inventory['logs/events.jsonl']=hashlib.sha256(x['events']).hexdigest()
   # This is an explicit IO recovery origin, NEVER a cooperative-pause receipt.
   write(dst/'run_status.json',dict(status='io_recovery_prepared',completed_stage=j['stage'],
    prior_failed_run=str(x['run']),checkpoint=x['meta'],requires_external_recovery_admission=True))
   proof=dict(schema='native-committed-CP-IO-recovery-v1',original_run=str(x['run']),branch=str(dst),
    source_sha256=x['source_sha256'],config_sha256=inventory['config.json'],checkpoint=x['meta'],
    original_status_sha256=x['original_status_sha256'],original_events_sha256=x['original_events_sha256'],
    canonical_events_sha256=inventory['logs/events.jsonl'],committed_event_count=x['committed_event_count'],
    omitted_uncommitted_event_count=x['omitted_uncommitted_event_count'],files_sha256=inventory,
    original_RUN_untouched=True,partial_model_not_retained=True,cooperative_pause=False,
    scientific_settings_changed=False,CP_SHA_verified=True,new_scores=0,new_models=0,
    resource_scope='Original resource log retained for native finish compatibility; multiple process scopes, never interpret cross-process before/after subtraction as single-process growth.')
   write(dst/'IO_RECOVERY_SOURCE.json',proof);done.append(dict(job=j,branch=str(dst),proof_sha256=sha(dst/'IO_RECOVERY_SOURCE.json')))
   write(OUT/(j['backbone']+'_'+j['method']+'_PREPARED.json'),done[-1])
  plan=dict(schema='native-IO-recovery-owner-plan-v1',prepared=done,original_owner='llmcl-two-backbone-native.service',
   existing_outer_entry=str(ROOT/'analysis/full_parameter_preflight_capacity_20261028.py'),
   baseline_entry=str(ROOT/'analysis/baseline_pct01_pct10_checkpoint_capacity_20261027.py'),
   queue_replacements=[dict(host=j['host'],key=j['key'],run_dir=str(x['dst'].relative_to(ROOT)),status='ready_resume',
    completed_stage=j['stage'],resume_origin='explicit_IO_recovery',IO_RECOVERY_SOURCE_sha256=sha(x['dst']/'IO_RECOVERY_SOURCE.json'),
    clear_stale_fields=['pid','capacity_identity','worker','wrapper','phase_exit','phase_status']) for j,x in zip(JOBS,items)],
   required_owner_adapter='One private resume_requested override only for the two bound recovery branches; all other runs use the original function. Original cap locks and idle/FullFT hook stay in force.',
   queue_written=False,owner_started=False,GPU_launches=0,
   note='Original unmodified allocator rejects non-paused ready_resume; do not forge pause/permit or run this plan without the narrow owner adapter. Old Qwen SMo fresh recovery remains a separate follow-up.')
  write(OUT/'PREPARED.json',plan);print(json.dumps(dict(prepared=True,path=str(OUT/'PREPARED.json'),sha256=sha(OUT/'PREPARED.json'),GPU_launches=0)))
 except BaseException:
  write(OUT/'FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),completed=done,no_automatic_retry=True));raise
if __name__=='__main__':main()
