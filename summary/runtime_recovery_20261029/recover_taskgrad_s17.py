"""Explicitly authorized S17 branch preparation/single GPU1 continuation.
Default: inspect metadata. --prepare copies committed prefix only, no model.
--supervise launches exactly run.py --resume after preparation; no retry,
preflight, scorer or new allocator. Original failed run is never modified.
Use on OLD after source review; this file is prepared on NEW, not auto-run.
"""
import argparse,fcntl,hashlib,json,math,os,shutil,signal,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/dev/shm/llmcl-task-gradient-formal-1084-20261028')
FRAME='exp/task_gradient_formal_20261028'
REL=FRAME+'/runs/onevision/pct01/TaskGradReuse/training_v1'
BRANCH=FRAME+'/runs/onevision/pct01/TaskGradReuse/training_recovery_S17_20261029'
CP_SHA='939b7081ad009f387cc142b9baffcbfb6ad1e5e031bc3b1d47463771e7c74ed4'
CP_BYTES=1928308132
BINDING_SHA='4313ef543860e0ef9919b24e1c6a29f5ee40caf49118add470587b9d101edfbb'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:
  json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
  if s[0]=='Z':return None
  return dict(pid=int(pid),startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
 except (OSError,ValueError,IndexError):return None

def inspect():
 old=ROOT/REL;new=ROOT/BRANCH;frame=ROOT/FRAME;report=ROOT/'summary/runtime_recovery_20261029'
 binding_path=ROOT/'summary/task_gradient_formal_20261028/BINDING.json'
 assert sha(binding_path)==BINDING_SHA;b=read(binding_path)
 assert b['sources']=={p.name:sha(p) for p in frame.glob('*.py')} and len(b['sources'])==29
 cfg=read(old/'config.json');assert cfg['execution_source_sha256']==b['sources']
 assert cfg['method']=='TaskGradReuse' and cfg['condition']=='pct01' and not cfg['preflight_only']
 assert cfg['checkpoint_all_stages'] and cfg['seed']==42
 failure=ROOT/'run_v1/FORMAL_FAILURE.json';slot_path=ROOT/'summary/task_gradient_formal_20261028/SLOT.json'
 f=read(failure);slot=read(slot_path)
 assert f['no_retry'] and f['binding_sha256']==BINDING_SHA
 assert slot['released'] and slot['status']=='failed_closed' and slot['binding_sha256']==BINDING_SHA
 assert slot['terminal_sha256']==sha(failure)
 for w in [f['identity']]+[x['identity'] for x in f['phases']]+([f['last_attempted_worker']] if f.get('last_attempted_worker') else []):
  assert identity(w['pid'])!=w,'Original model/controller still live'
 latest=read(old/'checkpoints/latest.json');meta=read(old/'checkpoints/stage_17.json')
 assert latest==meta and meta['stage']==17 and meta['path']=='checkpoints/stage_17.pt'
 assert meta['sha256']==CP_SHA and meta['bytes']==CP_BYTES and meta['full_architecture_and_runtime']
 cp=old/meta['path'];st=cp.stat();assert not cp.is_symlink() and st.st_size==CP_BYTES and st.st_uid==1084
 assert not (old/'checkpoints/stage_19.pt').exists()
 manifest=read(ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json')
 assert sha(ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json')==cfg['manifest_sha256']==b['manifest_sha256']
 start=read(ROOT/'run_v1/PHASE_training_STARTED.json');python=Path(start['command'][0]);assert python.is_absolute() and python.is_file()
 return old,new,frame,report,cfg,meta,manifest,python,dict(binding_sha256=BINDING_SHA,
  original_failure_sha256=sha(failure),original_slot_sha256=sha(slot_path),original_config_sha256=sha(old/'config.json'),
  checkpoint_sha256=CP_SHA,checkpoint_bytes=CP_BYTES,checkpoint_inode=st.st_ino,checkpoint_device=st.st_dev,
  checkpoint_mtime_ns=st.st_mtime_ns,checkpoint_payload_not_rehashed_by_inspect=True,
  native_loader_will_hash_complete_payload_before_restore=True)

def prepare():
 old,new,frame,report,cfg,meta,m,python,evidence=inspect()
 assert not new.exists() and shutil.disk_usage(ROOT).free>=4*1024**3
 assert sha(old/meta['path'])==CP_SHA  # One complete payload SHA during preparation; no model load.
 report.mkdir(parents=True,exist_ok=True)
 write(report/'PREPARE_INTENT.json',dict(source_sha256=sha(__file__),original_run=REL,branch=BRANCH,authorization='Explicit human recovery instruction',**evidence))
 new.mkdir(parents=True);copied={}
 def copy(rel,dest=None):
  src=old/rel;dst=new/(dest or rel)
  assert src.is_file() and not src.is_symlink()
  before=src.stat();h=sha(src);dst.parent.mkdir(parents=True,exist_ok=True)
  with src.open('rb') as a,dst.open('xb') as z:
   shutil.copyfileobj(a,z,8*1024**2);z.flush();os.fsync(z.fileno())
  assert sha(dst)==h and (before.st_size,before.st_mtime_ns)==(src.stat().st_size,src.stat().st_mtime_ns)
  copied[str(dst.relative_to(new))]=dict(source=str(src),sha256=h,bytes=before.st_size)
 copy('config.json')
 for p in sorted((old/'runner_source').rglob('*')):
  if p.is_file():copy(str(p.relative_to(old)))
 for t in range(18):
  wanted=range(1,20) if t==0 else (t,t+1)
  for task in wanted:
   for suffix in ('jsonl','scores.json','datasets.json'):
    copy(f'evaluations/stage_{t:02d}/T{task:02d}_test_predictions.{suffix}')
  if t:
   stage=read(old/f'stages/stage_{t:02d}.json');assert stage['complete'] and stage['weights_saved']
   for rel in (f'stages/stage_{t:02d}.json',f'validity/stage_{t:02d}.json',f'checkpoints/stage_{t:02d}.json',f'method_state/task_gradient_stage_{t:02d}.json'):copy(rel)
 # Fresh target state_root has no conflicting stage18 direction or partial predictions.
 assert not (new/'method_state/task_gradient_stage_18.json').exists()
 log=old/'logs/events.jsonl';source_raw=log.read_bytes();selected=[];counts={}
 for raw in source_raw.splitlines(keepends=True):
  event=json.loads(raw);assert event['event']=='train_update'
  if event['stage']<=17:
   counts.setdefault(event['stage'],[]).append(event['step']);selected.append(raw)
 for t in range(1,18):assert counts[t]==list(range(1,math.ceil(m['tasks'][t-1]['files']['train']['count']/16)+1))
 (new/'logs').mkdir();(new/'logs/events.jsonl').write_bytes(b''.join(selected))
 assert log.read_bytes()==source_raw
 copied['logs/events.jsonl']=dict(source=str(log),original_sha256=hashlib.sha256(source_raw).hexdigest(),
  sha256=sha(new/'logs/events.jsonl'),filtered_to_committed_stages=list(range(1,18)),updates=sum(map(len,counts.values())),uncommitted_events_retained_in_original=True)
 (new/'checkpoints').mkdir(exist_ok=True)
 src=old/meta['path'];dst=new/meta['path'];os.link(src,dst)
 a=src.stat();z=dst.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,CP_BYTES,evidence['checkpoint_mtime_ns'])
 copy('checkpoints/latest.json')
 copy('resources.jsonl')
 # Restore only from full S17 graph/RNG, never from changed-parameter snapshots.
 write(new/'run_status.json',dict(status='authorized_crash_recovery_checkpoint',completed_stage=17,
  source_run=REL,checkpoint=meta,not_a_cooperative_pause=True))
 command=[str(python),'-u',str(frame/'run.py'),'--method','TaskGradReuse','--condition','pct01','--run-dir',str(new),'--resume']
 receipt=dict(prepared=True,source_sha256=sha(__file__),original_run=REL,branch=BRANCH,
  copied_files=copied,checkpoint_hardlink=True,checkpoint_identity=evidence,checkpoint_full_sha_verified=True,
  resource_scope='resources.jsonl combines original and continuation PIDs; never subtract endpoints as one-process scope or hide failed-stage costs',
  excluded_uncommitted_stages=[18,19],no_original_files_removed_or_rewritten=True,
  prefix_cells=53,starting_completed_stage=17,next_stage=18,original_scientific_sources=cfg['execution_source_sha256'],
  command=command,environment=dict(LLMCL_ROOT=str(ROOT),LLMCL_HOST='old2080',LLMCL_BACKBONE='onevision',CUDA_VISIBLE_DEVICES='1'),
  main_updates_replayed_from_stage18=True,old_failed_S18_cost_not_erased=True,model_started=False,
  verification_plan='Independent branch terminal/source validator, unchanged original engine.verify(BRANCH,old2080), 74 cells/80316 originalF1; keep method_records math checks but point to branch. Never call old frozen formal terminal wrapper as if failure had not occurred.')
 write(report/'PREPARED.json',receipt);print(json.dumps(receipt))

def supervise():
 old,new,frame,report,cfg,meta,m,python,evidence=inspect();prepared=read(report/'PREPARED.json')
 assert prepared['branch']==BRANCH and prepared['checkpoint_identity']==evidence and prepared['checkpoint_full_sha_verified']
 assert sha(new/'config.json')==evidence['original_config_sha256'] and read(new/'checkpoints/latest.json')==meta
 for rel,record in prepared['copied_files'].items():assert sha(new/rel)==record['sha256']
 assert not any((report/n).exists() for n in ('STARTING.json','STARTED.json','EXIT.json','COMPLETE.json','FAILURE.json'))
 lock=(report/'GPU1_RECOVERY.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Parent coordinates GPU1 exclusion in the original baseline owner; no allocator here.
 assert not subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
 v=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
 assert int(v[0])<500 and int(v[1])<10
 command=prepared['command'];env=dict(os.environ,**prepared['environment'],PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
 owner=identity(os.getpid());assert owner and owner['pid']==owner['pgid'];child=None;who=None
 write(report/'STARTING.json',dict(unix=time.time(),supervisor=owner,GPU=1,source_sha256=sha(__file__),prepared_sha256=sha(report/'PREPARED.json'),command=command,no_retry=True))
 def interrupted(signum,frame):raise InterruptedError(str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  with (report/'supervisor.log').open('x') as log:
   child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   who=identity(child.pid);assert who and who['pid']==who['pgid'] and who['uid']==1084
   write(report/'STARTED.json',dict(unix=time.time(),supervisor=owner,worker=who,GPU=1,source_sha256=sha(__file__),prepared_sha256=sha(report/'PREPARED.json'),command=command,no_retry=True,no_new_preflight=True))
   # Original formal budget is finite stages/data/updates, not an old FIT timer.
   code=child.wait()
  ex=dict(unix=time.time(),worker=who,supervisor=owner,GPU=1,returncode=code,identity_absent=identity(child.pid)!=who)
  write(report/'EXIT.json',ex)
  assert code==0 and ex['identity_absent'],'Continuation worker failed'
  status=read(new/'run_status.json');latest=read(new/'checkpoints/latest.json')
  assert status['status']=='completed' and status['completed_stage']==19 and latest['stage']==19
  write(report/'COMPLETE.json',dict(ex,complete=True,original_failure_preserved=True,prepared_sha256=sha(report/'PREPARED.json'),
   final_checkpoint=latest,branch=BRANCH,new_scores=0,requires_independent_branch_verifier=True))
 except BaseException:
  error=traceback.format_exc()
  if child is not None:
   if child.poll() is None and who is not None and identity(child.pid)==who:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:
     if identity(child.pid)==who:os.killpg(child.pid,signal.SIGKILL)
     child.wait(timeout=15)
   if not (report/'EXIT.json').exists():
    write(report/'EXIT.json',dict(unix=time.time(),worker=who,supervisor=owner,GPU=1,returncode=child.poll(),identity_absent=who is not None and identity(child.pid)!=who))
  write(report/'FAILURE.json',dict(unix=time.time(),worker=who,supervisor=owner,GPU=1,traceback=error,
   scientific_quality_incomplete=True,no_automatic_retry=True,original_failure_preserved=True))
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--prepare',action='store_true');g.add_argument('--supervise',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.supervise:supervise()
 else:
  old,new,frame,report,*_,evidence=inspect();print(json.dumps(dict(inspect_only=True,branch_exists=new.exists(),evidence=evidence,model_started=False)))
