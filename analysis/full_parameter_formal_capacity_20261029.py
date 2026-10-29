"""Draft one-shot FullFT formal offer inside the unchanged single IO-recovery owner.
No action occurs on import. No second allocator, baseline signal, preflight retry,
scientific source edits or formal timeout. Activation is a separate reviewed step.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path('/data1/liyapeng/program/LLMCL')
OWNER=ROOT/'summary/storage_recovery_second_20261029/owner_recovery.py'
OWNER_SHA='12c4856158c5a8c8a00b3042a82f5cb158324f303b6c12999d05012eacb8cc53'
FULL=ROOT/'analysis/full_parameter_preflight_capacity_v2_20261028.py'
FULL_SHA='03fe7f0028ab4b8df479bc997aab7011f413eee6e47c4023fed26dd4442c9d56'
HERE=ROOT/'exp/full_parameter_seq_20261028/v2'
NATIVE=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/full_parameter_seq_20261028/formal_pct01_once_20261029'
RUN=HERE/'runs/onevision/pct01/formal_once_20261029'
PREFLIGHT=ROOT/'summary/full_parameter_seq_20261028/preflight_v2_once'
TERMINAL_SHA='9d4be1fd21ec693c436cf1864132c1b514dffc07f70846fc7dd4706099b364c3'
MANIFEST_SHA='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,value,exclusive=False):
 p=Path(p)
 if exclusive:
  with p.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 else:
  q=p.with_suffix(p.suffix+'.tmp')
  with q.open('w',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  q.replace(p)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return None if s[0]=='Z' else dict(pid=int(pid),startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
 except (OSError,ValueError,IndexError):return None

def validate(path):
 b=read(path)
 assert b['schema']=='fullft-formal-pct01-once-v1' and Path(b['root'])==ROOT
 assert b['wrapper_sha256']==sha(__file__) and sha(OWNER)==b['owner_sha256']==OWNER_SHA and sha(FULL)==FULL_SHA
 assert b['sources']=={p.name:sha(p) for p in HERE.glob('*.py')} and b['native_sources']=={p.name:sha(p) for p in NATIVE.glob('*.py')}
 assert b['condition']=='pct01' and b['backbone']=='onevision' and b['fresh'] and b['no_retry'] and b['formal_timeout_seconds'] is None
 assert Path(b['run'])==RUN and Path(b['report'])==REPORT and set(b['allowed_GPUs'])=={1,3}
 assert sha(b['protocol_path'])==b['protocol_sha256']
 assert sha(ROOT/'exp/two_backbone_native_checkpoint_20261027/execution_policy.json')==b['policy_sha256']
 slot=read(PREFLIGHT/'SLOT.json');assert sha(PREFLIGHT/'COMPLETE.json')==TERMINAL_SHA==slot['terminal_sha256']
 assert slot['released'] and slot['status']=='complete' and Path(slot['terminal_path'])==PREFLIGHT/'COMPLETE.json'
 prior=read(PREFLIGHT/'COMPLETE.json');assert prior['formal_training'] is False and all(v['returncode']==0 and v['identity_absent'] and not v['timeout'] for v in prior['phases'])
 assert all(identity(v['worker']['pid'])!=v['worker'] for v in prior['phases']) and identity(prior['identity']['pid'])!=prior['identity']
 manifest=Path(b['manifest']);assert manifest.is_absolute() and sha(manifest)==b['manifest_sha256']==MANIFEST_SHA
 data=read(manifest);assert sum(t['files']['train']['count'] for t in data['tasks'])==1313 and sum(t['files']['test']['count'] for t in data['tasks'])==21454
 assert len(data['tasks'])==19
 assert Path(b['python'])==ROOT/'.vision-env/bin/python' and Path(b['python']).is_file()
 assert type(b['minimum_free_mib']) is int and b['minimum_free_mib']>=16000 and type(b['minimum_disk_bytes']) is int and b['minimum_disk_bytes']>=24*1024**3
 return b

def supervise(request):
 q=read(request);admission=Path(q['admission']);assert sha(admission)==q['admission_sha256'];b=validate(admission)
 gpu=q['GPU'];assert gpu in (1,3) and q['run']==str(RUN) and q['wrapper_sha256']==sha(__file__)
 who=identity(os.getpid());assert who and who['uid']==os.getuid() and who['pid']==who['pgid'];child=None;worker=None
 write(REPORT/'STARTED.json',dict(identity=who,GPU=gpu,admission_sha256=sha(admission),unix=time.time(),formal_training=True,fresh=True,no_retry=True,formal_timeout_seconds=None),True)
 def interrupted(signum,frame):raise InterruptedError(str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  assert not RUN.exists() and shutil.disk_usage(ROOT).free>=b['minimum_disk_bytes']
  command=[b['python'],'-u',str(HERE/'run.py'),'--condition','pct01','--backbone','onevision','--manifest',b['manifest'],'--manifest-sha256',MANIFEST_SHA,'--run-dir',str(RUN)]
  env=dict(os.environ,LLMCL_ROOT=str(ROOT),LLMCL_HOST='new3090_onevision',LLMCL_BACKBONE='onevision',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1')
  with (REPORT/'worker.log').open('x') as log:
   child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   worker=identity(child.pid);assert worker and worker['uid']==os.getuid() and worker['pid']==worker['pgid']
   write(REPORT/'WORKER_STARTED.json',dict(identity=worker,GPU=gpu,command=command,unix=time.time(),formal_timeout_seconds=None),True)
   code=child.wait() # No preflight deadline is transferred to the formal job.
  assert identity(worker['pid'])!=worker
  write(REPORT/'EXIT.json',dict(identity=worker,GPU=gpu,returncode=code,identity_absent=True,unix=time.time()),True)
  assert code==0,'Formal failure is preserved and never automatically retried'
  status=read(RUN/'run_status.json');config=read(RUN/'config.json');meta=read(RUN/'checkpoints/latest.json')
  assert config['method']=='FullFT_Seq' and config['condition']=='pct01' and config['backbone']=='onevision' and not config['preflight_only']
  expected={str((NATIVE/name).relative_to(ROOT)):h for name,h in b['native_sources'].items()}
  expected.update({str((HERE/name).relative_to(ROOT)):h for name,h in b['sources'].items()})
  expected['exp/two_backbone_native_checkpoint_20261027/run.py']='ea347e655c8f3a20d07c86568b351ab1615868bcf9b88d209756c45d1cef1280'
  assert config['execution_source_sha256']==expected and config['manifest_sha256']==MANIFEST_SHA and config['seed']==42
  assert meta['full_architecture_and_runtime'] and meta['method']=='FullFT_Seq' and meta['manifest_sha256']==MANIFEST_SHA
  cp=(RUN/meta['path']).resolve();assert cp.is_relative_to((RUN/'checkpoints').resolve()) and cp.stat().st_size==meta['bytes']
  paused=status['status']=='paused_checkpoint'
  if paused:
   receipt=RUN/status['pause_receipt'];assert sha(receipt)==status['pause_receipt_sha256'];value=read(receipt)
   assert value['completed_stage']==meta['stage']==status['completed_stage'] and value['checkpoint']==meta and value['full_stage_complete']
   assert value['checkpoint_metadata_sha256']==sha(RUN/'checkpoints/latest.json') and value['config_sha256']==sha(RUN/'config.json')
  else:
   assert status['status']=='completed' and status['completed_stage']==meta['stage']==19
  if meta['stage']:
   stage=read(RUN/'stages'/('stage_%02d.json'%meta['stage']));assert stage['complete'] and stage['weights_saved'] and stage['checkpoint']==meta
  write(REPORT/'COMPLETE.json',dict(identity=who,worker=worker,GPU=gpu,admission_sha256=sha(admission),wrapper_sha256=sha(__file__),terminal_kind='paused_checkpoint' if paused else 'completed_S19_unverified',checkpoint=meta,config_sha256=sha(RUN/'config.json'),formal_training=True,no_retry=True,unix=time.time()),True)
 except BaseException:
  error=traceback.format_exc()
  if child is not None and child.poll() is None and worker and identity(worker['pid'])==worker:
   os.killpg(worker['pid'],signal.SIGTERM)
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:
    if identity(worker['pid'])==worker:os.killpg(worker['pid'],signal.SIGKILL)
    child.wait(timeout=30)
  write(REPORT/'FAILURE.json',dict(identity=who,worker=worker,GPU=gpu,admission_sha256=sha(admission),wrapper_sha256=sha(__file__),traceback=error,attempted_worker_pid=child.pid if child else None,no_retry=True,formal_training=True,unix=time.time()),True)
  raise

class FormalSlot:
 def __init__(self,admission,inspection):
  self.admission=Path(admission);self.b=validate(self.admission);self.inspection=inspection;self.path=REPORT/'SLOT.json';self.process=None
 def reconcile(self):
  if not self.path.exists():return
  value=read(self.path);assert value['admission_sha256']==sha(self.admission) and value['wrapper_sha256']==sha(__file__)
  if value.get('released'):return
  who=value.get('wrapper')
  if not who or identity(who['pid'])==who:return
  ends=[p for p in [REPORT/'COMPLETE.json',REPORT/'FAILURE.json'] if p.exists()]
  if len(ends)!=1:return
  terminal=read(ends[0]);assert terminal['identity']==who and terminal['GPU']==value['GPU'] and terminal['admission_sha256']==value['admission_sha256']
  worker=terminal.get('worker')
  unknown=terminal.get('attempted_worker_pid') if not worker else None
  if unknown and (Path('/proc')/str(unknown)).exists():return
  if worker and identity(worker['pid'])==worker:return
  try:
   text=subprocess.check_output(['nvidia-smi',f"--id={value['GPU']}",'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
  except (OSError,subprocess.SubprocessError):return # Observation failure preserves the slot.
  pids={who['pid']}|({worker['pid']} if worker else set())|({unknown} if unknown else set())
  if pids & {int(v.strip()) for v in text.splitlines() if v.strip()}:return
  code=self.process.poll() if self.process else None
  if self.process and code is None:return
  if ends[0].name=='COMPLETE.json' and code is not None:assert code==0
  value.update(status=('paused_checkpoint' if terminal.get('terminal_kind')=='paused_checkpoint' else 'completed_unverified') if ends[0].name=='COMPLETE.json' else 'failed_closed',released=True,terminal_path=str(ends[0]),terminal_sha256=sha(ends[0]),identities_absent=True,own_NVML_PIDs_absent=True,release_unix=time.time())
  write(self.path,value)
 def offer(self,gpu,minimum,original_idle,held,python):
  assert not self.inspection
  if gpu not in (1,3):return None
  if gpu in held:return False
  if self.path.exists():
   value=read(self.path);return False if not value.get('released') and value['GPU']==gpu else None
  if not original_idle(gpu,max(minimum,self.b['minimum_free_mib'])):return False
  if shutil.disk_usage(ROOT).free<self.b['minimum_disk_bytes']:return None
  assert Path(python)==Path(self.b['python']);validate(self.admission);assert not RUN.exists()
  REPORT.mkdir(parents=True,exist_ok=True);value=dict(status='claimed',GPU=gpu,wrapper=None,admission_sha256=sha(self.admission),wrapper_sha256=sha(__file__),released=False,no_retry=True,formal_training=True,unix=time.time())
  write(self.path,value,True);request=REPORT/'REQUEST.json';write(request,dict(admission=str(self.admission),admission_sha256=sha(self.admission),GPU=gpu,run=str(RUN),wrapper_sha256=sha(__file__)),True)
  try:
   with (REPORT/'supervisor.log').open('x') as log:self.process=subprocess.Popen([self.b['python'],'-u',str(Path(__file__).resolve()),'--formal-request',str(request)],cwd=ROOT,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   who=identity(self.process.pid);assert who and who['uid']==os.getuid() and who['pid']==who['pgid'];value.update(status='running',wrapper=who);write(self.path,value)
  except BaseException:value.update(status='orphaned_hold',traceback=traceback.format_exc());write(self.path,value);raise
  return False

def main():
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--formal-admission',type=Path);parser.add_argument('--formal-request',type=Path);parser.add_argument('--formal-inspect',action='store_true');args,remaining=parser.parse_known_args()
 if args.formal_request:
  assert not remaining and args.formal_admission is None and not args.formal_inspect;return supervise(args.formal_request)
 assert args.formal_admission and args.formal_admission.is_absolute();validate(args.formal_admission)
 if args.formal_inspect:assert not remaining;print(json.dumps(dict(validated=True,model_loaded=False,owner_started=False)));return
 owner=load('unchanged_IO_owner_with_formal_offer',OWNER);original_load=owner.load
 def intercepted(name,path):
  value=original_load(name,path)
  if Path(path)==FULL:
   original_slot=value.Slot
   class CombinedSlot:
    def __init__(self,admission,inspection):self.old=original_slot(admission,inspection);self.formal=FormalSlot(args.formal_admission,inspection)
    def reconcile(self):self.old.reconcile();self.formal.reconcile()
    def offer(self,*a,**kw):
     result=self.old.offer(*a,**kw)
     return self.formal.offer(*a,**kw) if result is None else result
   value.Slot=CombinedSlot
  return value
 owner.load=intercepted;sys.argv=[str(OWNER),*remaining];owner.main()

if __name__=='__main__':main()
