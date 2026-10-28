"""Independent v2 NEW idle offer after the preserved model-before-load v1 failure.
LLaVA pct01 FullFT preflight -> independent reload only.

--fullft-admission ABS plus ALL unchanged checkpoint-capacity argv.
--once remains metadata-only. --fullft-request is an isolated child supervisor,
not an allocator. No formal training, retry, GPU0, baseline signals or queue edits
outside the inherited capacity owner. New admission is prepared by the caller:
schema, root, wrapper_sha256, entry_sha256, policy_sha256, python, sources(3),
native_sources, preflight_seconds, reload_seconds. No admission is created here.
"""
from pathlib import Path
import argparse
import ast
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
ENTRY=ROOT/'analysis/baseline_pct01_pct10_checkpoint_capacity_20261027.py'
ENTRY_SHA='a90f28b4f671851cc922a2490a2b5f94b453b9129fdc4dfc96c24d1eed5c8ab5'
HERE=ROOT/'exp/full_parameter_seq_20261028/v2'
NATIVE=ROOT/'exp/two_backbone_native_cl_20261017_v2'
POLICY=ROOT/'exp/two_backbone_native_checkpoint_20261027/execution_policy.json'
REPORT=ROOT/'summary/full_parameter_seq_20261028/preflight_v2_once'
RUN=HERE/'runs/onevision/pct01/preflight_once_20261028'
SOURCES={'core.py':'67ec0cc0ba660201b24f842556f8291afa4153d1fee17ee1ccb86f7255a7a2d4',
 'run.py':'9468dc81582eebf2bd606f1d2514b8e8f804f3756f0c35b7c98585fdc01b9e67',
 'reload_probe.py':'779aa9d0c01237774c41d69487f84cb9a7e7b2b41fc64d5f12af67edaab8b64c'}
MANIFEST_SHA='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
PAUSE_REL='exp/two_backbone_native_checkpoint_20261027/run.py'
PAUSE_SHA='ea347e655c8f3a20d07c86568b351ab1615868bcf9b88d209756c45d1cef1280'
PAUSE_AST='3d34b44af1b0d9cf10e0e731cd971ad8e9c8829366a1b422935e55c67093d8b3'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v,exclusive=False):
 p=Path(p)
 if exclusive:
  with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 else:
  q=p.with_suffix(p.suffix+'.tmp')
  with q.open('w',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  q.replace(p)

def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
  if s[0]=='Z':return None
  return dict(pid=int(pid),startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
 except (OSError,ValueError,IndexError):return None

def gone(w):return identity(w['pid'])!=w

def validate(path):
 b=read(path)
 assert b['schema']=='fullft-preflight-once-v2' and Path(b['root'])==ROOT
 assert b['wrapper_sha256']==sha(__file__) and b['entry_sha256']==sha(ENTRY)==ENTRY_SHA
 assert b['sources']==SOURCES=={p.name:sha(p) for p in HERE.glob('*.py')}
 assert b['native_sources']=={p.name:sha(p) for p in NATIVE.glob('*.py')}
 helper=b['pause_helper'];assert Path(helper['path'])==ROOT/PAUSE_REL and helper['sha256']==sha(ROOT/PAUSE_REL)==PAUSE_SHA
 nodes=[n for n in ast.parse((ROOT/PAUSE_REL).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='pause_at_boundary']
 assert len(nodes)==1 and hashlib.sha256(ast.dump(nodes[0],include_attributes=False).encode()).hexdigest()==helper['function_AST_sha256']==PAUSE_AST
 prior=b['prior_v1_closed'];slot=Path(prior['slot_path']);terminal=Path(prior['terminal_path'])
 assert slot==ROOT/'summary/full_parameter_seq_20261028/preflight_once/SLOT.json'
 assert sha(slot)==prior['slot_sha256'] and sha(terminal)==prior['terminal_sha256']
 closed=read(slot);assert closed['released'] and closed['status']=='failed_closed'
 assert closed['terminal_path']==str(terminal) and closed['terminal_sha256']==prior['terminal_sha256']
 assert sha(POLICY)==b['policy_sha256']
 policy=read(POLICY);host=policy['hosts']['new3090_onevision'];data=policy['conditions']['pct01']
 assert host['backbone']=='onevision' and host['reserved_gpus']==[0]
 assert set((1,3))<=set(host['conditions']['pct01'])
 manifest=ROOT/data['manifest'];assert sha(manifest)==data['manifest_sha256']==MANIFEST_SHA
 assert data['train_fraction']==.01 and data['test_fraction']==1
 py=Path(b['python']);assert py.is_absolute() and py.is_file()
 assert py==ROOT/'.vision-env/bin/python'
 assert all(type(b[k]) is int and b[k]>0 for k in ('preflight_seconds','reload_seconds'))
 return b,manifest,max(16000,host['minimum_free_mib'])

def stop_own(child,who):
 if child.poll() is None and identity(child.pid)==who:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:
   if identity(child.pid)==who:os.killpg(child.pid,signal.SIGKILL)
   child.wait(timeout=15)
 assert gone(who)

def supervise(request):
 q=read(request);admission=Path(q['admission']);assert sha(admission)==q['admission_sha256']
 b,manifest,_=validate(admission);gpu=q['GPU'];assert gpu in (1,3)
 assert q['run']==str(RUN) and q['wrapper_sha256']==sha(__file__)
 owner=identity(os.getpid());assert owner and owner['pgid']==owner['pid']
 phases=[];child=None;active=None;last=None
 REPORT.mkdir(parents=True,exist_ok=True)
 write(REPORT/'STARTED.json',dict(identity=owner,GPU=gpu,request_sha256=sha(request),
  admission_sha256=sha(admission),unix=time.time(),no_retry=True,formal_training=False),True)
 def interrupted(signum,frame):raise InterruptedError(str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 env=dict(os.environ,LLMCL_ROOT=str(ROOT),LLMCL_HOST='new3090_onevision',
  LLMCL_BACKBONE='onevision',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1')
 try:
  assert not RUN.exists()
  for phase in ('preflight','reload'):
   validate(admission);assert sha(admission)==q['admission_sha256']
   need=20*1024**3+(4*1024**3 if phase=='preflight' else 0)
   assert shutil.disk_usage(ROOT).free>=need,'Preserve 20GiB plus preflight checkpoint allowance'
   if phase=='preflight':
    command=[b['python'],'-u',str(HERE/'run.py'),'--condition','pct01','--backbone','onevision',
     '--manifest',str(manifest),'--manifest-sha256',MANIFEST_SHA,'--run-dir',str(RUN),'--preflight-only']
   else:
    config=read(RUN/'config.json');meta=read(RUN/'checkpoints/latest.json');proof=read(RUN/'gpu_preflight.json')
    expected={str((NATIVE/n).relative_to(ROOT)):h for n,h in b['native_sources'].items()}
    expected.update({str((HERE/n).relative_to(ROOT)):h for n,h in SOURCES.items()})
    expected[PAUSE_REL]=PAUSE_SHA
    assert config['method']=='FullFT_Seq' and config['preflight_only'] and config['condition']=='pct01'
    assert config['backbone']=='onevision' and config['manifest_sha256']==MANIFEST_SHA
    assert config['execution_source_sha256']==expected and proof['passed'] and proof['not_formal']
    assert len(proof['cases'])==19 and [x['task'] for x in proof['cases']]==list(range(1,20))
    assert proof['checkpoint']==meta and meta['full_architecture_and_runtime'] and meta['stage']==1
    assert (RUN/meta['path']).stat().st_size==meta['bytes']
    command=[b['python'],'-u',str(HERE/'reload_probe.py'),'--run-dir',str(RUN),
     '--config-sha256',sha(RUN/'config.json'),'--checkpoint-sha256',meta['sha256'],
     '--checkpoint-metadata-sha256',sha(RUN/'checkpoints/latest.json')]
   started=time.monotonic()
   with (REPORT/(phase+'.log')).open('a') as log:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active=last=identity(child.pid);assert active and active['uid']==os.getuid() and active['pid']==active['pgid']
    write(REPORT/(phase+'_STARTED.json'),dict(worker=active,GPU=gpu,command=command,
     admission_sha256=sha(admission),unix=time.time(),limit=b[phase+'_seconds']),True)
    timeout=False
    try:code=child.wait(timeout=b[phase+'_seconds'])
    except subprocess.TimeoutExpired:timeout=True;stop_own(child,active);code=child.returncode
   item=dict(phase=phase,worker=active,GPU=gpu,returncode=code,timeout=timeout,
    identity_absent=gone(active),seconds=time.monotonic()-started)
   write(REPORT/(phase+'_EXIT.json'),item,True);phases.append(item)
   assert code==0 and not timeout and item['identity_absent'],phase+' failed; no retry'
   active=child=None
  proof=read(RUN/'checkpoint_reload_proof.json')
  assert proof['passed'] and proof['fresh_process'] and proof['same_used_TRAIN_QA']
  assert proof['TEST_QA']==proof['F1_scores']==proof['optimizer_updates']==0 and proof['backwards']==1
  assert proof['checkpoint_retained'] and proof['checkpoint_sha256']==read(RUN/'checkpoints/latest.json')['sha256']
  write(REPORT/'COMPLETE.json',dict(identity=owner,GPU=gpu,phases=phases,last_attempted_worker=last,
   admission_sha256=sha(admission),wrapper_sha256=sha(__file__),reload_proof_sha256=sha(RUN/'checkpoint_reload_proof.json'),
   preflight_QA=19,reload_QA=1,formal_training=False,no_retry=True,checkpoint_retained=True),True)
 except BaseException:
  error=traceback.format_exc()
  if child is not None and active is not None:
   try:stop_own(child,active)
   except BaseException:error+='\nOwn cleanup: '+traceback.format_exc()
  write(REPORT/'FAILURE.json',dict(identity=owner,GPU=gpu,phases=phases,last_attempted_worker=last,
   admission_sha256=sha(admission),wrapper_sha256=sha(__file__),traceback=error,no_retry=True,formal_training=False),True)
  raise

class Slot:
 def __init__(self,admission,inspection):
  self.admission=admission;self.b,self.manifest,self.minimum=validate(admission)
  self.inspection=inspection;self.path=REPORT/'SLOT.json';self.process=None
 def reconcile(self):
  if not self.path.exists():return
  v=read(self.path)
  assert v['admission_sha256']==sha(self.admission) and v['wrapper_sha256']==sha(__file__)
  if v.get('released'):return
  wrapper=v.get('wrapper')
  if wrapper and not gone(wrapper):return
  ends=[p for p in (REPORT/'COMPLETE.json',REPORT/'FAILURE.json') if p.exists()]
  if len(ends)!=1:return  # missing proof stays held, never reattempted
  terminal=read(ends[0]);assert terminal['identity']==wrapper and terminal['GPU']==v['GPU']
  assert terminal['admission_sha256']==v['admission_sha256'] and terminal['wrapper_sha256']==sha(__file__)
  workers=[x['worker'] for x in terminal['phases']]
  if terminal.get('last_attempted_worker'):workers.append(terminal['last_attempted_worker'])
  if any(not gone(w) for w in workers):return
  pids={int(w['pid']) for w in workers+[wrapper]}
  try:
   text=subprocess.check_output(['nvidia-smi',f"--id={v['GPU']}",'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
  except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as exc:
   # A failed observation is not a terminal model event or a release permit.
   v['last_NVML_release_observation_failure']=dict(unix=time.time(),exception=type(exc).__name__,message=str(exc),held_preserved=True)
   write(self.path,v)
   return
  present={int(s.strip()) for s in text.splitlines() if s.strip()}
  if pids & present:return
  code=self.process.poll() if self.process else None
  if self.process is not None and code is None:return
  if ends[0].name=='COMPLETE.json':
   assert all(x['returncode']==0 and not x['timeout'] and x['identity_absent'] for x in terminal['phases'])
   if code is not None:assert code==0
  v.update(status='complete' if ends[0].name=='COMPLETE.json' else 'failed_closed',released=True,
   terminal_path=str(ends[0]),terminal_sha256=sha(ends[0]),release_unix=time.time(),
   identities_absent=True,own_NVML_PIDs_absent=True,wrapper_returncode_if_parent=code)
  write(self.path,v)
 def offer(self,gpu,minimum,original_idle,held,python):
  assert not self.inspection, 'Metadata-only inspection cannot offer a GPU'
  if gpu not in (1,3):return None
  if gpu in held:return False
  if self.path.exists():
   v=read(self.path)
   return False if not v.get('released') and v['GPU']==gpu else None
  if not original_idle(gpu,max(minimum,self.minimum)):return False
  if shutil.disk_usage(ROOT).free<24*1024**3:return None
  assert Path(python)==Path(self.b['python'])
  validate(self.admission);assert not RUN.exists()
  REPORT.mkdir(parents=True,exist_ok=True)
  value=dict(status='claimed',GPU=gpu,wrapper=None,admission_sha256=sha(self.admission),
   wrapper_sha256=sha(__file__),released=False,no_retry=True,unix=time.time())
  write(self.path,value,True)
  request=REPORT/'REQUEST.json'
  write(request,dict(admission=str(self.admission),admission_sha256=sha(self.admission),GPU=gpu,
   run=str(RUN),wrapper_sha256=sha(__file__)),True)
  try:
   with (REPORT/'supervisor.log').open('a') as log:
    self.process=subprocess.Popen([self.b['python'],'-u',str(Path(__file__).resolve()),'--fullft-request',str(request)],
     cwd=ROOT,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   who=identity(self.process.pid);assert who and who['uid']==os.getuid() and who['pid']==who['pgid']
   value.update(status='running',wrapper=who);write(self.path,value)
  except BaseException:
   value.update(status='orphaned_hold',error=traceback.format_exc(),attempted_wrapper_pid=self.process.pid if self.process else None)
   write(self.path,value);raise
  return False

def main():
 p=argparse.ArgumentParser(add_help=False)
 p.add_argument('--fullft-admission',type=Path);p.add_argument('--fullft-request',type=Path)
 args,remaining=p.parse_known_args()
 if args.fullft_request is not None:
  assert not remaining and args.fullft_admission is None
  return supervise(args.fullft_request)
 assert args.fullft_admission and args.fullft_admission.is_absolute()
 assert sha(ENTRY)==ENTRY_SHA
 spec=importlib.util.spec_from_file_location('same_checkpoint_capacity_owner_fullft',ENTRY)
 entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
 factory=entry.addition_class
 slot=Slot(args.fullft_admission,'--once' in remaining)
 def wrapped(*a,**kw):
  base=factory(*a,**kw)
  class Addition(base):
   def reconcile(self):
    super().reconcile();slot.reconcile()
   def offer(self,gpu,minimum,original_idle):
    result=slot.offer(gpu,minimum,original_idle,super().held(),self.pipes['onevision'].PYTHON)
    return super().offer(gpu,minimum,original_idle) if result is None else result
  return Addition
 entry.addition_class=wrapped
 sys.argv=[str(ENTRY),*remaining]
 entry.main()

if __name__=='__main__':main()
