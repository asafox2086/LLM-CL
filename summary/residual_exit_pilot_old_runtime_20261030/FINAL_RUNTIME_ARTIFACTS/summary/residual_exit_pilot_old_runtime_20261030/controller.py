"""One exclusive GPU2 lease, two serial original teachers, never retry.
Draft control entry: no action on import; default inspect never starts a model.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,traceback
R=Path('/data2/liyapeng_grp/program/LLMCL');E=R/'exp/residual_exit_pilot_old_runtime_20261030';D=R/'summary/residual_exit_pilot_old_runtime_20261030'
O=D/'CONTROLLER_V1';METHODS=('PASs_MoE','CL_MoE')
SOURCES={'run.py':'2cc4a902a875da64e9a65a038966010e10563cee7e6a55d9d8113c38e9131306',
 'probe.py':'d4c756a932a7c15dccad31d2296d9722df80716b5b00935b170764f00b321063',
 'ridge.py':'b77350ac0df5557ac9496332fb65c79ca99bdc9dc549f8c57a0254abf49791bb'}
RT=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_')
RUN=RT/'exp/two_backbone_pct10_native_checkpoint_20261027/runs/onevision/pct10/OSFT/recovery_quota_fourth_S18_20261030'
UUID='GPU-25d3cb92-fa97-a722-f9cd-c012bad046a8'
OLD_WORKER={'pid':1342314,'startticks':'802265921','uid':1084}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,j):
 with p.open('x') as f:json.dump(j,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return None if s[0]=='Z' else dict(pid=pid,startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
 except FileNotFoundError:return None
def idle():
 assert subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==UUID
 assert not subprocess.check_output(['nvidia-smi','--id=2','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def terminal_guard():
 proof=read(D/'OSFT_TERMINAL_PROOF.json');assert proof['schema']=='original-old-osft-S19-exit-v1' and proof['verified'] is True
 for item in proof['source_records']:
  assert sha(item['path'])==item['sha256']
 assert proof['worker']==OLD_WORKER and proof['GPU']==2 and proof['returncode']==0
 live=identity(OLD_WORKER['pid']);assert live is None or (live['startticks'],live['uid'])!=(OLD_WORKER['startticks'],OLD_WORKER['uid'])
 st=read(RUN/'run_status.json');meta=read(RUN/'checkpoints/latest.json')
 assert st['status']=='completed' and st['completed_stage']==19 and meta['stage']==19 and meta['method']=='OSFT'
 assert meta['sha256']=='a041efb3fd178d603b164740b59a6de7ad60608e2fc38d2acae8023963ae69bd' and meta['bytes']==3028599989
 assert (RUN/meta['path']).stat().st_size==meta['bytes']
 return proof

def validate(binding,require_admission=True):
 assert os.getuid()==1084 and os.environ.get('PYTHONOPTIMIZE','') in ('','0')
 b=read(binding);assert b['schema']=='residual-exit-pilot-v1'
 assert Path(b['source_dir'])==E and b['sources']==SOURCES and b['gpu_uuid']==UUID and b['limit_seconds']==7200
 assert all(sha(E/n)==h for n,h in SOURCES.items())
 assert sha(b['protocol_path'])==b['protocol_sha256']
 assert sha(b['asset_receipt'])==b['asset_receipt_sha256']
 a=None
 if require_admission:
  assert b['admitted'] is True
  a=read(b['admission_path']);assert a['approved'] is True and a['binding_sha256']==sha(binding)
  assert a['controller_sha256']==sha(Path(__file__)) and a['scientific_source_review_accepted'] is True and a['final_metadata_verified'] is True
  assert sha(a['source_review_path'])==a['source_review_sha256'] and sha(a['metadata_finalization_path'])==a['metadata_finalization_sha256']
  assert a['OSFT_terminal_proof_sha256']==sha(D/'OSFT_TERMINAL_PROOF.json')
 proof=terminal_guard();idle()
 return b,a,proof
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 b,a,terminal=validate(args.binding,args.execute)
 if not args.execute:
  print(json.dumps({'inspection_only':True,'models_started':0,'GPU':2,'serial_methods':METHODS,'original_queue_and_locks_unmodified':True}));return
 assert not O.exists();assert all(not (Path(b['output_root'])/m).exists() for m in METHODS)
 O.mkdir();lease=(O/'lease.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Exclusive lease/start marker is never automatically removed or replayed.
 claim=dict(status='claimed',GPU=2,gpu_uuid=UUID,donor=None,OSFT_terminal_proof_sha256=sha(D/'OSFT_TERMINAL_PROOF.json'),binding_sha256=sha(args.binding),controller_sha256=sha(Path(__file__)),identity=identity(os.getpid()),methods=METHODS,original_locks=[],no_retry=True,released=False)
 write(O/'CLAIM.json',claim);write(O/'SLOT.json',claim)
 child=None;worker=None;phases=[]
 def interrupted(signum,*_):raise InterruptedError('research controller signal '+str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  for method in METHODS:
   validate(args.binding,True);idle() # No reading partial outcomes to change the next method.
   cmd=[str(R/'.vision-env/bin/python'),'-u',str(E/'run.py'),'--binding',str(args.binding),'--method',method,'--execute']
   env=dict(os.environ,LLMCL_ROOT=str(R),CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1')
   with (O/(method+'.log')).open('x') as log:child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   worker=identity(child.pid);assert worker and worker['uid']==1084 and worker['pid']==worker['pgid']
   write(O/(method+'_STARTED.json'),dict(identity=worker,command=cmd,GPU=2,unix=time.time(),binding_sha256=sha(args.binding),no_retry=True))
   code=child.wait(timeout=b['limit_seconds']);gone=not (Path('/proc')/str(worker['pid'])).exists()
   phase=dict(method=method,identity=worker,returncode=code,worker_gone=gone,unix=time.time());write(O/(method+'_EXIT.json'),phase);phases.append(phase)
   assert code==0 and gone,'No retry or next teacher after a failed model'
   folder=Path(b['output_root'])/method;assert not (folder/'FAILURE.json').exists()
   c=read(folder/'COMPLETE.json');assert c['identity']==worker and c['binding_sha256']==sha(args.binding) and c['RESULT_sha256']==sha(folder/'RESULT.json')
   result=read(folder/'RESULT.json');assert result['method']==method and result['not_quality_verification'] and result['counts']['TEST_QA']==result['counts']['F1']==result['counts']['optimizer_updates']==0
   terminal_guard();idle();child=None;worker=None
  write(O/'COMPLETE.json',dict(complete=True,phases=phases,binding_sha256=sha(args.binding),GPU=2,gpu_uuid=UUID,no_retry=True,released=True,quality_verified=False,OSFT_resume_still_required=False,unix=time.time()))
  current=read(O/'SLOT.json');assert current==claim
  current.update(status='complete',released=True,terminal_path=str(O/'COMPLETE.json'),terminal_sha256=sha(O/'COMPLETE.json'),release_unix=time.time())
  tmp=O/'SLOT.complete.tmp';write(tmp,current);os.replace(tmp,O/'SLOT.json');fd=os.open(O,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 except BaseException:
  error=traceback.format_exc()
  # Only this launcher's own research process can receive cleanup signals.
  if child is not None and child.poll() is None and worker and identity(worker['pid'])==worker:
   os.killpg(worker['pid'],signal.SIGTERM)
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:
    if identity(worker['pid'])==worker:os.killpg(worker['pid'],signal.SIGKILL)
    child.wait(timeout=30)
  gone=worker is None or not (Path('/proc')/str(worker['pid'])).exists()
  write(O/'FAILURE.json',dict(traceback=error,phases=phases,last_worker=worker,last_worker_gone=gone,slot_requires_explicit_terminal_audit=True,no_retry=True,unix=time.time()))
  raise
if __name__=='__main__':main()
