"""One exclusive GPU3 lease, two serial original teachers, never retry.
Draft control entry: no action on import; default inspect never starts a model.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,traceback
R=Path('/data1/liyapeng/program/LLMCL');E=R/'exp/residual_exit_pilot_20261030';D=R/'summary/residual_exit_pilot_20261030'
O=D/'CONTROLLER_V1';METHODS=('PASs_MoE','CL_MoE')
SOURCES={'run.py':'838a05c11f8b3ec6c2141c4dd45d4913f32b129829a977840c736e994e9ac2ed',
 'probe.py':'d4c756a932a7c15dccad31d2296d9722df80716b5b00935b170764f00b321063',
 'ridge.py':'b77350ac0df5557ac9496332fb65c79ca99bdc9dc549f8c57a0254abf49791bb'}
DONOR={'pid':261266,'startticks':'66873152','uid':1004,'pgid':261266}
OLD_WORKER={'pid':261356,'startticks':'66873512','uid':1004,'pgid':261356}
RUN=R/'exp/two_backbone_native_checkpoint_20261027/runs/qwen/pct01/OSFT/recovery_disk_third_20261030_S17'
UUID='GPU-fb856f23-d2d1-87af-54d4-f805742da76f'
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
 assert subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==UUID
 assert not subprocess.check_output(['nvidia-smi','--id=3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def original_locks():
 s=R/'summary/two_backbone_native_cl_20261017_v2';f=R/'exp/two_backbone_native_cl_20261017_v2';t=R/'exp/two_backbone_pct10_native_checkpoint_20261027'
 paths=[s/'baseline_pct01_capacity.lock',s/'hardware_queue.lock',f/'.pipeline_new3090.lock',f/'.pipeline_new3090_onevision.lock',R/'summary/two_backbone_pct10_20261025/dispatch_new.lock',t/'.pipeline_new_pct10_onevision.lock',t/'.pipeline_new_pct10_qwen.lock']
 lines=Path('/proc/locks').read_text().splitlines();out=[]
 assert identity(DONOR['pid'])==DONOR
 for p in paths:
  st=p.stat();key=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
  held=[v for v in lines if len(v.split())>5 and v.split()[5]==key]
  assert len(held)==1 and int(held[0].split()[4])==DONOR['pid'],str(p)
  out.append(dict(path=str(p),holder=DONOR))
 return out
def validate(binding,cp_full_sha=False):
 assert os.getuid()==1004 and os.environ.get('PYTHONOPTIMIZE','') in ('','0')
 b=read(binding);assert b['schema']=='residual-exit-pilot-v1' and b['admitted'] is True
 assert Path(b['source_dir'])==E and b['sources']==SOURCES and b['gpu_uuid']==UUID and b['limit_seconds']==7200
 assert all(sha(E/n)==h for n,h in SOURCES.items())
 a=read(b['admission_path']);assert a['approved'] is True and a['binding_sha256']==sha(binding)
 assert a['controller_sha256']==sha(Path(__file__)) and a['scientific_source_review_accepted'] is True and a['final_metadata_verified'] is True
 assert sha(a['source_review_path'])==a['source_review_sha256'] and sha(a['metadata_finalization_path'])==a['metadata_finalization_sha256']
 assert sha(a['pause_proof_path'])==a['pause_proof_sha256'];p=read(a['pause_proof_path'])
 assert p['schema']=='native-osft-S18-pause-closed-v1' and p['verified'] is True
 assert p['worker']==OLD_WORKER and p['worker_gone'] and p['exit_returncode']==0 and p['completed_stage']==18
 assert p['checkpoint_full_sha256_verified'] and p['checkpoint_actual_sha256']==p['checkpoint']['sha256']
 # Recheck actual immutable native proof, not only the prior observation JSON.
 assert not (Path('/proc')/str(OLD_WORKER['pid'])).exists()
 st=read(RUN/'run_status.json');assert st['status']=='paused_checkpoint' and st['completed_stage']==18
 receipt=RUN/st['pause_receipt'];assert sha(receipt)==st['pause_receipt_sha256']==p['pause_receipt_sha256']
 pr=read(receipt);m=read(RUN/'checkpoints/latest.json');assert m==pr['checkpoint']==p['checkpoint'] and m['stage']==18
 assert pr['full_stage_complete'] and pr['request_sha256']=='0eb23835cfb67ed789554c7caec1ad4241ae9bff65e5a51359ae688f9a4f93e1'
 assert sha(RUN/'config.json')==pr['config_sha256'] and sha(RUN/'checkpoints/latest.json')==pr['checkpoint_metadata_sha256']
 assert not (RUN/'RESUME_REQUEST.json').exists()
 cp=RUN/m['path'];assert cp.stat().st_size==m['bytes']
 if cp_full_sha:assert sha(cp)==m['sha256']
 narrow=R/'summary/storage_recovery_20261030/NEW_BASELINE_RECOVERY/new_native_narrow_resume.py'
 assert sha(narrow)=='1f8fb518156920312548232fd54d48fe331af27fa4ae8db615b355fb33b52cb6'
 ex=read(narrow.parent/'NARROW_LAUNCH/OSFT_EXIT.json');assert ex['identity']==OLD_WORKER and ex['returncode']==0 and ex['identity_absent']
 unit=subprocess.check_output(['systemctl','--user','show','llmcl-two-backbone-native.service','-p','MainPID','-p','ActiveState'],text=True)
 u=dict(v.split('=',1) for v in unit.splitlines());assert u['MainPID']=='0' and u['ActiveState'] in ('failed','inactive')
 locks=original_locks();idle()
 return b,a,locks
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 b,a,locks=validate(args.binding,False)
 if not args.execute:
  print(json.dumps({'inspection_only':True,'models_started':0,'GPU':3,'serial_methods':METHODS,'donor_locks_unmodified':True}));return
 assert not O.exists();assert all(not (Path(b['output_root'])/m).exists() for m in METHODS)
 O.mkdir();lease=(O/'lease.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Exclusive lease/start marker is never automatically removed or replayed.
 claim=dict(status='claimed',GPU=3,gpu_uuid=UUID,donor=DONOR,OSFT_pause_proof_sha256=a['pause_proof_sha256'],binding_sha256=sha(args.binding),controller_sha256=sha(Path(__file__)),identity=identity(os.getpid()),methods=METHODS,original_locks=locks,no_retry=True,released=False)
 write(O/'CLAIM.json',claim);write(O/'SLOT.json',claim)
 child=None;worker=None;phases=[]
 def interrupted(signum,*_):raise InterruptedError('research controller signal '+str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  for method in METHODS:
   validate(args.binding,False);idle() # No reading partial outcomes to change the next method.
   cmd=[str(R/'.vision-env/bin/python'),'-u',str(E/'run.py'),'--binding',str(args.binding),'--method',method,'--execute']
   env=dict(os.environ,LLMCL_ROOT=str(R),CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1')
   with (O/(method+'.log')).open('x') as log:child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   worker=identity(child.pid);assert worker and worker['uid']==1004 and worker['pid']==worker['pgid']
   write(O/(method+'_STARTED.json'),dict(identity=worker,command=cmd,GPU=3,unix=time.time(),binding_sha256=sha(args.binding),no_retry=True))
   code=child.wait(timeout=b['limit_seconds']);gone=not (Path('/proc')/str(worker['pid'])).exists()
   phase=dict(method=method,identity=worker,returncode=code,worker_gone=gone,unix=time.time());write(O/(method+'_EXIT.json'),phase);phases.append(phase)
   assert code==0 and gone,'No retry or next teacher after a failed model'
   folder=Path(b['output_root'])/method;assert not (folder/'FAILURE.json').exists()
   c=read(folder/'COMPLETE.json');assert c['identity']==worker and c['binding_sha256']==sha(args.binding) and c['RESULT_sha256']==sha(folder/'RESULT.json')
   result=read(folder/'RESULT.json');assert result['method']==method and result['not_quality_verification'] and result['counts']['TEST_QA']==result['counts']['F1']==result['counts']['optimizer_updates']==0
   original_locks();idle();child=None;worker=None
  write(O/'COMPLETE.json',dict(complete=True,phases=phases,binding_sha256=sha(args.binding),GPU=3,gpu_uuid=UUID,no_retry=True,released=True,quality_verified=False,OSFT_resume_still_required=True,unix=time.time()))
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
