"""One finite TRAIN observation after native CL S00 pause; no retry/allocator."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,traceback
P=Path('/data2/liyapeng_grp/program/LLMCL')
D=P/'summary/parsp_update_geometry_ce_20261030'
E=P/'exp/parsp_update_geometry_ce_20261030'
UUID='GPU-25d3cb92-fa97-a722-f9cd-c012bad046a8'
SCI={'run.py':'4a80684bf0090a048ee40c69833a5d1b85949e06c72a5315fc27688c453b29a6','measure.py':'db6f3a373a6372569201eefabf2fae3a31c25ae90c88da64d64c07236d7ddc80'}
REVIEW_SHA='6ea41a8945ff4df13d8c432ce7df76e1932cf353b78410aa462257be4b73c6f2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(8*1024**2),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def syncdir(p):
 fd=os.open(p,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
 syncdir(p.parent)
def replace(p,v):
 q=p.with_suffix('.next');write(q,v);os.replace(q,p);syncdir(p.parent)
def identity(pid):
 q=Path('/proc')/str(pid);s=(q/'stat').read_text().rsplit(') ',1)[1].split()
 return dict(pid=pid,startticks=s[19],uid=q.stat().st_uid,pgid=int(s[2]))
def inspect():
 assert os.getuid()==1084
 b=read(D/'BINDING.json');a=read(D/'ADMISSION.json')
 assert b['admitted'] is True and b['physical_GPU']==2 and b['gpu_uuid']==UUID
 assert a['approved'] is True and a['binding_sha256']==sha(D/'BINDING.json')
 assert a['controller_sha256']==sha(Path(__file__))
 for n,h in SCI.items():assert b['sources'][n]==h and sha(E/n)==h
 review=D/'SCIENTIFIC_SOURCE_REVIEW_LITERATURE_FINAL_20261030.json'
 assert sha(review)==REVIEW_SHA and a['scientific_review_sha256']==REVIEW_SHA
 proof=Path(a['CL_pause_proof_path']);assert sha(proof)==a['CL_pause_proof_sha256'];p=read(proof)
 assert p['complete'] and p['native_stage00_pause'] and p['full_checkpoint_SHA_verified'] and p['rng_present']
 assert p['worker_identity']==dict(pid=1650061,startticks='804287271',uid=1084,pgid=1650061)
 assert p['controller_identity']==dict(pid=1650035,startticks='804287245',uid=1084,pgid=1650035)
 for item in p['source_records']:assert sha(item['path'])==item['sha256']
 for old in (1650061,1650035):assert not Path('/proc',str(old)).exists()
 cp=Path(p['checkpoint']['path']);s=cp.stat()
 assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==tuple(p['checkpoint'][k] for k in ('device','inode','bytes','mtime_ns'))
 assert read(p['native_run']+'/run_status.json')['status']=='paused_checkpoint'
 assert not (Path(p['native_run'])/'RESUME_REQUEST.json').exists()
 uuid=subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==UUID
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(x.strip().split(',')[0].strip()==UUID for x in apps.splitlines() if x.strip())
 assert not Path(b['output_root']).exists()
 assert not (D/'CONTROL_STARTED.json').exists() and not (D/'SLOT.json').exists()
 return b,a,p
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 b,a,p=inspect()
 if not args.execute:print(json.dumps(dict(READY=True,models_started=0,claim_created=False)));return
 with (D/'observation_GPU2.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);b,a,p=inspect()
  claim=dict(status='claimed',released=False,unix=time.time(),binding_sha256=sha(D/'BINDING.json'),admission_sha256=sha(D/'ADMISSION.json'),identity=identity(os.getpid()),physical_GPU=2,gpu_uuid=UUID,no_retry=True)
  write(D/'CONTROL_STARTED.json',claim);write(D/'SLOT.json',claim);child=None
  try:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
   cmd=[str(P/'.vision-env/bin/python'),'-u',str(E/'run.py'),'--binding',str(D/'BINDING.json'),'--execute']
   with (D/'WORKER.log').open('x') as log:
    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    who=identity(child.pid);write(D/'WORKER_STARTED.json',dict(identity=who,command=cmd,binding_sha256=sha(D/'BINDING.json')))
    rc=child.wait(timeout=7200)
   assert not Path('/proc',str(child.pid)).exists()
   write(D/'WORKER_EXIT.json',dict(identity=who,returncode=rc,identity_absent=True,unix=time.time()))
   assert rc==0
   out=Path(b['output_root']);assert not (out/'FAILURE.json').exists()
   c=read(out/'COMPLETE.json');assert c['binding_sha256']==sha(D/'BINDING.json') and c['RESULT_sha256']==sha(out/'RESULT.json')
   write(D/'CONTROL_COMPLETE.json',dict(binding_sha256=sha(D/'BINDING.json'),worker_EXIT_sha256=sha(D/'WORKER_EXIT.json'),model_COMPLETE_sha256=sha(out/'COMPLETE.json'),result_sha256=sha(out/'RESULT.json'),released=True,quality_verified=False,new_scores=0,unix=time.time()))
   replace(D/'SLOT.json',dict(claim=claim,status='complete',released=True,CONTROL_COMPLETE_sha256=sha(D/'CONTROL_COMPLETE.json')))
  except BaseException:
   if child is not None and child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=20)
   write(D/'CONTROL_FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time(),no_retry=True,slot_released=False,child_returncode=child.returncode if child else None));raise
if __name__=='__main__':main()
