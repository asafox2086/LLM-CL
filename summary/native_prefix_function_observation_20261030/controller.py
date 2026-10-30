"""Finite two-checkpoint observation lease; original baseline paused separately."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,traceback
P=Path('/data2/liyapeng_grp/program/LLMCL');S=P/'summary/native_prefix_function_observation_20261030';E=P/'exp/native_prefix_function_observation_20261030'
UUID='GPU-3c72e4ce-0896-0b40-d610-be585d2bb51e'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):
 with p.open('x') as f:json.dump(j,f,indent=2,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def replace(p,j):
 q=p.with_suffix('.next');write(q,j);os.replace(q,p)
 fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def identity(pid):
 try:
  p=Path('/proc')/str(pid);a=(p/'stat').read_text().rsplit(') ',1)[1].split();return dict(pid=pid,startticks=a[19],uid=p.stat().st_uid,pgid=int(a[2]))
 except FileNotFoundError:return None
def inspect():
 b=read(S/'BINDING.json');a=read(S/'ADMISSION.json');assert os.getuid()==1084
 assert b['admitted'] and a['approved'] and a['binding_sha256']==sha(S/'BINDING.json') and a['controller_sha256']==sha(Path(__file__))
 assert b['gpu_uuid']==UUID and b['cuda_visible_devices']=='1'
 for n,h in b['scientific_sources'].items():assert sha(E/n)==h
 for x in a['review_records']:assert sha(x['path'])==x['sha256']
 assert sha(b['donor_pause_proof_path'])==b['donor_pause_proof_sha256'];proof=read(b['donor_pause_proof_path'])
 assert proof['complete'] and proof['paused_stage']==14 and proof['native_worker_returncode']==0 and proof['full_checkpoint_SHA_verified']
 assert proof['original_CPU_controller_returncode']==1 and proof['CPU_trace_available'] is False
 assert proof['checkpoint']['sha256']=='8ba5a626336f4531539cd44efdfdf0bb5e8002ad0e17bb2c7abec9bcb6a662da'
 assert {'rng','python','numpy','cpu','cuda'}<=set(proof['rng_metadata_keys_present'])
 for who in (proof['worker_identity'],proof['controller_identity']):assert identity(who['pid'])!=who
 for x in proof['source_records']:assert sha(x['path'])==x['sha256']
 cp=Path(proof['checkpoint']['path']);st=cp.stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==tuple(proof['checkpoint'][k] for k in ('device','inode','bytes','mtime_ns'))
 run=Path(proof['native_run']);assert read(run/'run_status.json')['status']=='paused_checkpoint' and not (run/'RESUME_REQUEST.json').exists()
 assert subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==UUID
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(x.split(',')[0].strip()==UUID for x in apps.splitlines() if x.strip())
 assert not (S/'CONTROL_STARTED.json').exists() and not (S/'SLOT.json').exists() and not Path(b['output_root']).exists()
 return b,a
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');arg=ap.parse_args();b,a=inspect()
 if not arg.execute:print(json.dumps(dict(metadata_ready=True,original_model_loads=0,new_generations=0)));return
 with (S/'GPU1_OBSERVATION.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);b,a=inspect();child=None;finished=[]
  claim=dict(identity=identity(os.getpid()),binding_sha256=sha(S/'BINDING.json'),status='running',released=False,no_automatic_retry=True,unix=time.time())
  write(S/'CONTROL_STARTED.json',claim);write(S/'SLOT.json',claim)
  try:
   for method in ('PASs_MoE','CL_MoE'):
    cmd=[str(P/'.vision-env/bin/python'),'-u',str(E/'run.py'),'--binding',str(S/'BINDING.json'),'--method',method,'--execute']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    with (S/(method+'_WORKER.log')).open('x') as log:
     child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);who=identity(child.pid);assert who and who['pgid']==child.pid and who['uid']==1084
     write(S/(method+'_STARTED.json'),dict(identity=who,command=cmd,binding_sha256=sha(S/'BINDING.json')))
     rc=child.wait(timeout=7260);log.flush();os.fsync(log.fileno())
    assert identity(who['pid'])!=who;write(S/(method+'_EXIT.json'),dict(identity=who,returncode=rc,identity_gone=True,unix=time.time()))
    assert rc==0,'Observation worker failed; preserve and stop, no automatic retry'
    out=Path(b['output_root'])/method;c=read(out/'COMPLETE.json');assert c['binding_sha256']==sha(S/'BINDING.json') and c['RESULT_sha256']==sha(out/'RESULT.json') and not (out/'FAILURE.json').exists()
    r=read(out/'RESULT.json');assert r['counts']['generations']==r['counts']['TRAIN_queries']==80 and r['weights_and_buffers_unchanged'] and r['generation_config_unchanged']
    assert r['counts']['F1']==r['counts']['CE']==r['counts']['optimizer_updates']==0
    finished.append(dict(method=method,COMPLETE_sha256=sha(out/'COMPLETE.json'),RESULT_sha256=c['RESULT_sha256']))
   write(S/'CONTROL_COMPLETE.json',dict(finished=finished,released=True,not_quality_verification=True,unix=time.time()))
   replace(S/'SLOT.json',dict(status='complete',released=True,CONTROL_COMPLETE_sha256=sha(S/'CONTROL_COMPLETE.json')))
  except BaseException:
   if child is not None and child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=20)
   own_child_gone=child is None or child.poll() is not None
   write(S/'CONTROL_FAILURE.json',dict(traceback=traceback.format_exc(),finished=finished,own_child_gone=own_child_gone,unix=time.time(),no_automatic_retry=True))
   replace(S/'SLOT.json',dict(status='failed',released=own_child_gone,CONTROL_FAILURE_sha256=sha(S/'CONTROL_FAILURE.json')));raise
if __name__=='__main__':main()
