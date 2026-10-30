"""One finite CL fixed-effective-G worker; original SpaRTA baseline pauses separately.
Runtime control only. No queue, allocator, scientific source edit or quality parsing.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,traceback
P=Path('/data2/liyapeng_grp/program/LLMCL');S=P/'summary/cl_fixed_effective_gate_necessity_20261030';E=P/'exp/cl_fixed_effective_gate_necessity_20261030'
F=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_/exp/two_backbone_pct10_native_checkpoint_20261027');DONOR=F/'runs/onevision/pct10/SpaRTA/training_gpu1_20261030';LOCK=P/'summary/native_prefix_function_observation_20261030/path_compat_v1/GPU1_OBSERVATION.lock'
UUID='GPU-3c72e4ce-0896-0b40-d610-be585d2bb51e';REQUEST_SHA='57fc5f402e2ff62f77c87b65dee2c4883d17ad1d93cd057260c9c6351b037614';NATIVE_SHA='882057aab8dadfed58663eb46e66213936581e0ad6b4bb2d1f4a96b776a414f4'
SCI={'run.py':'3a9c40ce95fa11c0fcd4337b29ee4c4808c2aab0eea07e08ebb8352e01c3bbca','verify.py':'7aebbb6438c09c93e59a9e1a884ff46aac67b23be6118e50c92c8bb65d1ced24','fixed_gate.py':'60323dba3c5145a3dcce44da585c5c917891a29e9f4a9260f86b6e16b1534f81','controls.py':'c8a232bd20cfd2deb360316daa5557e34605bdba41257344337fcddd426dac23'}
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
 assert b['schema']=='cl-fixed-effective-gate-v1' and b['method']=='CL_MoE' and b['condition']=='pct01' and b['backbone']=='onevision'
 assert b['gpu_uuid']==UUID and b['cuda_visible_devices']=='1' and b['limit_seconds']==86400 and Path(b['source_dir'])==E and Path(b['worker_exit_receipt_path'])==S/'WORKER_EXIT.json'
 assert b['scientific_sources']==SCI
 for n,h in SCI.items():assert sha(E/n)==h
 assert b['protocol_sha256']=='a35619fa0a5e2dd43e56b50924727e362ed02b3d3e2d8b3c5d0ec61697743bfb' and sha(b['protocol_path'])==b['protocol_sha256']
 assert a['review_records']
 for x in a['review_records']:assert sha(x['path'])==x['sha256']
 assert sha(b['donor_boundary_proof_path'])==b['donor_boundary_proof_sha256'];proof=read(b['donor_boundary_proof_path']);k=proof['paused_stage']
 assert proof['complete'] and 16<=k<=18 and proof['native_worker_returncode']==0 and proof['full_checkpoint_SHA_verified'] and proof['original_CPU_controller_returncode']==0
 assert proof['worker_identity']==dict(pid=1946673,startticks='806193331',uid=1084,pgid=1946673) and proof['controller_identity']==dict(pid=1946636,startticks='806193288',uid=1084,pgid=1946636)
 assert {'rng','python','numpy','cpu','cuda'}<=set(proof['rng_metadata_keys_present'])
 for who in (proof['worker_identity'],proof['controller_identity']):assert identity(who['pid'])!=who
 for x in proof['source_records']:assert sha(x['path'])==x['sha256']
 cp=Path(proof['checkpoint']['path']);st=cp.stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==tuple(proof['checkpoint'][q] for q in ('device','inode','bytes','mtime_ns'))
 assert Path(proof['native_run'])==DONOR and cp.resolve()==(DONOR/'checkpoints'/('stage_%02d.pt'%k)).resolve() and sha(F/'run.py')==NATIVE_SHA
 status=read(DONOR/'run_status.json');assert status['status']=='paused_checkpoint' and status['completed_stage']==k and not (DONOR/'RESUME_REQUEST.json').exists() and not (DONOR/'PAUSE_REQUEST.json').exists()
 pr=DONOR/status['pause_receipt'];assert sha(pr)==status['pause_receipt_sha256'];pause=read(pr);latest=read(DONOR/'checkpoints/latest.json')
 assert pause['completed_stage']==k and pause['next_stage']==k+1 and pause['full_stage_complete'] and pause['request_sha256']==REQUEST_SHA and pause['request_id']=='cl-fixed-effective-g-functional-observation-20261030'
 assert latest==pause['checkpoint'] and latest['stage']==k and latest['sha256']==proof['checkpoint']['sha256'] and latest['bytes']==st.st_size
 assert subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==UUID
 assert not subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 assert LOCK.is_file() and not (S/'CONTROL_STARTED.json').exists() and not (S/'SLOT.json').exists() and not Path(b['output_root']).exists()
 return b,a

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');arg=ap.parse_args();b,a=inspect()
 if not arg.execute:print(json.dumps(dict(metadata_ready=True,models_started=0,quality_values_read=0,one_fixed_CL_worker=True)));return
 with LOCK.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);b,a=inspect();child=None;who=None
  claim=dict(identity=identity(os.getpid()),binding_sha256=sha(S/'BINDING.json'),status='running',released=False,no_automatic_retry=True,unix=time.time())
  write(S/'CONTROL_STARTED.json',claim);write(S/'SLOT.json',claim)
  try:
   cmd=[str(P/'.vision-env/bin/python'),'-u',str(E/'run.py'),'--binding',str(S/'BINDING.json'),'--execute']
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
   with (S/'WORKER.log').open('x') as log:
    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);who=identity(child.pid);assert who and who['pgid']==child.pid and who['uid']==1084
    write(S/'WORKER_STARTED.json',dict(identity=who,command=cmd,binding_sha256=sha(S/'BINDING.json'),limit_seconds=86400))
    rc=child.wait(timeout=86400);log.flush();os.fsync(log.fileno())
   assert identity(who['pid'])!=who;write(S/'WORKER_EXIT.json',dict(identity=who,returncode=rc,identity_gone=True,unix=time.time()))
   assert rc==0,'Observation worker failed; preserve and stop, no automatic retry'
   out=Path(b['output_root']);c=read(out/'COMPLETE.json');assert c['identity']==who and c['binding_sha256']==sha(S/'BINDING.json') and c['RESULT_sha256']==sha(out/'RESULT.json') and not (out/'FAILURE.json').exists()
   # Hash only: the scientific verifier owns TEST coverage, scores and interpretation.
   write(S/'CONTROL_COMPLETE.json',dict(worker_identity=who,worker_returncode=rc,worker_gone=True,worker_COMPLETE_sha256=sha(out/'COMPLETE.json'),RESULT_sha256=c['RESULT_sha256'],released=True,quality_verified=False,no_scientific_result_parsed=True,unix=time.time()))
   replace(S/'SLOT.json',dict(status='complete',released=True,CONTROL_COMPLETE_sha256=sha(S/'CONTROL_COMPLETE.json')))
  except BaseException:
   if child is not None and child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=20)
   own_child_gone=child is None or child.poll() is not None
   if child is not None and not (S/'WORKER_EXIT.json').exists():write(S/'WORKER_EXIT.json',dict(identity=who,returncode=child.returncode,identity_gone=(identity(child.pid) is None if who is None else identity(child.pid)!=who),unix=time.time(),abnormal_parent_exit=True))
   write(S/'CONTROL_FAILURE.json',dict(traceback=traceback.format_exc(),own_child_gone=own_child_gone,unix=time.time(),no_automatic_retry=True))
   replace(S/'SLOT.json',dict(status='failed',released=own_child_gone,CONTROL_FAILURE_sha256=sha(S/'CONTROL_FAILURE.json')));raise
if __name__=='__main__':main()
