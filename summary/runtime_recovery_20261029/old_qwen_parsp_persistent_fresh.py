from pathlib import Path
import json,os,hashlib,subprocess,importlib.util,time,fcntl
R=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_');F=R/'exp/two_backbone_pct10_native_checkpoint_20261027';PERSIST=Path('/data2/liyapeng_grp/program/LLMCL/.deployment/storage_recovery_20261029_second');O=PERSIST/'QWEN_PARSP_FRESH';NEW=F/'runs/qwen/pct10/PaRSP/training_storage_recovery_20261029'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def ident(pid):
 try:
  p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return None if f[0]=='Z' else dict(pid=pid,startticks=f[19],uid=p.stat().st_uid)
 except FileNotFoundError:return None
assert os.getuid()==1084 and not O.exists() and not NEW.exists()
assert 3 in read(F/'execution_policy.json')['hosts']['old_pct10_qwen']['conditions']['pct10']
lock=(F/'.pipeline_old_pct10_qwen.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
source={p.name:sha(p) for p in F.glob('*.py')};base=F/'runs/qwen/pct10/PaRSP';old=base/'training_20261029_131134';pre=base/'preflight_20261027_200815'
cfg=read(old/'config.json');assert (cfg['method'],cfg['backbone'],cfg['condition'])==('PaRSP','qwen','pct10')
assert len(source)==25 and read(old/'config.json')['execution_source_sha256']==source
assert not list((old/'checkpoints').glob('*.pt'))
e=old/'logs/events.jsonl';assert not e.exists() or not e.read_bytes().strip()
for n in ['gpu_preflight.json','checkpoint_reload_proof.json']:
 v=read(pre/n);assert v['passed'] and v['execution_source_sha256']==source and v['manifest_sha256']=='02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'
assert read(pre/'checkpoint_reload_proof.json')['fresh_process']
assert not subprocess.check_output(['nvidia-smi','--id=3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid==1084:assert str(old).encode() not in (p/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
O.mkdir();write(O/'INTENT.json',dict(unix=time.time(),old_run=str(old),sources=source,original_updates=0,fresh_not_resume=True,GPU=3,script_sha256=sha(Path(__file__))))
os.environ.update(LLMCL_ROOT=str(R),LLMCL_HOST='old_pct10_qwen',LLMCL_BACKBONE='qwen')
sp=importlib.util.spec_from_file_location('original_pipeline',F/'pipeline.py');pipe=importlib.util.module_from_spec(sp);sp.loader.exec_module(pipe)
assert pipe.HOST=='old_pct10_qwen' and pipe.HERE==F
NEW.mkdir(parents=True,exist_ok=False)
cp=PERSIST/'runs/Qwen_PaRSP_fresh/checkpoints';cp.mkdir(parents=True,exist_ok=False)
(NEW/'checkpoints').symlink_to(cp,target_is_directory=True)
j=dict(method='PaRSP',condition='pct10',gpu=3,run_dir=str(NEW),host='old_pct10_qwen',fresh_not_resume=True)
command=[str(pipe.PYTHON),'-u',str(F/'run.py'),'--method','PaRSP','--condition','pct10','--run-dir',str(NEW)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
with (NEW/'supervisor.log').open('x') as log:p=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(O/'STARTED.json',dict(identity=ident(p.pid),job=j,original_failed_run=str(old),fresh_not_resume=True,command=command))
c=p.wait();write(O/'EXIT.json',dict(unix=time.time(),returncode=c,identity_absent=ident(p.pid) is None,phase='training',job=j,not_quality_verification=True))
