from pathlib import Path
import argparse,subprocess,os,json,hashlib,time,signal,traceback
p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);args=p.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
b=json.loads(args.binding.read_bytes());d=Path(b['admission_path']).parent;e=Path(b['observation_source_dir']);ad=json.loads(Path(b['admission_path']).read_bytes())
assert ad['approved'] and ad['binding_sha256']==sha(args.binding) and ad['controller_sha256']==sha(__file__)
assert os.getuid()==1084
write(d/'CONTROLLER_STARTED.json',dict(unix=time.time(),pid=os.getpid(),binding_sha256=sha(args.binding),no_retry=True,methods=['PASs_MoE','CL_MoE']))
try:
 for method in ['PASs_MoE','CL_MoE']:
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
  assert not any(line.split(',')[0].strip()==b['gpu_uuid'] for line in gpu.splitlines() if ',' in line),'Research GPU is occupied; no worker started'
  out=Path(b['output_root'])/method
  assert not out.exists()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
  with (d/(method+'_worker.log')).open('xb') as log:
   proc=subprocess.Popen(['/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python',str(e/'run.py'),'--binding',str(args.binding),'--method',method,'--execute'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   stat=Path('/proc')/str(proc.pid)/'stat';fields=stat.read_text().rsplit(') ',1)[1].split();who=dict(pid=proc.pid,startticks=fields[19],pgid=os.getpgid(proc.pid),uid=1084)
   write(d/(method+'_SPAWNED.json'),dict(identity=who,unix=time.time(),binding_sha256=sha(args.binding)))
   try:ret=proc.wait(timeout=7260)
   except subprocess.TimeoutExpired:
    current=stat.read_text().rsplit(') ',1)[1].split();assert current[19]==who['startticks'] and os.getpgid(proc.pid)==proc.pid
    os.killpg(proc.pid,signal.SIGTERM)
    try:ret=proc.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);ret=proc.wait()
    write(d/(method+'_DEADLINE.json'),dict(identity=who,returncode=ret));raise TimeoutError('Fixed observer deadline; no retry')
  exit_={'identity':who,'returncode':ret,'unix':time.time()};write(d/(method+'_EXIT.json'),exit_)
  if out.exists():write(out/'EXIT.json',exit_)
  assert ret==0 and (out/'COMPLETE.json').exists() and (out/'RESULT.json').exists() and not (out/'FAILURE.json').exists()
  assert not stat.exists() or stat.read_text().rsplit(') ',1)[1].split()[19]!=who['startticks']
 write(d/'CONTROLLER_COMPLETE.json',dict(complete=True,unix=time.time(),binding_sha256=sha(args.binding),no_retry=True,model_attempts=2))
except BaseException:
 write(d/'CONTROLLER_FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time()));raise
