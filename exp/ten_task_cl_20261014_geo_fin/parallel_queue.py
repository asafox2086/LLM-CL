"""New split only: one full sequential run per method, one GPU per run."""
import os,json,time,subprocess,hashlib,fcntl,csv,io
from pathlib import Path
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPORT=ROOT/"summary"/HERE.name
PY=str(ROOT/".vision-env/bin/python")
BOOT_ID=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
METHODS=["SMoLoRA","PASs_MoE","Hyper_LLaVA","PaRSP","OSFT","SpaRTA","FV_guided_Training","ALER"]
def write(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n");tmp.replace(path)
def environment(gpu=None):
 e=os.environ.copy()
 for k in list(e):
  if k.lower() in ["http_proxy","https_proxy","all_proxy"]:e.pop(k)
 e.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1",PYTHONUTF8="1",PYTHONIOENCODING="utf-8",LANG="C.UTF-8",LC_ALL="C.UTF-8")
 if gpu is not None:e["CUDA_VISIBLE_DEVICES"]=str(gpu)
 return e
def spawn(name,run,gpu,preflight):
 h=(run/"logs"/("gpu_preflight.log" if preflight else "train.log")).open("a")
 cmd=[PY,str(HERE/"run.py"),"--method",name,"--seed","42","--run-dir",str(run)]
 if preflight:cmd.append("--preflight-only")
 env=environment(gpu)
 if name=="PaRSP":env.update(OMP_NUM_THREADS="8",MKL_NUM_THREADS="8")
 return subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT),h
def record_interruption(name,job,reason):
 record={"method":name,"previous_run_dir":job["run_dir"],"previous_phase":job["status"],"last_completed_stage":job.get("completed_stage",0),"reason":reason,"restart_from_base_required":True,"checkpoint_saved":False,"boot_id":BOOT_ID,"detected_beijing":datetime.now(timezone(timedelta(hours=8))).isoformat(),"old_predictions_never_merged":True}
 run=ROOT/job["run_dir"];write(run/"interruption.json",record)
 path=REPORT/"interrupted_attempts.json";history=json.loads(path.read_text()) if path.exists() else [];write(path,history+[record])
class AttachedProcess:
 def __init__(self,pid,run,phase):self.pid,self.run,self.phase=pid,run,phase
 def poll(self):
  proc=Path("/proc")/str(self.pid)
  try:
   if proc.exists() and (proc/"stat").read_text().split()[2]!="Z" and str(self.run).encode() in (proc/"cmdline").read_bytes():return None
  except FileNotFoundError:pass
  proof=self.run/("gpu_preflight.json" if self.phase=="preflight" else "run_status.json")
  if not proof.exists():return 1
  v=json.loads(proof.read_text());return 0 if v.get("passed") or v.get("status")=="completed" else 1
policy=json.loads((HERE/"execution_policy.json").read_text());assert policy["training_enabled"]
lock=(HERE/".queue.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest=ROOT/policy["manifest"];assert hashlib.sha256(manifest.read_bytes()).hexdigest()==policy["manifest_sha256"]
runid=HERE.name+"_"+datetime.now().strftime("%Y%m%d_%H%M%S");active={};jobs={n:{"status":"adapter_pending","completed_stage":0,"seed":42} for n in METHODS}
old=REPORT/"run_status.json"
if old.exists():
 prior=json.loads(old.read_text());assert prior.get("manifest_sha256")==policy["manifest_sha256"]
 for name,job in prior.get("jobs",{}).items():
  if job["status"] in ["completed","failed"]:
   jobs[name]=job;continue
  if job["status"] not in ["running","gpu_preflight"]:continue
  pid=job["pid"];proc=Path("/proc")/str(pid)
  if prior.get("boot_id") and prior["boot_id"]!=BOOT_ID:
   record_interruption(name,job,"host_reboot");continue
  if not proc.exists():
   record_interruption(name,job,"process_missing");continue
  run=ROOT/job["run_dir"];assert proc.stat().st_uid==os.getuid() and str(run).encode() in (proc/"cmdline").read_bytes()
  phase="training" if job["status"]=="running" else "preflight";handle=(run/"logs/supervisor_adoption.log").open("a");jobs[name]=job
  active[name]=(AttachedProcess(pid,run,phase),handle,run,job["gpu"],phase)
(HERE/"launcher.pid").write_text(str(os.getpid())+"\n")
while True:
 registry=json.loads((HERE/"method_registry.json").read_text())
 for n in METHODS:
  if jobs[n]["status"]=="adapter_pending" and registry[n]["status"]=="ready":jobs[n]["status"]="waiting_gpu"
 for name,(proc,h,run,gpu,phase) in list(active.items()):
  progress=run/"run_status.json"
  if progress.exists():jobs[name]["completed_stage"]=json.loads(progress.read_text()).get("completed_stage",0)
  code=proc.poll()
  if code is None:continue
  h.close();del active[name]
  if code:jobs[name].update(status="failed",exit_code=code);continue
  if phase=="preflight":
   proof=json.loads((run/"gpu_preflight.json").read_text());assert proof["passed"]
   proc,h=spawn(name,run,gpu,False);active[name]=(proc,h,run,gpu,"training");jobs[name].update(status="running",pid=proc.pid,preflight_passed=True)
  else:
   completed=progress.exists() and json.loads(progress.read_text()).get("status")=="completed"
   jobs[name].update(status="completed" if completed else "failed")
   if completed:
    from collect_report import collect
    collect()
    log=(run/"logs/plots.log").open("a",encoding="utf-8")
    subprocess.Popen([str(ROOT/".plot-env/bin/python"),str(HERE/"plot_results.py"),"--run-dir",str(run)],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT)
    log.close();jobs[name]["figures_requested"]=True
 inventory=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.total,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
 for i,total,used,util in csv.reader(io.StringIO(inventory)):
  gpu=int(i)
  if gpu not in policy.get("allowed_gpus",[1,2,3]):continue
  if gpu in {v[3] for v in active.values()}:continue
  name=next((n for n in METHODS if jobs[n]["status"]=="waiting_gpu"),None)
  if not name:break
  admission=policy.get("low_load_gpu_admission",{})
  if int(used)>=1024:
   if name not in admission.get("methods",[]) or int(total)-int(used)<admission.get("minimum_free_mib",19000) or int(util)>=admission.get("max_existing_util_percent",10):continue
  elif int(util)>=10:continue
  run=HERE/"runs"/name/"seed_42"/runid;(run/"logs").mkdir(parents=True,exist_ok=False)
  write(run/"split_identity.json",{"experiment_id":HERE.name,"manifest_sha256":policy["manifest_sha256"],"split_lock":policy["split_lock"],"seed":42,"training_order_shared":True})
  proc,h=spawn(name,run,gpu,True);active[name]=(proc,h,run,gpu,"preflight");jobs[name].update(status="gpu_preflight",gpu=gpu,pid=proc.pid,run_dir=str(run.relative_to(ROOT)))
 status={"experiment_id":HERE.name,"experiment_budget_id":policy["experiment_budget_id"],"manifest_sha256":policy["manifest_sha256"],"split_lock":policy["split_lock"],"updated_unix":time.time(),"boot_id":BOOT_ID,"phase":"parallel_preflight_or_training","launcher_pid":os.getpid(),"task_counts":{"train":100,"dev":0,"test":50},"seed":42,"save_checkpoints":False,"allowed_gpus":policy.get("allowed_gpus",[1,2,3]),"reserved_gpus":policy.get("reserved_gpus",[0]),"training_started":any(x["status"]=="running" for x in jobs.values()),"jobs":jobs}
 write(REPORT/"run_status.json",status);write(REPORT/"comparison_status.json",{"experiment_id":HERE.name,"paper_methods":jobs,"manifest_sha256":policy["manifest_sha256"]})
 lines=["# 本轮并发基线状态","",f"实验：{HERE.name}；同一划分 SHA-256：`{policy['manifest_sha256']}`。","", "GPU0保留；仅GPU1/2/3并发。每方法独立完成 T1→T10，阶段前／后评测全部十任务。100 训练／50 测试；seed42；无验证、无 checkpoint。","", "| 方法 | 状态 | GPU | 完成阶段 |","|---|---|---|---|"]
 lines += [f"| {n} | {v['status']} | {v.get('gpu', '')} | {v['completed_stage']} |" for n,v in jobs.items()]
 lines += ["","adapter_pending 不表示已排队或已启动；只有保留作者机制、通过新任务 GPU 预检的方法才能正式训练。"]
 for n in ["report.md","comparison_report.md"]:(REPORT/n).write_text("\n".join(lines)+"\n")
 from collect_report import collect
 collect()
 if all(job["status"]=="completed" for job in jobs.values()):break
 time.sleep(15)
