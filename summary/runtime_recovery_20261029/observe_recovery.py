from pathlib import Path
import subprocess,json,time,shutil
ROOT=Path('/data1/liyapeng/program/LLMCL')
old="""from pathlib import Path
import json,subprocess,shutil,time
r=Path('/dev/shm/llmcl-task-gradient-formal-1084-20261028');d=r/'summary/runtime_recovery_20261029'
def ident(w):
 try:
  p=Path('/proc')/str(w['pid']);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return dict(pid=w['pid'],startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]),state=s[0])
 except (OSError,ValueError,IndexError):return None
v={'unix':time.time(),'tmpfs_free':shutil.disk_usage(r).free}
d=r/'summary/seq_s16_four_domain_final_20261029'
for n in ['WORKER_STARTED.json','EXIT.json']:
 p=d/n;v[n]=json.loads(p.read_text()) if p.exists() else None
start=v['WORKER_STARTED.json']
if start:
 v['worker_actual']=ident(start['identity']);u=Path(start['output'])
 for n in ['GRAPH.json','COMPLETE.json','FAILURE.json']:
  p=u/n;v[n+'_exists']=p.exists()
  if n!='GRAPH.json' and p.exists():v[n]=json.loads(p.read_text())
p=d/'worker.log';last=[]
if p.exists():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-5000));raw=f.read().decode(errors='replace')
 for line in raw.splitlines():
  try:x=json.loads(line)
  except ValueError:continue
  if x.get('event')=='evaluation':last.append({k:x[k] for k in ['event','stage','task','done','total'] if k in x})
v['progress']=last[-1:]
bd=Path('/data2/liyapeng_grp/program/LLMCL/.deployment/storage_recovery_20261029_second')
v['baseline_recovery']={}
for sub,names in [('SUPERVISOR_NATIVE_ENTRY_V2',['PaRSP_STARTED.json','OSFT_STARTED.json','PaRSP_EXIT.json','OSFT_EXIT.json','LAUNCH_FAILURE.json']),('QWEN_PARSP_FRESH',['STARTED.json','EXIT.json'])]:
 for name in names:
  p=bd/sub/name
  if p.exists():
   x=json.loads(p.read_text());entry={'receipt':x}
   if x.get('identity'):entry['actual']=ident(x['identity'])
   job=x.get('job',{});rd=job.get('run_dir')
   if rd:
    q=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_')/rd/'run_status.json'
    if q.exists():entry['native']=json.loads(q.read_text())
   v['baseline_recovery'][sub+'/'+name]=entry
print(json.dumps(v))
"""
c=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','-i',str(ROOT/'.deployment/old_pct01_id_ed25519'),'-p','8666','liyapeng_grp@10.254.29.8','/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python','-']
p=subprocess.run(c,input=old,text=True,capture_output=True,timeout=60)
print(json.dumps({'old_returncode':p.returncode,'old':json.loads(p.stdout) if p.returncode==0 else p.stderr,'new_free':shutil.disk_usage(ROOT).free}))
