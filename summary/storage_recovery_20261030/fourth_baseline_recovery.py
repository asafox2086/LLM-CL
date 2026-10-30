from pathlib import Path
import os,json,hashlib,shutil,math,subprocess,time,fcntl,traceback
R=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_');F=R/'exp/two_backbone_pct10_native_checkpoint_20261027';P=Path('/data2/liyapeng_grp/program/LLMCL/.deployment/storage_recovery_20261030_fourth');O=R/'summary/runtime_recovery_20261030/FOURTH_BASELINE_RECOVERY';PY=Path('/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python')
J=[('PaRSP',18,0,'recovery_quota_third_S13_20261029','0b704968b1314595f5382464a1acc6cdaabc600da6a9e560692583040d992d06'),('OSFT',18,2,'recovery_quota_third_S07_20261029','6ed3426adf03e25a885d75b518221d005af1b8d20b139b812aa33cdb3bb12897')]
def read(p):return json.loads(p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def ident(pid):
 try:
  p=Path('/proc')/str(pid);x=(p/'stat').read_text().rsplit(') ',1)[1].split();return None if x[0]=='Z' else dict(pid=pid,startticks=x[19],uid=p.stat().st_uid)
 except FileNotFoundError:return None
def copy(a,b):
 h=sha(a);b.parent.mkdir(parents=True,exist_ok=True)
 with a.open('rb') as x,b.open('xb') as y:shutil.copyfileobj(x,y,8*1024**2);y.flush();os.fsync(y.fileno())
 assert sha(a)==h==sha(b);return h
def main():
 assert os.getuid()==1084 and not O.exists() and not P.exists()
 handles=[]
 for p in [R/'summary/two_backbone_pct10_20261025/dispatch_old.lock',F/'.pipeline_old_pct10_onevision.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(h)
 sources={p.name:sha(p) for p in F.glob('*.py')};assert len(sources)==25
 policy=read(F/'execution_policy.json');manifest=read(R/policy['conditions']['pct10']['manifest']);prepared=[]
 for method,k,gpu,previous,expected in J:
  u=F/'runs/onevision/pct10'/method/previous;n=F/'runs/onevision/pct10'/method/('recovery_quota_fourth_S%02d_20261030'%k)
  assert not n.exists();c=read(u/'config.json');m=read(u/'checkpoints/latest.json')
  assert c['execution_source_sha256']==sources and c['method']==method and c['backbone']=='onevision' and c['condition']=='pct10'
  assert m==read(u/f'checkpoints/stage_{k:02d}.json') and m['stage']==k and m['sha256']==expected
  assert sha(u/m['path'])==expected and (u/m['path']).stat().st_size==m['bytes']
  assert read(u/f'stages/stage_{k:02d}.json')['complete']
  assert not subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  for pp in Path('/proc').glob('[0-9]*'):
   try:
    if pp.stat().st_uid==1084:assert str(u).encode() not in (pp/'cmdline').read_bytes().split(b'\0')
   except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  events=[];counts={};raw=(u/'logs/events.jsonl').read_bytes()
  for line in raw.splitlines(keepends=True):
   e=json.loads(line);assert e['event']=='train_update'
   if e['stage']<=k:events.append(line);counts.setdefault(e['stage'],[]).append(e['step'])
  for t in range(1,k+1):assert counts[t]==list(range(1,math.ceil(manifest['tasks'][t-1]['files']['train']['count']/16)+1))
  resource=(u/'resources.jsonl').read_bytes()
  for line in resource.splitlines():json.loads(line)
  prepared.append(dict(method=method,k=k,gpu=gpu,u=u,n=n,c=c,m=m,events=b''.join(events),raw=raw,resources=resource))
 O.mkdir();P.mkdir();write(O/'INTENT.json',dict(unix=time.time(),source_sha256=sha(Path(__file__)),sources=sources,originals=[str(x['u']) for x in prepared],quota=subprocess.run(['quota','-v'],capture_output=True,text=True).stdout))
 for x in prepared:
  u,n,k=x['u'],x['n'],x['k'];n.mkdir();cp=P/x['method']/'checkpoints';cp.mkdir(parents=True);(n/'checkpoints').symlink_to(cp,target_is_directory=True)
  copy(u/'config.json',n/'config.json')
  for dirname in ['runner_source','pause_history','prior_recovery_lineage']:
   for p in (u/dirname).rglob('*'):
    if p.is_file():copy(p,n/p.relative_to(u))
  for p in u.glob('*RECOVERY*.json'):copy(p,n/'prior_recovery_lineage'/('previous_'+p.name))
  for t in range(k+1):
   if t:
    for rel in [f'stages/stage_{t:02d}.json',f'validity/stage_{t:02d}.json',f'checkpoints/stage_{t:02d}.json']:copy(u/rel,n/rel)
   for tid in (range(1,20) if t==0 else (t,t+1)):
    for suffix in ['jsonl','scores.json','datasets.json']:copy(u/f'evaluations/stage_{t:02d}/T{tid:02d}_test_predictions.{suffix}',n/f'evaluations/stage_{t:02d}/T{tid:02d}_test_predictions.{suffix}')
  target=n/x['m']['path'];os.link(u/x['m']['path'],target);assert target.stat().st_ino==(u/x['m']['path']).stat().st_ino
  copy(u/'checkpoints/latest.json',n/'checkpoints/latest.json')
  (n/'logs').mkdir();(n/'logs/events.jsonl').write_bytes(x['events']);(n/'resources.jsonl').write_bytes(x['resources'])
  write(n/'run_status.json',dict(status='authorized_quota_recovery',completed_stage=k,original_run=str(u)))
  write(n/'QUOTA_RECOVERY_SOURCE.json',dict(original_run=str(u),stage=k,checkpoint=x['m'],original_events_sha256=hashlib.sha256(x['raw']).hexdigest(),canonical_events_sha256=sha(n/'logs/events.jsonl'),source_sha256=sources,resources_sha256=sha(n/'resources.jsonl'),original_unchanged=True,not_cooperative_pause=True,persistent_checkpoints=str(cp)))
 write(O/'PREPARED.json',dict(runs=[str(x['n']) for x in prepared],no_scientific_changes=True))
 children=[]
 try:
  for x in prepared:
   assert not subprocess.check_output(['nvidia-smi',f"--id={x['gpu']}",'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   env=dict(os.environ,LLMCL_ROOT=str(R),LLMCL_HOST='old_pct10_onevision',LLMCL_BACKBONE='onevision',CUDA_VISIBLE_DEVICES=str(x['gpu']),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1')
   cmd=[str(PY),'-u',str(F/'run.py'),'--method',x['method'],'--condition','pct10','--run-dir',str(x['n']),'--resume']
   with (O/(x['method']+'.log')).open('x') as log:p=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   children.append((x,p));write(O/(x['method']+'_STARTED.json'),dict(identity=ident(p.pid),GPU=x['gpu'],command=cmd,run=str(x['n'])))
 except BaseException:write(O/'LAUNCH_FAILURE.json',dict(traceback=traceback.format_exc(),started=[p.pid for x,p in children]))
 pending=list(children)
 while pending:
  for x,p in list(pending):
   c=p.poll()
   if c is not None:write(O/(x['method']+'_EXIT.json'),dict(unix=time.time(),returncode=c,identity_absent=ident(p.pid) is None,run=str(x['n'])));pending.remove((x,p))
  if pending:time.sleep(30)
if __name__=='__main__':main()
