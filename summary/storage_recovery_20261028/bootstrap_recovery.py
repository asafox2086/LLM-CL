"""One explicit bootstrap under the ORIGINAL owner's locks. Default inspect.
--execute invokes original pipeline.launch(resume=True) for two exact committed
CP branches and persists only their job identities. NEVER restart systemd,
allocate via another loop, signal a GPU process, or auto-kill partial launches.
"""
from pathlib import Path
import argparse,copy,fcntl,hashlib,importlib.util,json,os,subprocess,time,traceback
ROOT=Path('/data1/liyapeng/program/LLMCL');REPORT=ROOT/'summary/storage_recovery_20261028'
PREPARED=REPORT/'BRANCH_PREPARATION/PREPARED.json';PSHA='9f356ac3fba318b05c063988571c88ae5ff98d72ae1e8271d35b76e19eec5fe4'
SUPPLEMENT=REPORT/'RESOURCE_SUPPLEMENT.json';SSHA='28a87ef8021e136f84ccfdf053fc3b6b80775a2b7ab15930165cd85991280256'
FRAME=ROOT/'exp/two_backbone_native_checkpoint_20261027';OLD=ROOT/'exp/two_backbone_native_cl_20261017_v2'
PCT10=ROOT/'exp/two_backbone_pct10_native_checkpoint_20261027';STATES=ROOT/'summary/two_backbone_native_cl_20261017_v2'
OUT=REPORT/'BOOTSTRAP';PIPE_SHA='2b18ce445a01d76b2f065100752c05626bdaf800dd33f47aa8c6e75ff52f9d2d'
UNIT='llmcl-two-backbone-native.service'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def ident(pid):
 try:
  p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return None if f[0]=='Z' else dict(pid=int(pid),startticks=f[19],uid=p.stat().st_uid,pgid=int(f[2]))
 except FileNotFoundError:return None
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def show():
 env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/1004',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1004/bus')
 p=subprocess.run(['systemctl','--user','show',UNIT,'-p','MainPID','-p','ActiveState','-p','SubState','-p','KillMode','-p','Restart'],env=env,capture_output=True,text=True);assert p.returncode==0,p.stderr
 v=dict(x.split('=',1) for x in p.stdout.splitlines());assert v['MainPID']=='0' and v['ActiveState'] in ('inactive','failed')
 assert v['KillMode']=='process' and v['Restart']=='no' and ident(122374) is None
 return v
def idle(gpu):
 text=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
 fields=[x.strip() for x in text.split(',')];assert len(fields)==3 and fields[0].startswith('GPU-') and int(fields[1])<500 and int(fields[2])<10
 apps=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15).strip();assert not apps,(gpu,apps)
 return dict(GPU=gpu,UUID=fields[0],memory_MiB=int(fields[1]),util=int(fields[2]))
def metadata():
 assert sha(PREPARED)==PSHA and sha(SUPPLEMENT)==SSHA
 prepared=read(PREPARED);assert len(prepared['prepared'])==2
 rows=[];protected={str(PREPARED):PSHA,str(SUPPLEMENT):SSHA}
 for item in prepared['prepared']:
  j=item['job'];run=Path(item['branch']);assert run.resolve()==run and run.is_relative_to(FRAME/'runs')
  proof=run/'IO_RECOVERY_SOURCE.json';assert sha(proof)==item['proof_sha256'];v=read(proof);cfg=read(run/'config.json')
  sources={p.name:sha(p) for p in FRAME.glob('*.py')};assert sources==cfg['execution_source_sha256']==v['source_sha256']
  assert sha(run/'config.json')==v['config_sha256'] and read(run/'run_status.json')['status']=='io_recovery_prepared'
  m=read(run/'checkpoints/latest.json');assert m==v['checkpoint'] and m['stage']==j['stage'] and m['sha256']==j['sha256']
  cp=run/m['path'];original=Path(v['original_run'])/m['path']
  a,b=cp.stat(),original.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,j['bytes'],b.st_mtime_ns)
  assert read(run/f"stages/stage_{j['stage']:02d}.json")['complete']
  note=read(run/'IO_RECOVERY_RESOURCE_SUPPLEMENT.json');assert sha(run/'resources.jsonl')==note['sha256']
  assert sha(run/'logs/events.jsonl')==v['canonical_events_sha256']
  for name in ('PAUSE_REQUEST.json','RESUME_REQUEST.json'):assert not (run/name).exists()
  statepath=STATES/(j['host']+'_run_status.json');state=read(statepath);job=state['jobs'][j['key']]
  assert job['method']==j['method'] and job['condition']=='pct01' and job['host']==j['host']
  assert job['run_dir']==str((FRAME/j['original']).relative_to(ROOT))
  for field in ('pid','worker','wrapper','capacity_identity'):
   who=job.get(field)
   if who:assert ident(who if isinstance(who,int) else who['pid']) is None
  protected[str(statepath)]=sha(statepath);protected[str(proof)]=sha(proof)
  rows.append(dict(item=item,job=j,run=run,statepath=statepath,state=state,old_job=copy.deepcopy(job),proof=proof))
 assert {x['job']['gpu'] for x in rows}=={1,2}
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=1004:continue
   argv=(proc/'cmdline').read_bytes().split(b'\0')
   assert not any(str(x['run']).encode() in argv for x in rows)
  except (FileNotFoundError,ProcessLookupError):continue
 assert sha(FRAME/'pipeline.py')==PIPE_SHA
 return rows,protected
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 assert os.getuid()==1004 and ROOT.resolve()==ROOT and not OUT.exists()
 before=show();rows,protected=metadata();cards=[idle(1),idle(2)]
 if not a.execute:print(json.dumps(dict(inspection_only=True,owner=before,cards=cards,branches=[str(x['run']) for x in rows],writes=0)));return
 paths=[STATES/'baseline_pct01_capacity.lock',STATES/'hardware_queue.lock',OLD/'.pipeline_new3090.lock',OLD/'.pipeline_new3090_onevision.lock',
  ROOT/'summary/two_backbone_pct10_20261025/dispatch_new.lock',PCT10/'.pipeline_new_pct10_onevision.lock',PCT10/'.pipeline_new_pct10_qwen.lock']
 handles=[]
 for path in paths:
  assert path.is_file() and not path.is_symlink();h=path.open('r');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(h)
 assert show()==before;rows2,protected2=metadata();assert protected2==protected
 assert [idle(1),idle(2)]==cards
 OUT.mkdir(exist_ok=False);started=[];processes=[]
 try:
  write(OUT/'INTENT.json',dict(unix=time.time(),script_sha256=sha(__file__),owner=before,cards=cards,protected_sha256=protected,
   jobs=[dict(host=x['job']['host'],key=x['job']['key'],old_job=x['old_job'],branch=str(x['run'])) for x in rows],
   locks=[str(p) for p in paths],no_systemd_start=True,no_GPU_signals=True))
  # Preserve exact pre-bootstrap state bytes, including any existing tmp file.
  for x in rows:
   (OUT/(x['job']['host']+'_BEFORE.json')).write_bytes(x['statepath'].read_bytes())
   tmp=x['statepath'].with_suffix('.json.tmp')
   if tmp.exists():(OUT/(x['job']['host']+'_PREEXISTING_TMP.json')).write_bytes(tmp.read_bytes())
  for x in rows:
   show();j=x['job'];idle(j['gpu']);assert sha(x['statepath'])==protected[str(x['statepath'])]
   os.environ.update(LLMCL_ROOT=str(ROOT),LLMCL_HOST=j['host'],LLMCL_BACKBONE=j['backbone'])
   spec=importlib.util.spec_from_file_location('IO_recovery_original_pipeline_'+j['backbone'],FRAME/'pipeline.py');pipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipe)
   assert Path(pipe.HERE)==FRAME and pipe.HOST==j['host'] and pipe.BACKBONE==j['backbone']
   assert Path(pipe.PYTHON)==ROOT/'.vision-env/bin/python'
   job=copy.deepcopy(x['old_job'])
   for name in ('pid','capacity_identity','worker','wrapper','phase_exit','phase_status'):job.pop(name,None)
   job.update(run_dir=str(x['run'].relative_to(ROOT)),completed_stage=j['stage'],IO_recovery_source_sha256=sha(x['proof']),
    IO_recovery_original_run=x['old_job']['run_dir'])
   child=pipe.launch(job,'training',j['gpu'],resume=True);processes.append(child)
   who=ident(child.pid);assert who and who['pid']==who['pgid'] and who['uid']==1004
   record=dict(host=j['host'],key=j['key'],worker=who,GPU=j['gpu'],run_dir=job['run_dir'],CP_stage=j['stage'],CP_sha256=j['sha256'],native_resume=True)
   started.append(record);write(OUT/(j['backbone']+'_CHILD_STARTED.json'),record)
   job['capacity_identity']=who;state=x['state'];other_jobs={k:copy.deepcopy(v) for k,v in state['jobs'].items() if k!=j['key']}
   state['jobs'][j['key']]=job
   assert {k:v for k,v in state['jobs'].items() if k!=j['key']}==other_jobs
   # Atomic write only after a genuine child identity has been durably reported.
   pipe.write(x['statepath'],state);write(OUT/(j['backbone']+'_QUEUE_UPDATED.json'),dict(path=str(x['statepath']),sha256=sha(x['statepath']),worker=who))
  assert all(ident(w['worker']['pid'])==w['worker'] for w in started)
  write(OUT/'COMPLETE.json',dict(unix=time.time(),workers=started,systemd_started=False,allocator_not_running=True,
   GPU_signals=0,other_jobs_unchanged=True,next_step='Operator must start the original fullFT outer unit only after checking these exact identities; original owner adopts training PIDs. Do not execute this bootstrap again.'))
  print(json.dumps(dict(complete=True,workers=started,receipt_sha256=sha(OUT/'COMPLETE.json'),systemd_started=False)))
 except BaseException:
  write(OUT/'FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),started=started,
   Popen_handles=[dict(pid=p.pid,returncode=p.poll()) for p in processes],no_auto_kill=True,no_auto_retry=True,
   action='Inspect actual children and per-child receipts before any coordinator action. Never repeat bootstrap.'))
  raise
if __name__=='__main__':main()
