"""Queue-only explicit IO recovery, default inspect. No GPU/model/service starts.
Requires prepare_S14_branches --execute success first. Only two S14 resume jobs
and zero-update PAS fresh restart are changed; original failed RUNs stay intact.
"""
from pathlib import Path
import argparse,copy,fcntl,hashlib,importlib.util,json,os,subprocess,time,traceback
ROOT=Path('/data1/liyapeng/program/LLMCL');REPORT=ROOT/'summary/storage_recovery_20261029'
FRAME=ROOT/'exp/two_backbone_native_checkpoint_20261027';OLD=ROOT/'exp/two_backbone_native_cl_20261017_v2';PCT10=ROOT/'exp/two_backbone_pct10_native_checkpoint_20261027'
STATES=ROOT/'summary/two_backbone_native_cl_20261017_v2';OUT=REPORT/'QUEUE_ENQUEUE'
PAS=FRAME/'runs/qwen/pct01/PASs_MoE/training_20261028_223446'
OWNER=REPORT/'owner_recovery.py';OWNER_SHA='8db6c0ed9b256a3dd462612ecd4b050980da2619fa9b831bd26c5077575466c1'
FULL=ROOT/'analysis/full_parameter_preflight_capacity_v2_20261028.py';FULL_SHA='03fe7f0028ab4b8df479bc997aab7011f413eee6e47c4023fed26dd4442c9d56'
PREPARED=REPORT/'BRANCH_PREPARATION/PREPARED.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def absent(pid):
 p=Path('/proc')/str(pid)/'stat'
 return not p.exists() or p.read_text().rsplit(') ',1)[1].split()[0]=='Z'
def owner_gone():
 e=dict(os.environ,XDG_RUNTIME_DIR='/run/user/1004',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1004/bus')
 p=subprocess.run(['systemctl','--user','show','llmcl-two-backbone-native.service','-p','MainPID','-p','ActiveState','-p','KillMode','-p','Restart'],env=e,text=True,capture_output=True);assert p.returncode==0
 v=dict(x.split('=',1) for x in p.stdout.splitlines());assert v['MainPID']=='0' and v['ActiveState'] in ('inactive','failed') and v['Restart']=='no' and v['KillMode']=='process'
 assert all(absent(pid) for pid in (797412,790980,782824,782837))
 return v
def check():
 assert sha(OWNER)==OWNER_SHA and sha(FULL)==FULL_SHA
 prepared=read(PREPARED);assert len(prepared['prepared'])==2
 paths={host:STATES/(host+'_run_status.json') for host in ('new3090','new3090_onevision')}
 states={h:read(p) for h,p in paths.items()};before=copy.deepcopy(states);changes=[];branches=[]
 sources={p.name:sha(p) for p in FRAME.glob('*.py')}
 for item in prepared['prepared']:
  j=item['job'];run=Path(item['branch']);proofpath=run/'IO_RECOVERY_SOURCE.json';assert sha(proofpath)==item['proof_sha256']
  proof=read(proofpath);assert proof['CP_SHA_verified'] and proof['checkpoint']['stage']==14 and proof['source_sha256']==sources
  assert read(run/'run_status.json')['status']=='io_recovery_prepared' and read(run/'checkpoints/latest.json')==proof['checkpoint']
  cp=run/proof['checkpoint']['path'];assert cp.stat().st_size==proof['checkpoint']['bytes']
  assert sha(run/'config.json')==proof['config_sha256'] and sha(run/'logs/events.jsonl')==proof['canonical_events_sha256']
  assert sha(run/'resources.jsonl')==proof['files_sha256']['resources.jsonl']
  job=states[j['host']]['jobs'][j['key']];assert job['run_dir']==str((FRAME/j['original']).relative_to(ROOT))
  assert job['method']==j['method'] and job['condition']=='pct01'
  for field in ('pid','worker','wrapper','capacity_identity'):
   who=job.get(field)
   if who:assert absent(who if isinstance(who,int) else who['pid'])
  for field in ('pid','worker','wrapper','capacity_identity','phase_exit','phase_status'):job.pop(field,None)
  job.update(status='ready_resume',run_dir=str(run.relative_to(ROOT)),completed_stage=14,
   IO_recovery_source_sha256=item['proof_sha256'],IO_recovery_original_run=j['original'],IO_recovery_reason='User-authorized recovery after ENOSPC; not a cooperative pause')
  changes.append((j['host'],j['key']));branches.append(dict(run=str(run),proof_sha256=item['proof_sha256']))
 pstate=read(PAS/'run_status.json');pcfg=read(PAS/'config.json')
 assert pstate['status']=='initial_evaluation' and pstate['completed_stage']==0
 assert pcfg['method']=='PASs_MoE' and pcfg['condition']=='pct01' and pcfg['execution_source_sha256']==sources
 assert not (PAS/'logs/events.jsonl').exists() or (PAS/'logs/events.jsonl').stat().st_size==0
 assert not list((PAS/'checkpoints').glob('stage_*.pt')) and not list((PAS/'stages').glob('stage_*.json'))
 pj=states['new3090']['jobs']['pct01:PASs_MoE'];assert pj['run_dir']==str(PAS.relative_to(ROOT))
 preflight=ROOT/pj['preflight_run'];gp=read(preflight/'gpu_preflight.json');rp=read(preflight/'checkpoint_reload_proof.json')
 assert gp['passed'] and rp['passed'] and gp['execution_source_sha256']==rp['execution_source_sha256']==sources
 for field in ('pid','worker','wrapper','capacity_identity','phase_exit','phase_status'):pj.pop(field,None)
 pj.update(status='ready_training',completed_stage=0,fresh_restart_reason='User-authorized zero-update ENOSPC recovery; new RUN/W0/initial evaluation; no CP resume',prior_failed_run=str(PAS.relative_to(ROOT)))
 changes.append(('new3090','pct01:PASs_MoE'))
 for h,state in states.items():
  assert set(state['jobs'])==set(before[h]['jobs'])
  assert all(j==before[h]['jobs'][k] for k,j in state['jobs'].items() if (h,k) not in changes)
 fullad=ROOT/'summary/full_parameter_seq_20261028/PREFLIGHT_V2_ADMISSION.json'
 admission=dict(schema='NEW-three-baseline-IO-recovery-20261029',owner_sha256=OWNER_SHA,fullft_outer_sha256=FULL_SHA,
  fullft_admission=dict(path=str(fullad),sha256=sha(fullad)),prepared=dict(path=str(PREPARED),sha256=sha(PREPARED)),resume_branches=branches,
  PAS_fresh_original_run=str(PAS),other_jobs_unchanged=True,science_sources_unchanged=True,GPU0_reserved=True)
 return paths,states,before,changes,admission

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 assert os.getuid()==1004 and not OUT.exists();status=owner_gone();paths,states,before,changes,admission=check()
 pins={str(p):sha(p) for p in paths.values()}
 if not a.execute:print(json.dumps(dict(inspection_only=True,owner=status,changed_jobs=changes,model_starts=0,queue_writes=0)));return
 locks=[STATES/'baseline_pct01_capacity.lock',STATES/'hardware_queue.lock',OLD/'.pipeline_new3090.lock',OLD/'.pipeline_new3090_onevision.lock',ROOT/'summary/two_backbone_pct10_20261025/dispatch_new.lock',PCT10/'.pipeline_new_pct10_onevision.lock',PCT10/'.pipeline_new_pct10_qwen.lock'];handles=[]
 for p in locks:
  assert p.is_file() and not p.is_symlink();f=p.open('r');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
 assert owner_gone()==status and all(sha(Path(p))==h for p,h in pins.items())
 OUT.mkdir(exist_ok=False)
 try:
  write(OUT/'INTENT.json',dict(unix=time.time(),script_sha256=sha(__file__),old_queue_sha256=pins,changed_jobs=changes,owner=status,no_model_launch=True))
  for h,p in paths.items():
   (OUT/(h+'_BEFORE.json')).write_bytes(p.read_bytes())
   for q in (p.with_suffix('.tmp'),p.with_suffix('.json.tmp')):
    if q.exists():(OUT/(h+'_'+q.name+'_OLD_TMP')).write_bytes(q.read_bytes())
  aq=REPORT/'RECOVERY_ADMISSION.json';write(aq,admission)
  spec=importlib.util.spec_from_file_location('source_only_recovery_validate',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  m.validate(aq,sha(aq))
  for h,p in paths.items():
   assert sha(p)==pins[str(p)];temp=p.with_name(p.name+'.io_recovery_20261029.tmp');write(temp,states[h]);os.replace(temp,p)
   fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  write(OUT/'COMPLETE.json',dict(unix=time.time(),changed_jobs=changes,new_queue_sha256={str(p):sha(p) for p in paths.values()},admission_sha256=sha(aq),owner_source_sha256=OWNER_SHA,
   models_started=0,services_started=0,other_jobs_unchanged=True,GPU0_reserved=True,
   next='Root reviews/updates the original unit to owner_recovery.py with --recovery-admission and --recovery-admission-sha256, preserving all existing FullFT/checkpoint/watch args; original idle allocator waits for ownerless GPUs.'))
  print(json.dumps(dict(queue_prepared=True,admission=str(aq),admission_sha256=sha(aq),models_started=0,services_started=0)))
 except BaseException:
  write(OUT/'FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),no_retry=True,models_started=0,services_started=0));raise
if __name__=='__main__':main()
