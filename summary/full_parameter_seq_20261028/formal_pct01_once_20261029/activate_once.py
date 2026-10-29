from pathlib import Path
import hashlib,json,os,subprocess,time
ROOT=Path('/data1/liyapeng/program/LLMCL')
D=ROOT/'summary/full_parameter_seq_20261028/formal_pct01_once_20261029'
WRAP=ROOT/'analysis/full_parameter_formal_capacity_20261029.py'
UNIT='llmcl-two-backbone-native.service'
ADMISSION=D/'ADMISSION.json'
CAPACITY=ROOT/'summary/two_backbone_native_cl_20261017_v2/baseline_pct01_capacity_status.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 s=(p/'stat').read_text().rsplit(') ',1)[1].split()
 return None if s[0]=='Z' else dict(pid=pid,startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
def call(args):return subprocess.run(args,check=True,text=True,capture_output=True)
def exclusive(name,value):
 with (D/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def unit_state():
 return dict(line.split('=',1) for line in call(['systemctl','--user','show',UNIT,'--property=MainPID,ActiveState,SubState']).stdout.splitlines())
def safe_sleep(old,workers):
 # Actual NEW observation: S / poll_schedule_timeout.constprop.0. This Python
 # capacity loop sleeps with select; do not accept an arbitrary blocked state.
 deadline=time.monotonic()+30
 while True:
  assert ident(old['pid'])==old and all(ident(int(p))==v for p,v in workers.items())
  proc=Path('/proc')/str(old['pid']);parts=(proc/'stat').read_text().rsplit(') ',1)[1].split();wchan=(proc/'wchan').read_text().strip()
  if parts[0]=='S' and wchan=='poll_schedule_timeout.constprop.0':
   time.sleep(.1)
   after=(proc/'stat').read_text().rsplit(') ',1)[1].split();again=(proc/'wchan').read_text().strip()
   if ident(old['pid'])==old and after[0]=='S' and again==wchan:
    return dict(unix=time.time(),state='S',wchan=wchan,two_samples_seconds=.1)
  if time.monotonic()>=deadline:raise TimeoutError('CPU did not reach the observed safe sleep within30s; no stop issued')
  time.sleep(.2)

assert not (D/'CPU_HANDOFF.json').exists() and not (D/'ACTIVATION.json').exists()
assert sha(WRAP)=='369c67cc22cb53626563744b6bc4f4cf193d06d87e35824b9a6d563dd4558800'
assert sha(ADMISSION)=='aa9656e711a49717334a4ce9752b6f4109259ce100f6d41ae8bac3d359bf4a75'
check=call([str(ROOT/'.assets-env/bin/python'),str(WRAP),'--formal-admission',str(ADMISSION),'--formal-inspect'])
props=dict(line.split('=',1) for line in call(['systemctl','--user','show',UNIT,'--property=KillMode,MainPID,FragmentPath']).stdout.splitlines())
assert props['KillMode']=='process' and props['MainPID']=='3217671'
old=ident(3217671);assert old==dict(pid=3217671,startticks='61619749',uid=1004,pgid=3217671)
workers={str(p):ident(p) for p in (3217693,3217709)}
assert workers=={'3217693':dict(pid=3217693,startticks='61619775',uid=1004,pgid=3217693),'3217709':dict(pid=3217709,startticks='61619786',uid=1004,pgid=3217709)}
unit=Path(props['FragmentPath']);assert unit==Path('/data1/liyapeng/.config/systemd/user')/UNIT
text=unit.read_text();assert text.count('ExecStart=')==1 and 'KillMode=process' in text and 'Restart=no' in text
argv=(Path('/proc/3217671/cmdline').read_bytes().rstrip(b'\0').split(b'\0'));argv=[x.decode() for x in argv]
assert argv[2]==str(ROOT/'summary/storage_recovery_second_20261029/owner_recovery.py') and argv[-1]=='--watch'
new=[*argv[:2],str(WRAP),'--formal-admission',str(ADMISSION),*argv[3:]]
assert all(' ' not in x and '\n' not in x for x in new)
line=next(s for s in text.splitlines() if s.startswith('ExecStart='));assert line=='ExecStart='+' '.join(argv)
newtext=text.replace(line,'ExecStart='+' '.join(new))
sleep_observation=safe_sleep(old,workers)
exclusive('CPU_HANDOFF.json',dict(unix=time.time(),old_controller=old,workers_before=workers,old_unit=text,new_unit=newtext,admission_sha256=sha(ADMISSION),safe_sleep=sleep_observation,GPU_signals=0))
# Recheck immediately before the only stop; no signalling on an identity change.
assert ident(old['pid'])==old and unit_state()['MainPID']==str(old['pid'])
assert all(ident(int(p))==v for p,v in workers.items())
assert (Path('/proc')/str(old['pid'])/'wchan').read_text().strip()==sleep_observation['wchan']
# KillMode=process stops only the CPU capacity owner. Its GPU workers remain.
call(['systemctl','--user','stop',UNIT]);assert ident(old['pid'])!=old
assert all(ident(int(p))==v for p,v in workers.items())
tmp=unit.with_suffix('.service.formal.tmp');tmp.write_text(newtext);os.replace(tmp,unit)
call(['systemctl','--user','daemon-reload']);call(['systemctl','--user','start',UNIT])
pid=int(call(['systemctl','--user','show',UNIT,'--property=MainPID','--value']).stdout.strip())
newwho=ident(pid);assert newwho and newwho['uid']==1004 and newwho['pgid']==pid
assert all(ident(int(p))==v for p,v in workers.items())
actual=Path('/proc')/str(pid)/'cmdline';assert actual.read_bytes().rstrip(b'\0').split(b'\0')==[v.encode() for v in new]
deadline=time.monotonic()+30
while True:
 assert ident(pid)==newwho and all(ident(int(p))==v for p,v in workers.items())
 state=unit_state();raw=None;capacity=None
 try:raw=CAPACITY.read_bytes();capacity=json.loads(raw)
 except (OSError,json.JSONDecodeError):pass
 if (state['MainPID']==str(pid) and state['ActiveState']=='active' and state['SubState']=='running'
     and capacity is not None and capacity.get('CPU_pid')==pid):
  active=list(capacity.get('active',{}).values())
  if all(any(all(item.get(k)==v for k,v in expected.items()) for item in active) for expected in workers.values()):break
 if time.monotonic()>=deadline:raise TimeoutError('New owner did not publish adoption under its original capacity lock within30s')
 time.sleep(.2)
assert ident(pid)==newwho and unit_state()==state
assert all(ident(int(p))==v for p,v in workers.items())
value=dict(unix=time.time(),new_controller=newwho,old_controller_gone=True,unit=UNIT,unit_state=state,unit_sha256=sha(unit),workers_unchanged=workers,admission_sha256=sha(ADMISSION),wrapper_sha256=sha(WRAP),GPU_signals=0,formal_slot_created=(D/'SLOT.json').exists(),capacity_state_path=str(CAPACITY),capacity_state_sha256=hashlib.sha256(raw).hexdigest(),capacity_state_unix=capacity['unix'],capacity_CPU_pid=capacity['CPU_pid'],capacity_active=capacity['active'],status='single_original_owner_formal_offer_activated')
exclusive('ACTIVATION.json',value);print(json.dumps(value))
