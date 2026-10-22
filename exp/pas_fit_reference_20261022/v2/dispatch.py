"""One new GPU1 fresh PASs reference, no retries after STARTED or full TEST."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from core import ROOT,REPORT,WORK,sha,write
from common import check_ready,verify

QUEUE=REPORT/'queue'
UNIT='llmcl-onevision-pas-fit-reference-20261022-v2.service'
ENTRY=Path(__file__).relative_to(ROOT).as_posix()

def identity(pid):
    p=Path('/proc')/str(pid)
    try:return dict(startticks=(p/'stat').read_text().rsplit(') ',1)[1].split()[19],uid=p.stat().st_uid,
                    cmdline=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode())
    except (FileNotFoundError,ProcessLookupError):return None

def inventory():
    sys.path.insert(0,str(ROOT/'analysis'));import onevision_research_device_entry as device
    return device.inventory()

def available(cards):
    ours={g for g,v in cards.items() if any(p['uid']==os.getuid() for p in v['owners'])};c=cards[1]
    return not (ours&{0,1}) and len(ours|{1})<=3 and not c['owners'] and c['memory_MiB']<500 and c['utilization']<10

def watch():
    QUEUE.mkdir(parents=True,exist_ok=True)
    lock=(QUEUE/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_ready()
    assert not any((REPORT/n).exists() for n in ['STARTED.json','FAILURE.json','COMPLETE.json','VERIFIED.json']) and not WORK.exists()
    while True:
        if not available(inventory()):
            write(QUEUE/'state.json',dict(phase='wait',pid=os.getpid(),only_GPU=1,one_condition=True,TEST_QA=0))
            time.sleep(30);continue
        with (QUEUE/'worker_nohup.log').open('ab') as log:
            worker=subprocess.Popen([str(ROOT/'.vision-env/bin/python'),'-u',str(Path(__file__).with_name('run.py'))],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'))
        info=identity(worker.pid);assert info
        write(QUEUE/'state.json',dict(phase='running',pid=os.getpid(),worker_pid=worker.pid,
            worker_startticks=info['startticks'],only_GPU=1,TEST_QA=0))
        while worker.poll() is None:time.sleep(30)
        code=worker.wait()
        if code==75 and not (REPORT/'STARTED.json').exists():continue
        assert code==0,'Worker failed; preserve and never automatically retry'
        begin=time.monotonic();proof=verify();proof['CPU_verification_seconds']=time.monotonic()-begin
        write(REPORT/'VERIFIED.json',proof)
        write(QUEUE/'state.json',dict(phase='complete_verified',pid=os.getpid(),worker_exit=code,
            verified_sha256=sha(REPORT/'VERIFIED.json'),only_GPU=1,automatic_full_confirmation=False))
        return

def activate():
    check_ready();QUEUE.mkdir(parents=True,exist_ok=True)
    assert not (QUEUE/'ACTIVATION_PUBLIC.json').exists()
    assert not any((REPORT/n).exists() for n in ['STARTED.json','FAILURE.json','COMPLETE.json'])
    cards=inventory();pids={o['pid'] for c in cards.values() for o in c['owners']}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            info=identity(int(p.name))
            if info and info['uid']==os.getuid() and str(ROOT) in info['cmdline'] and ENTRY not in info['cmdline']:pids.add(int(p.name))
        except PermissionError:pass
    protected={str(p):identity(p) for p in pids if identity(p)}
    audit=Path(__file__).parent/'.activation_private';audit.mkdir(exist_ok=False)
    write(audit/'BEFORE.json',dict(protected=protected,inventory=cards))
    unit=Path.home()/'.config/systemd/user'/UNIT;assert not unit.exists()
    unit.write_text('[Unit]\nDescription=One fresh fixed PAS reference v2 after sealed external reboot\n[Service]\nType=simple\nWorkingDirectory='+str(ROOT)+
        '\nExecStart=/usr/bin/nohup '+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/ENTRY)+' --watch\nRestart=no\nKillMode=process\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True))
    own=identity(pid);assert own and ENTRY in own['cmdline']
    assert {p:identity(int(p)) for p in protected}==protected,'Protected identity changed; inspect, do not restart GPUs'
    write(QUEUE/'ACTIVATION_PUBLIC.json',dict(pid=pid,startticks=own['startticks'],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        protected={p:{k:v[k] for k in ['startticks','uid']} for p,v in protected.items()},
        ready_sha256=sha(REPORT/'READY.json'),only_GPU=1,not_enabled_on_boot=True,raw_commandlines_private=True))
    print(json.dumps(dict(activated=True,pid=pid,startticks=own['startticks'],no_GPU_interruption=True)))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--activate',action='store_true');a=p.parse_args()
    try:
        if a.watch:watch()
        elif a.activate:activate()
        else:raise ValueError('Explicit --watch or --activate required')
    except Exception:write(QUEUE/'needs_attention.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
