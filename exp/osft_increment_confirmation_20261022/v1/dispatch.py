"""One new GPU1 confirmation owner; no retries after STARTED, no old service edits."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import ROOT,REPORT,WORK,sha,write,check_ready,verify
QUEUE=REPORT/'queue'
UNIT='llmcl-onevision-full-confirm-20261022.service'
ENTRY=Path(__file__).relative_to(ROOT).as_posix()
def identity(pid):
    p=Path('/proc')/str(pid)
    try:return dict(startticks=(p/'stat').read_text().rsplit(') ',1)[1].split()[19],
        cmdline=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(),uid=p.stat().st_uid)
    except FileNotFoundError:return None

def inventory():
    sys.path.insert(0,str(ROOT/'analysis'))
    import onevision_research_device_entry as d
    return d.inventory()
def available(cards):
    ours={g for g,v in cards.items() if any(p['uid']==os.getuid() for p in v['owners'])};c=cards[1]
    return not (ours&{0,1}) and len(ours|{1})<=3 and not c['owners'] and c['memory_MiB']<500 and c['utilization']<10

def watch():
    QUEUE.mkdir(parents=True,exist_ok=True)
    lock=(QUEUE/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_ready()
    assert not any((REPORT/n).exists() for n in ['STARTED.json','FAILURE.json','COMPLETE.json','VERIFIED.json'])
    assert not WORK.exists(), 'No reuse of partial/full prior attempt'
    while True:
        cards=inventory()
        if not available(cards):
            write(QUEUE/'state.json',dict(phase='wait',pid=os.getpid(),inventory=cards,only_GPU=1,one_fixed_confirmation=True))
            time.sleep(30);continue
        log=(QUEUE/'worker_nohup.log').open('ab')
        worker=subprocess.Popen([str(ROOT/'.vision-env/bin/python'),'-u',str(Path(__file__).with_name('run.py'))],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'))
        log.close();info=identity(worker.pid);assert info
        write(QUEUE/'state.json',dict(phase='running',pid=os.getpid(),worker_pid=worker.pid,worker_identity=info,only_GPU=1))
        while worker.poll() is None:time.sleep(30)
        code=worker.wait()
        if code==75 and not (REPORT/'STARTED.json').exists():continue
        assert code==0, 'Worker failed; preserve failure and never automatically retry'
        started=time.monotonic();proof=verify();proof['CPU_verification_seconds']=time.monotonic()-started
        write(REPORT/'VERIFIED.json',proof)
        write(QUEUE/'state.json',dict(phase='complete_verified',pid=os.getpid(),worker_exit=code,
            verified_sha256=sha(REPORT/'VERIFIED.json'),only_GPU=1))
        return

def activate():
    check_ready();QUEUE.mkdir(parents=True,exist_ok=True)
    assert not (QUEUE/'ACTIVATION.json').exists()
    assert not any((REPORT/n).exists() for n in ['STARTED.json','FAILURE.json','COMPLETE.json'])
    # Protect every existing GPU owner plus actual repository baseline/coordinator/publisher processes.
    cards=inventory();pids={o['pid'] for c in cards.values() for o in c['owners']}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            i=identity(int(proc.name))
            if i and i['uid']==os.getuid() and str(ROOT) in i['cmdline'] and ENTRY not in i['cmdline']:pids.add(int(proc.name))
        except (PermissionError,ProcessLookupError):pass
    protected={str(pid):identity(pid) for pid in pids if identity(pid)}
    write(QUEUE/'BEFORE.json',dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        protected=protected,inventory=cards))
    unit=Path.home()/'.config/systemd/user'/UNIT
    assert not unit.exists()
    unit.write_text('[Unit]\nDescription=One fixed LoRA all full-data research confirmation\n[Service]\nType=simple\nWorkingDirectory='+str(ROOT)+
        '\nExecStart=/usr/bin/nohup '+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/ENTRY)+' --watch\nRestart=no\nKillMode=process\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True))
    assert identity(pid) and ENTRY in identity(pid)['cmdline']
    after={p:identity(int(p)) for p in protected}
    assert after==protected, 'Protected process identities changed; inspect, do not restart GPUs'
    write(QUEUE/'ACTIVATION.json',dict(pid=pid,identity=identity(pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        protected_identities_preserved=protected,ready_sha256=sha(REPORT/'READY.json'),only_GPU=1,
        old_screen_service_not_modified=True,new_unit=UNIT,not_enabled_on_boot=True))
    print(json.dumps(dict(activated=True,pid=pid,no_GPU_interruption=True)))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--activate',action='store_true');a=p.parse_args()
    try:
        if a.watch:watch()
        elif a.activate:activate()
        else:raise ValueError('Explicit --watch or --activate required')
    except Exception:
        write(QUEUE/'needs_attention.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
