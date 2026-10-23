"""Two frozen flows on physical GPU1; never retries a started model."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from core import ROOT,REPORT,WORK,CASES,sha,write
from common import check_ready
QUEUE=REPORT/'queue'
UNIT='llmcl-onevision-answer-guided-rank-renewal-20261023.service'
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
    assert not WORK.exists() and not any((REPORT/'cases'/c/'STARTED.json').exists() for c in CASES)
    for case in CASES:
        while True:
            if not available(inventory()):
                write(QUEUE/'state.json',dict(phase='wait',case=case,pid=os.getpid(),only_GPU=1,TEST_QA=0))
                time.sleep(60);continue
            with (QUEUE/f'{case}_worker_nohup.log').open('ab') as log:
                worker=subprocess.Popen([str(ROOT/'.vision-env/bin/python'),'-u',str(Path(__file__).with_name('run.py')),'--case',case],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'))
            info=identity(worker.pid);assert info
            write(QUEUE/'state.json',dict(phase='running',case=case,pid=os.getpid(),worker_pid=worker.pid,
                worker_startticks=info['startticks'],only_GPU=1,TEST_QA=0))
            while worker.poll() is None:time.sleep(30)
            code=worker.wait()
            if code==75 and not (REPORT/'cases'/case/'STARTED.json').exists():continue
            if code!=0:
                write(QUEUE/'state.json',dict(phase='closed_worker_failure',case=case,pid=os.getpid(),worker_exit=code,no_retry=True))
                raise RuntimeError('Started worker failed: stop both-flow pilot, no retry/extension')
            assert json.loads((REPORT/'cases'/case/'COMPLETE.json').read_text())['complete']
            break
    with (QUEUE/'verification.log').open('ab') as log:
        subprocess.run([str(ROOT/'.assets-env/bin/python'),'-u',str(Path(__file__).with_name('common.py'))],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,timeout=300)
    proof=json.loads((REPORT/'VERIFIED.json').read_text());assert proof['complete']
    write(QUEUE/'state.json',dict(phase='complete_verified',pid=os.getpid(),only_GPU=1,TEST_QA=0,
        verified_sha256=sha(REPORT/'VERIFIED.json'),automatic_full_confirmation=False))

def activate():
    check_ready();QUEUE.mkdir(parents=True,exist_ok=True)
    assert not WORK.exists() and not (QUEUE/'ACTIVATION_PUBLIC.json').exists()
    assert not any((REPORT/'cases'/c/'STARTED.json').exists() for c in CASES)
    cards=inventory();pids={o['pid'] for c in cards.values() for o in c['owners']}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            i=identity(int(p.name))
            if i and i['uid']==os.getuid() and str(ROOT) in i['cmdline'] and ENTRY not in i['cmdline']:pids.add(int(p.name))
        except PermissionError:pass
    protected={str(p):identity(p) for p in pids if identity(p)}
    audit=Path(__file__).parent/'.activation_private';audit.mkdir(exist_ok=False)
    write(audit/'BEFORE.json',dict(protected=protected,inventory=cards))
    unit=Path.home()/'.config/systemd/user'/UNIT;assert not unit.exists()
    unit.write_text('[Unit]\nDescription=Two fixed compact rank-renewal FIT DEV flows\n[Service]\nType=simple\nWorkingDirectory='+str(ROOT)+
        '\nExecStart=/usr/bin/nohup '+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/ENTRY)+' --watch\nRestart=no\nKillMode=process\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True));i=identity(pid)
    assert i and ENTRY in i['cmdline']
    assert {p:identity(int(p)) for p in protected}==protected,'Protected identity changed: inspect, never restart GPUs'
    write(QUEUE/'ACTIVATION_PUBLIC.json',dict(pid=pid,startticks=i['startticks'],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),ready_sha256=sha(REPORT/'READY.json'),
        protected={p:{k:v[k] for k in ('startticks','uid')} for p,v in protected.items()},
        only_GPU=1,not_enabled_on_boot=True,raw_commands_private=True))
    print(json.dumps(dict(activated=True,pid=pid,startticks=i['startticks'],no_GPU_interruption=True)))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--activate',action='store_true');a=p.parse_args()
    try:
        if a.watch:watch()
        elif a.activate:activate()
        else:raise ValueError('Explicit --watch or --activate required')
    except Exception:write(QUEUE/'needs_attention.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
