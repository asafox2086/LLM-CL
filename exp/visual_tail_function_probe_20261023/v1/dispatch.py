"""One frozen probe on physical GPU1, no restart after an actual attempt."""
import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from core import ROOT,REPORT,WORK,sha,write
QUEUE=REPORT/'queue'
UNIT='llmcl-onevision-visual-tail-function-probe-20261023.service'
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
    ours={g for g,c in cards.items() if any(p['uid']==os.getuid() for p in c['owners'])};c=cards[1]
    return not ours&{0,1} and len(ours|{1})<=3 and not c['owners'] and c['memory_MiB']<500 and c['utilization']<10

def check_ready():
    r=json.loads((REPORT/'READY.json').read_text())
    for key in ('sources','dependencies','evidence'):
        for n,h in r[key].items():assert sha(ROOT/n)==h,n
    a=json.loads((REPORT/'ADMISSION_REVIEW.json').read_text());assert a['decision']=='accept'
    assert a['ready_sha256']==sha(REPORT/'READY.json')
    assert not (REPORT/'STARTED.json').exists() and not WORK.exists()
    return r

def watch():
    QUEUE.mkdir(parents=True,exist_ok=True)
    lock=(QUEUE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_ready()
    while True:
        if not available(inventory()):
            write(QUEUE/'state.json',dict(phase='wait',pid=os.getpid(),only_GPU=1,no_model_attempt=True));time.sleep(60);continue
        with (QUEUE/'worker_nohup.log').open('ab') as log:
            w=subprocess.Popen([str(ROOT/'.vision-env/bin/python'),'-u',str(Path(__file__).with_name('run.py'))],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'),start_new_session=True)
        i=identity(w.pid);assert i
        write(QUEUE/'state.json',dict(phase='running',pid=os.getpid(),worker_pid=w.pid,worker_startticks=i['startticks'],only_GPU=1))
        while w.poll() is None:
            marker=REPORT/'STARTED.json'
            remaining=None
            if marker.exists():
                started=json.loads(marker.read_text())
                assert started['pid']==w.pid and started['startticks']==i['startticks']
                remaining=started['monotonic_start']+1800-time.monotonic()
                if remaining<=0:
                    def signal_own_group(sig):
                        if w.poll() is not None:return
                        current=identity(w.pid)
                        assert current and current['uid']==os.getuid() and current['startticks']==i['startticks']
                        assert str(Path(__file__).with_name('run.py')) in current['cmdline'] and os.getpgid(w.pid)==w.pid
                        try:os.killpg(w.pid,sig)
                        except ProcessLookupError:pass
                    signal_own_group(signal.SIGTERM)
                    try:w.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        signal_own_group(signal.SIGKILL);w.wait(timeout=10)
                    timeout=dict(pid=w.pid,startticks=i['startticks'],external_deadline_monotonic=started['monotonic_start']+1800,
                        termination_observed_monotonic=time.monotonic(),worker_exit=w.returncode,no_retry=True,
                        cleanup_latency_is_not_extra_model_budget=True)
                    write(REPORT/'WATCHDOG_TIMEOUT.json',timeout)
                    if not (REPORT/'FAILURE.json').exists():write(REPORT/'FAILURE.json',dict(external_budget_timeout=timeout))
                    write(QUEUE/'state.json',dict(phase='closed_budget_timeout',pid=os.getpid(),no_retry=True))
                    raise RuntimeError('External STARTED deadline reached; own worker stopped, no retry')
            try:w.wait(timeout=1 if remaining is None else min(1,max(.001,remaining)))
            except subprocess.TimeoutExpired:pass
        code=w.wait()
        if code==75 and not (REPORT/'STARTED.json').exists():continue
        if code:
            write(QUEUE/'state.json',dict(phase='closed_worker_failure',pid=os.getpid(),worker_exit=code,no_retry=True))
            raise RuntimeError('Probe stopped, do not restart or extend')
        break
    with (QUEUE/'verification.log').open('ab') as log:
        subprocess.run([str(ROOT/'.assets-env/bin/python'),str(Path(__file__).with_name('verify.py'))],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,timeout=180)
    write(QUEUE/'state.json',dict(phase='complete_verified',pid=os.getpid(),verified_sha256=sha(REPORT/'VERIFIED.json'),
        automatic_continual_training=False))

def activate():
    check_ready();QUEUE.mkdir(parents=True,exist_ok=True)
    assert not (QUEUE/'ACTIVATION.json').exists()
    protected={}
    for c in inventory().values():
        for p in c['owners']:
            i=identity(p['pid'])
            if i:protected[str(p['pid'])]=i
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:i=identity(int(p.name))
        except PermissionError:continue
        if i and i['uid']==os.getuid() and str(ROOT) in i['cmdline'] and ENTRY not in i['cmdline']:
            protected[p.name]=i
    private=Path(__file__).parent/'.activation_private';private.mkdir(exist_ok=False)
    write(private/'BEFORE.json',protected)
    unit=Path.home()/'.config/systemd/user'/UNIT;assert not unit.exists()
    unit.write_text('[Unit]\nDescription=One S1 visual function substitution diagnostic\n[Service]\nType=simple\nWorkingDirectory='+str(ROOT)+
        '\nExecStart=/usr/bin/nohup '+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/ENTRY)+' --watch\nRestart=no\nKillMode=process\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True));i=identity(pid)
    assert i and ENTRY in i['cmdline']
    assert {p:identity(int(p)) for p in protected}==protected,'Protected process identity changed; inspect, never signal GPUs'
    write(QUEUE/'ACTIVATION.json',dict(pid=pid,startticks=i['startticks'],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        ready_sha256=sha(REPORT/'READY.json'),protected={p:{k:v[k] for k in ('uid','startticks')} for p,v in protected.items()},
        raw_commands_private=True,only_GPU=1,not_enabled_on_boot=True))
    print(json.dumps(dict(activated=True,pid=pid,startticks=i['startticks'])))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--activate',action='store_true');a=p.parse_args()
    try:
        if a.watch:watch()
        elif a.activate:activate()
        else:raise ValueError('Choose explicit --activate or --watch')
    except Exception:
        write(QUEUE/'needs_attention.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
