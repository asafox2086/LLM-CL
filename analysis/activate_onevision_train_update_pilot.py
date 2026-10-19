"""Versioned CPU scheduling; never interrupt an active GPU or baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'summary/onevision_diagnostics_20261016/train_update_pilot_20261019/v1'
REPORT=STUDY/'queue'
OLD=ROOT/'summary/onevision_diagnostics_20261016/directional_coeff_20261018/v2/queue'
UNIT='llmcl-onevision-causal-priority.service'
NEW='analysis/onevision_train_update_pilot_queue.py'
SOURCES=['analysis/onevision_train_only_update_pilot.py',NEW,'analysis/activate_onevision_train_update_pilot.py','analysis/onevision_train_update_pilot_cpu.py']


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(*args):return subprocess.check_output(args,text=True).strip()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
        if s[0]=='Z':return None
        return dict(startticks=int(s[19]),UID=p.stat().st_uid,cmdline_sha256=sha(p/'cmdline'))
    except (FileNotFoundError,ProcessLookupError):return None


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True)
    check=read(STUDY/'CPU_CHECK.json');assert check['passed'] and check['no_CUDA_initialized']
    assert sha(ROOT/SOURCES[0])==read(STUDY/'PLAN.json')['worker_sha256']
    if (REPORT/'READY.json').exists():
        a=read(REPORT/'AUTHORIZATION.json');assert a['TRAIN_update_sources']=={n:sha(ROOT/n) for n in SOURCES}
        assert read(REPORT/'READY.json')['authorization_sha256']==sha(REPORT/'AUTHORIZATION.json');return
    auth=read(OLD/'AUTHORIZATION.json')
    auth.update(TRAIN_update_sources={n:sha(ROOT/n) for n in SOURCES},
        TRAIN_update_protocol_sha256=sha(STUDY/'PROTOCOL.md'),TRAIN_update_plan_sha256=sha(STUDY/'PLAN.json'),
        TRAIN_update_order=['current_GPU_natural_finish','both_coefficient_full21454','TRAIN_only12pilot','all_original_work'],
        no_TEST_score_selection=True,no_fixed_module_count=True)
    write(REPORT/'AUTHORIZATION.json',auth)
    with (REPORT/'CPU_CHECK_STDOUT.log').open('w') as f:
        subprocess.run([str(ROOT/'.assets-env/bin/python'),str(ROOT/NEW),'--check'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
    assert read(REPORT/'PILOT_QUEUE_CPU_CHECK.json')['checks']==6
    write(REPORT/'READY.json',dict(CPU_only=True,authorization_sha256=sha(REPORT/'AUTHORIZATION.json'),GPU_started=False))


def activate():
    prepare();assert not (REPORT/'ACTIVATION.json').exists()
    assert cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    old=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'));assert old and identity(old)
    assert b'onevision_directional_coeff_queue_v2.py' in (Path('/proc')/str(old)/'cmdline').read_bytes()
    gpu_pids={int(s) for s in cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines()}
    protected={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==old:continue
        try:
            if int(p.name) in gpu_pids or (p.stat().st_uid==os.getuid() and b'LLMCL' in (p/'cmdline').read_bytes()):
                value=identity(int(p.name))
                if value:protected[p.name]=value
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    state=read(OLD/'queue_state.json')
    write(REPORT/'BEFORE.json',dict(old_CPU_pid=old,old_state=state,protected_processes=protected))
    subprocess.run(['systemctl','--user','stop',UNIT],check=True)
    assert cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value')=='0'
    if state.get('phase')=='running':
        item=identity(state['worker_pid'])
        if item:
            protected[str(state['worker_pid'])]=item
            write(REPORT/'PILOT_INHERITED_WORKER.json',dict(worker_pid=state['worker_pid'],worker_startticks=item['startticks'],
                kind=state['kind'],target=state['target']))
        else:
            import onevision_directional_coeff_report_v2 as reporter
            assert state['kind']=='directional_coeff'
            assert reporter.complete(state['target']['phase'],state['target']['view'],read(OLD.parent/'CPU_PLAN.json')['spec'])
    assert all(identity(int(p))==v for p,v in protected.items())
    drop=Path.home()/'.config/systemd/user'/f'{UNIT}.d/99-zzzzzzzzzzzzzzzz-train-update-pilot.conf'
    assert not drop.exists();drop.write_text('[Service]\nExecStart=\nExecStart=/usr/bin/nohup '+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/NEW)+'\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True);subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert identity(pid) and all(identity(int(p))==v for p,v in protected.items())
    write(REPORT/'ACTIVATION.json',dict(new_CPU_pid=pid,new_CPU_identity=identity(pid),retired_CPU_only=old,
        protected_process_identities_preserved=protected,no_GPU_interruption=True,all_original_work_retained=True))
    write(OLD/'queue_state.json',dict(phase='redirect',authoritative=str((REPORT/'queue_state.json').relative_to(ROOT)),
        new_CPU_pid=pid,old_CPU_intentionally_retired=True,no_GPU_interruption=True))
    print(json.dumps(dict(activated=True,new_CPU_pid=pid,protected_GPU_pids=sorted(gpu_pids))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true')
    if p.parse_args().activate:activate()
    else:prepare();print(json.dumps(dict(prepared=True,CPU_only=True)))
