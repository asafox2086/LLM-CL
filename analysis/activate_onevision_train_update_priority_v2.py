"""Replace only the CPU owner; preserve every live GPU/baseline identity."""
import argparse
import os
from pathlib import Path
import subprocess

import activate_onevision_train_update_pilot as base

ROOT=base.ROOT
SCIENCE=base.STUDY
REPORT=SCIENCE/'priority_v2'
OLD=base.REPORT
UNIT=base.UNIT
ENTRY='analysis/onevision_train_update_priority_v2.py'
SOURCES=[ENTRY,'analysis/activate_onevision_train_update_priority_v2.py']


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True)
    assert base.read(SCIENCE/'CPU_CHECK.json')['passed']
    assert base.sha(ROOT/'analysis/onevision_train_only_update_pilot.py')==base.read(SCIENCE/'PLAN.json')['worker_sha256']
    auth=base.read(OLD/'AUTHORIZATION.json')
    for rel,h in auth['TRAIN_update_sources'].items():assert base.sha(ROOT/rel)==h
    if (REPORT/'READY.json').exists():
        existing=base.read(REPORT/'AUTHORIZATION.json')
        assert existing['priority_v2_sources']=={n:base.sha(ROOT/n) for n in SOURCES}
        assert base.read(REPORT/'READY.json')['authorization_sha256']==base.sha(REPORT/'AUTHORIZATION.json');return
    auth.update(priority_v2_sources={n:base.sha(ROOT/n) for n in SOURCES},
        priority_v2_schedule_sha256=base.sha(REPORT/'SCHEDULE_AMENDMENT.md'),
        priority_v2_order=['current_GPU_natural_finish','existing_TRAIN_only_pilot12','all_previous_fulls_and_work'],
        authorization='User delegated GPU1 plan and asked direct efficient paper-inspired experiments. Operational ordering only; no score-dependent decisions; all fulls retained.')
    base.write(REPORT/'AUTHORIZATION.json',auth)
    with (REPORT/'CPU_CHECK_STDOUT.log').open('w') as f:
        subprocess.run([str(ROOT/'.assets-env/bin/python'),str(ROOT/ENTRY),'--check'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
    assert base.read(REPORT/'PRIORITY_CPU_CHECK.json')['checks']==6
    base.write(REPORT/'READY.json',dict(CPU_only=True,authorization_sha256=base.sha(REPORT/'AUTHORIZATION.json'),GPU_started=False))


def activate():
    prepare();assert not (REPORT/'ACTIVATION.json').exists()
    assert base.cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    old=int(base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert old and base.identity(old)
    assert b'onevision_train_update_pilot_queue.py' in (Path('/proc')/str(old)/'cmdline').read_bytes()
    gpu_pids={int(x) for x in base.cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines()}
    protected={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==old:continue
        try:
            if int(p.name) in gpu_pids or (p.stat().st_uid==os.getuid() and b'LLMCL' in (p/'cmdline').read_bytes()):
                identity=base.identity(int(p.name))
                if identity:protected[p.name]=identity
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    state_before=base.read(OLD/'queue_state.json')
    base.write(REPORT/'BEFORE.json',dict(old_CPU_pid=old,old_state=state_before,protected_processes=protected))
    subprocess.run(['systemctl','--user','stop',UNIT],check=True)
    assert base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value')=='0'
    state=base.read(OLD/'queue_state.json')
    base.write(REPORT/'OLD_STATE_AFTER_CPU_STOP.json',state)
    inherited=state.get('state',{}).get('inherited_worker')
    if state.get('phase')=='running':
        identity=base.identity(state['worker_pid'])
        if identity:inherited=dict(worker_pid=state['worker_pid'],worker_startticks=identity['startticks'],kind=state['kind'],target=state['target'])
    if inherited:
        identity=base.identity(inherited['worker_pid'])
        if identity:
            assert identity['startticks']==inherited['worker_startticks']
            base.write(REPORT/'PILOT_INHERITED_WORKER.json',inherited)
        else:
            import onevision_directional_coeff_report_v2 as reporter
            spec=base.read(reporter.REPORT/'CPU_PLAN.json')['spec']
            assert inherited['kind']=='directional_coeff' and reporter.complete(inherited['target']['phase'],inherited['target']['view'],spec)
    assert all(base.identity(int(p))==v for p,v in protected.items())
    drop=Path.home()/'.config/systemd/user'/f'{UNIT}.d'/('99-'+'z'*17+'-train-update-priority-v2.conf')
    assert not drop.exists()
    drop.write_text('[Service]\nExecStart=\nExecStart=/usr/bin/nohup '+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/ENTRY)+'\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True);subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert base.identity(pid) and all(base.identity(int(p))==v for p,v in protected.items())
    base.write(REPORT/'ACTIVATION.json',dict(new_CPU_pid=pid,new_CPU_identity=base.identity(pid),retired_CPU_only=old,
        protected_processes_preserved=protected,no_GPU_interruption=True,all_old_work_retained=True,scientific_worker_unchanged=True))
    base.write(OLD/'queue_state.json',dict(phase='redirect',authoritative=str((REPORT/'queue_state.json').relative_to(ROOT)),
        new_CPU_pid=pid,old_CPU_intentionally_retired=True,no_GPU_interruption=True))
    print(__import__('json').dumps(dict(activated=True,new_CPU_pid=pid,protected_GPU_pids=sorted(gpu_pids))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--activate',action='store_true')
    if parser.parse_args().activate:activate()
    else:prepare();print('{"prepared": true, "CPU_only": true}')
