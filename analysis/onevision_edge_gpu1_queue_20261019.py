"""Operational GPU1-only restriction; all frozen experiments and order unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import onevision_lora_rite_queue_20261019 as parent
import onevision_research_device_entry as device

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/edge_deployment_20261019/queue'
OLD = parent.REPORT
UNIT = 'llmcl-onevision-causal-priority.service'
ORIGINAL_SELECT = device.select
PARENT_EXTEND = parent.extend


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def select(cards,uid):
    chosen=ORIGINAL_SELECT(cards,uid)
    return 1 if chosen==1 else None


def checks():
    def cards():return {g:dict(owners=[],memory_MiB=0,utilization=0) for g in range(4)}
    uid=1004;other=1024;c=cards();assert select(c,uid)==1
    c[1]['owners']=[dict(pid=12,uid=other)];assert select(c,uid) is None
    c=cards();c[1]['memory_MiB']=600;assert select(c,uid) is None
    c=cards();c[1]['utilization']=20;assert select(c,uid) is None
    c=cards();c[0]['owners']=[dict(pid=13,uid=uid)];assert select(c,uid) is None
    c=cards();c[2]['owners']=[dict(pid=14,uid=uid)];c[3]['owners']=[dict(pid=15,uid=uid)];assert select(c,uid)==1
    write(REPORT/'CPU_CHECK.json',dict(checks=6,allowed_research_GPUs=[1],max_our_cards=3,
        no_GPU_started=True,all_original_trials_retained=True,scientific_sources_unchanged=True))


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True)
    checks()
    prior=read(OLD/'AUTHORIZATION.json')
    auth=dict(prior,edge_queue_source_sha256=sha(__file__),edge_parent_auth_sha256=sha(OLD/'AUTHORIZATION.json'),
              edge_research_GPUs=[1],edge_target='phone or ordinary CPU',
              edge_scientific_schedule_unchanged=True,edge_only_narrows_device_policy=True)
    if (REPORT/'READY.json').exists():
        assert read(REPORT/'AUTHORIZATION.json')==auth
        assert read(REPORT/'READY.json')['authorization_sha256']==sha(REPORT/'AUTHORIZATION.json')
        return
    for p,h in prior['lora_rite_sources'].items():assert sha(ROOT/p)==h
    (REPORT/'SCHEDULE_AMENDMENT.md').write_bytes((OLD/'SCHEDULE_AMENDMENT.md').read_bytes())
    write(REPORT/'AUTHORIZATION.json',auth)
    write(REPORT/'READY.json',dict(CPU_only=True,authorization_sha256=sha(REPORT/'AUTHORIZATION.json'),
                                  source_FROZEN=True,science_unchanged=True))


def extend(original,spec):
    PARENT_EXTEND(original,spec)
    warm=original.prior.cold.previous
    old_next=warm.next_action
    def next_action(*args):
        action,state=old_next(*args)
        state.update(research_allowed_GPUs=[1],edge_target='phone_or_ordinary_CPU',
                     max_our_GPU_cards=3,all_scientific_work_retained=True)
        if not state.get('waiting_for_natural_completion') and isinstance(action[1],dict) and 'gpu' in action[1]:
            assert action[1]['gpu']==1
        return action,state
    warm.next_action=next_action


def watch():
    auth=read(REPORT/'AUTHORIZATION.json')
    assert sha(__file__)==auth['edge_queue_source_sha256']
    assert auth['edge_research_GPUs']==[1]
    assert sha(OLD/'AUTHORIZATION.json')==auth['edge_parent_auth_sha256']
    assert sha(REPORT/'AUTHORIZATION.json')==read(REPORT/'READY.json')['authorization_sha256']
    device.select=select
    parent.REPORT,parent.extend=REPORT,extend
    parent.main(False)


def activate():
    prepare()
    assert not (REPORT/'ACTIVATION.json').exists()
    import activate_onevision_lora_rite_queue_20261019 as activation
    base=activation.base
    assert base.cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    old=int(base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    ident=base.identity(old)
    assert ident and b'onevision_lora_rite_queue_20261019.py' in (Path('/proc')/str(old)/'cmdline').read_bytes()
    gpu_pids={int(s) for s in base.cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines()}
    protected={str(p):base.identity(p) for p in gpu_pids}
    for p in [51266,482168]:
        if base.identity(p):protected[str(p)]=base.identity(p)
    write(REPORT/'BEFORE.json',dict(retiring_CPU_only=old,identity=ident,protected_processes=protected,
                                  old_state=read(OLD/'queue_state.json')))
    subprocess.run(['systemctl','--user','stop',UNIT],check=True)
    assert base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value')=='0'
    state=read(OLD/'queue_state.json')
    write(REPORT/'OLD_STATE_AFTER_CPU_STOP.json',state)
    if state.get('phase')=='running' and base.identity(state['worker_pid']):
        worker=base.identity(state['worker_pid'])
        write(REPORT/'PILOT_INHERITED_WORKER.json',dict(worker_pid=state['worker_pid'],
            worker_startticks=worker['startticks'],kind=state['kind'],target=state['target']))
    assert all(base.identity(int(p))==v for p,v in protected.items())
    drop=Path.home()/'.config/systemd/user'/f'{UNIT}.d'/('99-'+'z'*40+'-edge-gpu1.conf')
    assert not drop.exists()
    drop.write_text('[Service]\nExecStart=\nExecStart=/usr/bin/nohup '+str(ROOT/'.assets-env/bin/python')+
                    ' -u '+str(Path(__file__).resolve())+' --watch\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    new=int(base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert base.identity(new) and all(base.identity(int(p))==v for p,v in protected.items())
    write(REPORT/'ACTIVATION.json',dict(new_CPU_pid=new,new_CPU_identity=base.identity(new),
        retired_CPU_only=old,protected_processes_preserved=protected,no_GPU_interruption=True,
        allowed_research_GPUs=[1],scientific_order_unchanged=True))
    write(OLD/'queue_state.json',dict(phase='redirect',authoritative=str((REPORT/'queue_state.json').relative_to(ROOT)),
        new_CPU_pid=new,old_CPU_intentionally_retired=True,no_GPU_interruption=True))
    print(json.dumps(dict(activated=True,new_CPU_pid=new,protected_GPU_pids=sorted(gpu_pids))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--activate',action='store_true')
    a=parser.parse_args()
    if a.watch:
        try:watch()
        except Exception:
            import traceback
            write(REPORT/'needs_attention.json',dict(traceback=traceback.format_exc(),no_blind_retry=True,no_GPU_killed=True))
            raise
    elif a.activate:activate()
    else:prepare();print('{"prepared": true, "GPU_started": false}')
