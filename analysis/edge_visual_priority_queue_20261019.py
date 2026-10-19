"""One GPU1 owner: edge feasibility first, then all existing research work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback
import onevision_edge_gpu1_queue_20261019 as previous

ROOT=previous.ROOT
STUDY=ROOT/'summary/edge_deployment_20261019/gpu_visual_probe/v1'
REPORT=STUDY/'queue'
UNIT=previous.UNIT
ENTRY='analysis/edge_visual_priority_queue_20261019.py'
WORKER='analysis/edge_visual_gpu_probe_20261019.py'
read,write,sha=previous.read,previous.write,previous.sha


def complete():
    p=STUDY/'COMPLETE.json'
    if not p.exists():return False
    v=read(p)
    assert v['complete'] and v['TEST_QA']==0 and v['no_model_updates']
    assert v['views']==read(STUDY/'READY.json')['views'] and len(v['requests'])==80
    assert v['source_sha256']==sha(ROOT/WORKER) and v['readiness_sha256']==sha(STUDY/'READY.json')
    assert len(v['original_dynamic'])==5 and v['physical_GPU']==1
    assert all(len([r for r in v['requests'] if r['view']==view])==20 for view in v['views'])
    return True


def choose(action,inherited,done,gpu):
    if inherited or done:return action
    if gpu is None:return 'wait',None
    assert gpu==1
    return 'edge_visual_probe',dict(gpu=1)


def extend(original,spec):
    previous.extend(original,spec)
    warm=original.prior.cold.previous
    old_next,old_command,old_verify,old_write=warm.next_action,warm.command_for,warm.previous.verify_worker,warm.write
    def next_action(*args):
        action,state=old_next(*args)
        assert not (STUDY/'FAILURE.json').exists(),'Preserve failure; no blind retry'
        done=complete()
        state.update(edge_probe_complete=done,edge_target='personal_computer_CPU_GPU_or_phone',
            research_allowed_GPUs=[1],all_original_jobs_retained=True,edge_probe_before_old_work=True)
        return choose(action,state.get('waiting_for_natural_completion',False),done,
                      previous.select(previous.device.inventory(),os.getuid())),state
    def command(kind,target,specs):
        if kind!='edge_visual_probe':return old_command(kind,target,specs)
        return [WORKER,'--gpu','1'],STUDY/'nohup.log'
    def verify(kind,target,*args):
        if kind!='edge_visual_probe':return old_verify(kind,target,*args)
        assert complete() and target['gpu']==1
        started=read(STUDY/'STARTED.json');assert not started['inventory']['1']['owners']
        return True
    def status_write(path,value):
        if Path(path).name=='queue_state.json':
            value=dict(value,research_allowed_GPUs=[1],max_our_GPU_cards=3)
        return old_write(path,value)
    warm.next_action,warm.command_for,warm.previous.verify_worker,warm.write=next_action,command,verify,status_write


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True)
    previous.prepare()
    # Ancestor gates bind their original operational document by exact SHA.
    (REPORT/'SCHEDULE_AMENDMENT.md').write_bytes((previous.REPORT/'SCHEDULE_AMENDMENT.md').read_bytes())
    assert read(STUDY/'CPU_CHECK.json')['checks']==3
    auth=dict(read(previous.REPORT/'AUTHORIZATION.json'),
        edge_visual_sources={p:sha(ROOT/p) for p in [ENTRY,WORKER]},
        edge_visual_plan_sha256=sha(STUDY/'READY.json'),
        edge_visual_protocol_sha256=sha(STUDY/'PROTOCOL.md'),
        edge_previous_auth_sha256=sha(previous.REPORT/'AUTHORIZATION.json'),
        all_previous_jobs_retained=True,only_GPU1=True)
    if (REPORT/'READY.json').exists():assert read(REPORT/'AUTHORIZATION.json')==auth;return
    write(REPORT/'AUTHORIZATION.json',auth)
    # Real ancestor checks and new routing controls run with CUDA disabled.
    previous.parent.REPORT,previous.parent.extend=REPORT,extend
    previous.parent.main(True)
    a=('TRAIN_update_pilot',dict(gpu=1))
    assert choose(a,True,False,1)==a
    assert choose(a,False,True,1)==a
    assert choose(a,False,False,None)==('wait',None)
    assert choose(a,False,False,1)==('edge_visual_probe',dict(gpu=1))
    write(REPORT/'CPU_CHECK.json',dict(new_policy_checks=4,all_ancestor_checks_completed=True,
        worker_CPU_checks=3,all_old_work_retained=True,no_GPU_started=True))
    write(REPORT/'READY.json',dict(authorization_sha256=sha(REPORT/'AUTHORIZATION.json'),sources_FROZEN=True))


def watch():
    auth=read(REPORT/'AUTHORIZATION.json')
    for p,h in auth['edge_visual_sources'].items():assert sha(ROOT/p)==h
    assert sha(STUDY/'READY.json')==auth['edge_visual_plan_sha256']
    assert sha(STUDY/'PROTOCOL.md')==auth['edge_visual_protocol_sha256']
    assert sha(previous.REPORT/'AUTHORIZATION.json')==auth['edge_previous_auth_sha256']
    assert sha(REPORT/'AUTHORIZATION.json')==read(REPORT/'READY.json')['authorization_sha256']
    previous.device.select=previous.select
    previous.parent.REPORT,previous.parent.extend=REPORT,extend
    previous.parent.main(False)


def activate():
    assert (REPORT/'READY.json').exists() and not (REPORT/'ACTIVATION.json').exists()
    import activate_onevision_lora_rite_queue_20261019 as activation
    base=activation.base
    assert base.cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    old=int(base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert base.identity(old) and b'onevision_edge_gpu1_queue_20261019.py' in (Path('/proc')/str(old)/'cmdline').read_bytes()
    gpu_pids={int(s) for s in base.cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines()}
    protected={str(p):base.identity(p) for p in gpu_pids}
    for pid in [51266,482168]:
        if base.identity(pid):protected[str(pid)]=base.identity(pid)
    before=read(previous.REPORT/'queue_state.json')
    write(REPORT/'BEFORE.json',dict(retired_CPU_only=old,old_identity=base.identity(old),old_state=before,protected=protected))
    subprocess.run(['systemctl','--user','stop',UNIT],check=True)
    assert base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value')=='0'
    state=read(previous.REPORT/'queue_state.json')
    write(REPORT/'OLD_STATE_AFTER_CPU_STOP.json',state)
    if state.get('phase')=='running' and base.identity(state['worker_pid']):
        item=base.identity(state['worker_pid'])
        write(REPORT/'PILOT_INHERITED_WORKER.json',dict(worker_pid=state['worker_pid'],worker_startticks=item['startticks'],kind=state['kind'],target=state['target']))
    assert all(base.identity(int(p))==v for p,v in protected.items())
    drop=Path.home()/'.config/systemd/user'/f'{UNIT}.d'/('99-'+'z'*45+'-edge-visual.conf')
    assert not drop.exists()
    drop.write_text('[Service]\nExecStart=\nExecStart=/usr/bin/nohup '+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/ENTRY)+' --watch\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    new=int(base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert base.identity(new) and all(base.identity(int(p))==v for p,v in protected.items())
    write(REPORT/'ACTIVATION.json',dict(new_CPU_pid=new,new_identity=base.identity(new),retired_CPU_only=old,
        protected_processes_preserved=protected,no_GPU_interruption=True,allowed_research_GPUs=[1],all_old_work_retained=True))
    write(previous.REPORT/'queue_state.json',dict(phase='redirect',authoritative=str((REPORT/'queue_state.json').relative_to(ROOT)),new_CPU_pid=new))
    print(json.dumps(dict(activated=True,new_CPU_pid=new,no_GPU_interruption=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--activate',action='store_true');a=p.parse_args()
    try:
        if a.watch:watch()
        elif a.activate:activate()
        else:prepare()
    except Exception:
        write(REPORT/'needs_attention.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
