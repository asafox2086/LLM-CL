"""Single GPU1 owner: authored capacity controls, then all previous work.

No active baseline source is changed. This only prioritizes a predeclared,
question-driven ablation; original controls and negative confirmations remain.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback
import edge_visual_priority_queue_20261019 as previous

ROOT=previous.ROOT
STUDY=ROOT/'summary/edge_deployment_20261019/architecture_capacity/v1'
REPORT=STUDY/'queue'
UNIT=previous.UNIT
ENTRY='analysis/architecture_capacity_priority_20261019.py'
WORKER='exp/architecture_capacity_20261019/v1/run.py'
read,write,sha=previous.read,previous.write,previous.sha
CASES=['CL_original','CL_shared_equal_rank','CL_shared_small_rank',
       'SMo_original','SMo_two_equal_rank','SMo_two_small_rank']
COMPLETE_CACHE={}


def complete(case,phase):
    p=STUDY/(case+'_'+phase+'_COMPLETE.json')
    if not p.exists():return False
    v=read(p)
    assert v['complete'] and v['case']==case and v['source_sha256']==sha(ROOT/WORKER)
    assert v['ready_sha256']==sha(STUDY/'READY.json')
    if phase=='train':
        assert v['TRAIN_QA']==1313 and v['stages']==19 and v['updates']==93
        cp=v['checkpoint'];path=ROOT/cp['path'];s=path.stat()
        identity=(cp['sha256'],s.st_ino,s.st_size,s.st_mtime_ns)
        if COMPLETE_CACHE.get(str(path))!=identity:
            assert sha(path)==cp['sha256'];COMPLETE_CACHE[str(path)]=identity
    else:
        assert v['TEST_QA']==21454 and sum(x['QA'] for x in v['domains'].values())==21454
    return True


def choose(action,inherited,trained,tested,gpu):
    if inherited:return action
    pending=next((('capacity_train',dict(case=c,gpu=1)) for c in CASES if not trained[c]),None)
    if pending is None:pending=next((('capacity_full',dict(case=c,gpu=1)) for c in CASES if not tested[c]),None)
    if pending is None:return action
    return ('wait',None) if gpu is None else pending


def extend(original,spec):
    previous.extend(original,spec)
    warm=original.prior.cold.previous
    old_next,old_command,old_verify,old_write=warm.next_action,warm.command_for,warm.previous.verify_worker,warm.write
    def next_action(*args):
        action,state=old_next(*args)
        failures=list(STUDY.glob('*_FAILURE.json'))
        assert not failures, 'Preserve failure; no blind retry: '+str(failures)
        trained={c:complete(c,'train') for c in CASES}
        tested={c:complete(c,'full') for c in CASES}
        state.update(capacity_train_complete=trained,capacity_full_complete=tested,
            capacity_ablation_before_auxiliary_controls=True,all_original_jobs_retained=True,
            significance='Separate expert specialization from total adapter rank in two published mechanisms',
            research_allowed_GPUs=[1],not_new_method_or_quality_result=True)
        gpu=previous.previous.select(previous.previous.device.inventory(),os.getuid())
        return choose(action,state.get('waiting_for_natural_completion',False),trained,tested,gpu),state
    def command(kind,target,specs):
        if kind not in ['capacity_train','capacity_full']:return old_command(kind,target,specs)
        phase='train' if kind=='capacity_train' else 'full'
        return [WORKER,'--case',target['case'],'--phase',phase],STUDY/(target['case']+'_'+phase+'_nohup.log')
    def verify(kind,target,*args):
        if kind not in ['capacity_train','capacity_full']:return old_verify(kind,target,*args)
        return complete(target['case'],'train' if kind=='capacity_train' else 'full')
    def status_write(path,value):
        if Path(path).name=='queue_state.json':value=dict(value,research_allowed_GPUs=[1],max_our_GPU_cards=3)
        return old_write(path,value)
    warm.next_action,warm.command_for,warm.previous.verify_worker,warm.write=next_action,command,verify,status_write


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True)
    assert read(STUDY/'READY.json')['cases'].keys()==dict.fromkeys(CASES).keys()
    train0={c:False for c in CASES};test0=dict(train0);all_done={c:True for c in CASES}
    old=('edge_visual_probe',dict(gpu=1))
    assert choose(old,True,train0,test0,1)==old
    assert choose(old,False,train0,test0,None)==('wait',None)
    assert choose(old,False,train0,test0,1)==('capacity_train',dict(case=CASES[0],gpu=1))
    assert choose(old,False,all_done,test0,1)==('capacity_full',dict(case=CASES[0],gpu=1))
    assert choose(old,False,all_done,all_done,1)==old
    # The inherited owner verifies all original frozen authorizations itself.
    prior=read(previous.REPORT/'AUTHORIZATION.json')
    auth=dict(prior,capacity_sources={p:sha(ROOT/p) for p in [ENTRY,WORKER,'exp/architecture_capacity_20261019/v1/capacity_core.py']},
        capacity_ready_sha256=sha(STUDY/'READY.json'),capacity_protocol_sha256=sha(STUDY/'PROTOCOL.md'),
        prior_edge_authorization_sha256=sha(previous.REPORT/'AUTHORIZATION.json'),
        scientific_question_before_execution=True,all_previous_work_retained=True,only_GPU1=True)
    if (REPORT/'READY.json').exists():assert read(REPORT/'AUTHORIZATION.json')==auth;return
    write(REPORT/'AUTHORIZATION.json',auth)
    # An ancestor's byte-bound schedule file remains intact; new priority is explicit.
    (REPORT/'SCHEDULE_AMENDMENT.md').write_bytes((previous.REPORT/'SCHEDULE_AMENDMENT.md').read_bytes())
    write(REPORT/'CPU_POLICY.json',dict(new_checks=5,no_GPU_started=True,
        inherited_worker_preserved=True,all_six_train_before_all_six_full=True,old_work_retained=True))
    write(REPORT/'READY.json',dict(authorization_sha256=sha(REPORT/'AUTHORIZATION.json'),sources_FROZEN=True))


def watch():
    auth=read(REPORT/'AUTHORIZATION.json')
    assert all(sha(ROOT/p)==h for p,h in auth['capacity_sources'].items())
    assert sha(STUDY/'READY.json')==auth['capacity_ready_sha256']
    assert sha(STUDY/'PROTOCOL.md')==auth['capacity_protocol_sha256']
    assert sha(previous.REPORT/'AUTHORIZATION.json')==auth['prior_edge_authorization_sha256']
    assert sha(REPORT/'AUTHORIZATION.json')==read(REPORT/'READY.json')['authorization_sha256']
    previous.previous.device.select=previous.previous.select
    previous.previous.parent.REPORT,previous.previous.parent.extend=REPORT,extend
    previous.previous.parent.main(False)


def activate():
    assert (REPORT/'READY.json').exists() and not (REPORT/'ACTIVATION.json').exists()
    import activate_onevision_lora_rite_queue_20261019 as activation
    base=activation.base
    assert base.cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    old=int(base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert base.identity(old) and b'edge_visual_priority_queue_20261019.py' in (Path('/proc')/str(old)/'cmdline').read_bytes()
    gpu_pids={int(s.strip()) for s in base.cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines() if s.strip()}
    protected={str(p):base.identity(p) for p in gpu_pids}
    for p in [51266,482168,1408452,1409431]:
        if base.identity(p):protected[str(p)]=base.identity(p)
    before=read(previous.REPORT/'queue_state.json')
    write(REPORT/'BEFORE.json',dict(retired_CPU_only=old,old_identity=base.identity(old),old_state=before,protected=protected))
    subprocess.run(['systemctl','--user','stop',UNIT],check=True)
    assert base.cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value')=='0'
    state=read(previous.REPORT/'queue_state.json');write(REPORT/'OLD_STATE_AFTER_CPU_STOP.json',state)
    if state.get('phase')=='running' and base.identity(state['worker_pid']):
        item=base.identity(state['worker_pid'])
        write(REPORT/'PILOT_INHERITED_WORKER.json',dict(worker_pid=state['worker_pid'],worker_startticks=item['startticks'],kind=state['kind'],target=state['target']))
    assert all(base.identity(int(p))==v for p,v in protected.items())
    drop=Path.home()/'.config/systemd/user'/f'{UNIT}.d'/('99-'+'z'*50+'-architecture-capacity.conf')
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
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');ap.add_argument('--activate',action='store_true');args=ap.parse_args()
    try:
        if args.watch:watch()
        elif args.activate:activate()
        else:prepare()
    except Exception:
        write(REPORT/'needs_attention.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
