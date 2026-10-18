"""Run the small, fixed source-initialization controls before old diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
import onevision_simple_research_focus_queue as previous
import onevision_research_device_entry as device
import onevision_initialization_source_report as reporter

ROOT=previous.ROOT
STUDY=ROOT/'summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1'
REPORT=STUDY/'queue'
ENTRY='analysis/onevision_initialization_source_entry.py'
OLD_EXTEND=previous.previous.extend


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def complete(phase,index,spec):
    name='sources' if phase=='sources' else spec['cases'][index]
    p=STUDY/(name+'_complete.json')
    if not p.exists():return False
    obj=read(p);assert obj['complete'] and obj['spec']==spec
    run=ROOT/obj['run'];assert run.resolve().is_relative_to((ROOT/'exp/onevision_initialization_source_20261018/v1').resolve())
    for rel,s in obj['artifact_sha256'].items():
        p=run/rel;assert p.resolve().is_relative_to(run.resolve()) and sha(p)==s
    if phase=='case':
        assert obj['target_full_TEST_QA']==507 and read(run/'initial_function_gate.json')['passed']
        reporter.main()
    return True


def pending(source_done,case_done):
    if not source_done:return dict(phase='sources',index=None)
    for i,done in enumerate(case_done):
        if not done:return dict(phase='case',index=i)
    return None


def extend(original,native_spec):
    OLD_EXTEND(original,native_spec)
    spec=read(STUDY/'CPU_PLAN_CHECK.json')['spec'];warm=original.prior.cold.previous
    old_next,old_command,old_verify=warm.next_action,warm.command_for,warm.previous.verify_worker
    def next_action(*args):
        action,state=old_next(*args)
        if state.get('waiting_for_natural_completion') or action[0]=='acquired_anchor':return action,state
        assert not list(STUDY.glob('*needs_attention.json')), 'Preserve control failure, no blind retry'
        built=complete('sources',None,spec)
        done=[complete('case',i,spec) for i in range(5)] if built else [False]*5
        target=pending(built,done)
        state.update(initialization_sources_complete=built,initialization_cases_complete=done,
                     formal_trials_deferred_by_latest_user=True,no_new_routing_module=True)
        if target is None:return action,state
        gpu=device.select(device.inventory(),os.getuid())
        if gpu is None:return ('wait',None),state
        return ('initialization_source',dict(target,gpu=gpu)),state
    def command(kind,target,specs):
        if kind!='initialization_source':return old_command(kind,target,specs)
        cmd=[ENTRY,'--gpu',str(target['gpu']),'--phase',target['phase']]
        if target['phase']=='case':cmd+=['--case-index',str(target['index'])]
        label='sources' if target['phase']=='sources' else spec['cases'][target['index']]
        return cmd,STUDY/(label+'_nohup.log')
    def verify(kind,target,*args):
        if kind!='initialization_source':return old_verify(kind,target,*args)
        assert complete(target['phase'],target['index'],spec)
        name='sources' if target['phase']=='sources' else spec['cases'][target['index']]
        proof=read(STUDY/(name+'_complete.json'));receipt=read(ROOT/proof['run']/'physical_device_execution.json')
        assert receipt['physical_GPU']==target['gpu'] and not receipt['admission_inventory'][str(target['gpu'])]['owners']
        return True
    warm.next_action,warm.command_for,warm.previous.verify_worker=next_action,command,verify
    assert command('initialization_source',dict(gpu=0,phase='case',index=4),None)[0]==[ENTRY,'--gpu','0','--phase','case','--case-index','4']
    assert command('data_factorial',2,None)==old_command('data_factorial',2,None)


def main(check=False):
    REPORT.mkdir(parents=True,exist_ok=True);auth=read(REPORT/'AUTHORIZATION.json')
    for n,s in auth['initialization_control_sources'].items():assert sha(ROOT/n)==s
    assert sha(STUDY/'PROTOCOL.md')==auth['initialization_protocol_sha256']
    spec=read(STUDY/'CPU_PLAN_CHECK.json')['spec']
    for n,s in spec['source_sha256'].items():assert sha(ROOT/n)==s
    previous.REPORT=REPORT;previous.previous.extend=extend
    previous.main(check)
    if check:
        assert pending(False,[False]*5)==dict(phase='sources',index=None)
        assert pending(True,[True,False,False,False,False])==dict(phase='case',index=1)
        assert pending(True,[True]*5) is None
        (REPORT/'SOURCE_QUEUE_CPU_CHECK.json').write_text(json.dumps(dict(CPU_only=True,checks=3,new_old_command_checks=2,
            all5_conditions_required=True,all_original_jobs_retained=True,GPU_started=False))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    try:main(a.check)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
