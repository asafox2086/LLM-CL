"""After both coefficient full tests, run one TRAIN-only update pilot.

All earlier work is retained. No score-dependent scheduling or GPU competition.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import onevision_directional_coeff_queue_v2 as previous
import onevision_research_device_entry as device

ROOT=previous.ROOT
STUDY=ROOT/'summary/onevision_diagnostics_20261016/train_update_pilot_20261019/v1'
REPORT=STUDY/'queue'
OLD_EXTEND=previous.extend


def read(p):return json.loads(p.read_text())
def sha(p):return previous.sha(p)


def complete():
    pointer=STUDY/'complete.json'
    if not pointer.exists():return False
    p=read(pointer);assert p['complete'] and p['conditions']==12 and p['no_F1_claim']
    assert sha(STUDY/'PLAN.json')==p['plan_sha256']
    run=(ROOT/p['run']).resolve();assert run.is_relative_to((ROOT/'exp/onevision_train_update_pilot_20261019/v1').resolve())
    for rel,h in p['artifact_sha256'].items():
        f=(run/rel).resolve();assert f.is_relative_to(run) and sha(f)==h
    return True


def choose(action,inherited,both_previous_full,done,gpu):
    if inherited or action[0]=='acquired_anchor' or not both_previous_full or done:return action
    if gpu is None:return 'wait',None
    return 'TRAIN_update_pilot',dict(gpu=gpu)


def extend(original,native_spec):
    OLD_EXTEND(original,native_spec)
    warm=original.prior.cold.previous
    old_next,old_command,old_verify=warm.next_action,warm.command_for,warm.previous.verify_worker
    def next_action(*args):
        inherited=REPORT/'PILOT_INHERITED_WORKER.json'
        if inherited.exists() and not (REPORT/'PILOT_INHERITED_COMPLETE.json').exists():
            item=read(inherited)
            try:
                stat=(Path('/proc')/str(item['worker_pid'])/'stat').read_text().split(') ',1)[1].split()
                ticks=int(stat[19]) if stat[0]!='Z' else None
            except (FileNotFoundError,ProcessLookupError):ticks=None
            if ticks==item['worker_startticks']:
                return ('wait',None),dict(waiting_for_natural_completion=True,inherited_worker=item)
            assert old_verify(item['kind'],item['target'],args[0],args[3],args[4])
            (REPORT/'PILOT_INHERITED_COMPLETE.json').write_text(json.dumps(dict(verified=True,worker=item))+'\n')
        action,state=old_next(*args)
        if state.get('waiting_for_natural_completion'):return action,state
        assert not (STUDY/'needs_attention.json').exists()
        spec=read(previous.STUDY/'CPU_PLAN.json')['spec']
        fulls=[previous.reporter.complete('full',v,spec) for v in previous.reporter.VIEWS]
        done=complete();state.update(TRAIN_update_pilot_complete=done,no_TEST_driven_scheduling=True)
        return choose(action,False,all(fulls),done,device.select(device.inventory(),os.getuid())),state
    def command(kind,target,specs):
        if kind!='TRAIN_update_pilot':return old_command(kind,target,specs)
        return ['analysis/onevision_train_only_update_pilot.py','--gpu',str(target['gpu'])],STUDY/'pilot_nohup.log'
    def verify(kind,target,*args):
        if kind!='TRAIN_update_pilot':return old_verify(kind,target,*args)
        assert complete()
        run=ROOT/read(STUDY/'complete.json')['run'];receipt=read(run/'physical_device_execution.json')
        assert receipt['gpu']==target['gpu'] and not receipt['inventory'][str(target['gpu'])]['owners']
        return True
    warm.next_action,warm.command_for,warm.previous.verify_worker=next_action,command,verify
    assert command('TRAIN_update_pilot',dict(gpu=1),None)[0]==['analysis/onevision_train_only_update_pilot.py','--gpu','1']


def main(check=False):
    REPORT.mkdir(parents=True,exist_ok=True);auth=read(REPORT/'AUTHORIZATION.json')
    for rel,h in auth['TRAIN_update_sources'].items():assert sha(ROOT/rel)==h
    assert sha(STUDY/'PROTOCOL.md')==auth['TRAIN_update_protocol_sha256']
    assert sha(STUDY/'PLAN.json')==auth['TRAIN_update_plan_sha256']
    previous.REPORT,previous.extend=REPORT,extend
    previous.main(check)
    if check:
        a=('osft_budget',dict(gpu=1,phase='train',view='svd_qv_r8_orthogonal'))
        assert choose(a,True,True,False,1)==a
        assert choose(a,False,False,False,1)==a
        assert choose(('acquired_anchor',4),False,True,False,1)==('acquired_anchor',4)
        assert choose(a,False,True,True,1)==a
        assert choose(a,False,True,False,None)==('wait',None)
        assert choose(a,False,True,False,0)==('TRAIN_update_pilot',dict(gpu=0))
        (REPORT/'PILOT_QUEUE_CPU_CHECK.json').write_text(json.dumps(dict(checks=6,CPU_only=True,GPU_started=False))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    try:main(p.parse_args().check)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
