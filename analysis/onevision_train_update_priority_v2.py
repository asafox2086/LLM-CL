"""Operational priority amendment; scientific workers/protocol remain frozen.

After the current GPU finishes naturally, run the existing TRAIN-only pilot
before other queued full confirmations, then return to all original work.
No score, prediction, rank, LR, data, budget or stopping rule is selected here.
"""
import argparse
import json
import os
import traceback

import onevision_train_update_pilot_queue as previous
import onevision_research_device_entry as device

ROOT=previous.ROOT
REPORT=previous.STUDY/'priority_v2'
OLD_EXTEND=previous.extend


def choose(action,inherited,trains_ready,pilot_done,gpu):
    if inherited or action[0]=='acquired_anchor' or not trains_ready or pilot_done:return action
    if gpu is None:return 'wait',None
    return 'TRAIN_update_pilot',dict(gpu=gpu)


def extend(original,native_spec):
    OLD_EXTEND(original,native_spec)
    warm=original.prior.cold.previous
    old_next=warm.next_action
    def next_action(*args):
        action,state=old_next(*args)
        if state.get('waiting_for_natural_completion'):return action,state
        coeff=previous.previous
        spec=coeff.read(coeff.STUDY/'CPU_PLAN.json')['spec']
        ready=all(coeff.reporter.complete('train',view,spec) for view in coeff.reporter.VIEWS)
        done=previous.complete()
        state.update(TRAIN_pilot_before_queued_full_confirmations=True,
                     all_original_fulls_still_required=True,no_score_dependent_ordering=True)
        return choose(action,False,ready,done,device.select(device.inventory(),os.getuid())),state
    warm.next_action=next_action
    # Same frozen pilot command and verifier already installed by parent.
    assert warm.command_for('TRAIN_update_pilot',dict(gpu=1),None)[0]==['analysis/onevision_train_only_update_pilot.py','--gpu','1']


def main(check=False):
    REPORT.mkdir(parents=True,exist_ok=True)
    auth=previous.read(REPORT/'AUTHORIZATION.json')
    for rel,h in auth['priority_v2_sources'].items():assert previous.sha(ROOT/rel)==h
    assert previous.sha(REPORT/'SCHEDULE_AMENDMENT.md')==auth['priority_v2_schedule_sha256']
    previous.REPORT,previous.extend=REPORT,extend
    previous.main(check)
    if check:
        action=('directional_coeff',dict(gpu=1,phase='full',view='random_coeff_r16'))
        assert choose(action,True,True,False,1)==action
        assert choose(action,False,False,False,1)==action
        assert choose(action,False,True,True,1)==action
        assert choose(action,False,True,False,None)==('wait',None)
        assert choose(action,False,True,False,0)==('TRAIN_update_pilot',dict(gpu=0))
        assert choose(('acquired_anchor',4),False,True,False,1)==('acquired_anchor',4)
        (REPORT/'PRIORITY_CPU_CHECK.json').write_text(json.dumps(dict(checks=6,CPU_only=True,
            old_pilot_worker_and_scientific_plan_unchanged=True,all_old_fulls_retained=True,GPU_started=False))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    try:main(parser.parse_args().check)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
