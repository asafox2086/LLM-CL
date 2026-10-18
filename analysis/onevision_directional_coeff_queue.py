"""Prioritize direct lightweight trials; preserve current GPU and all old work."""
import argparse
import json
import os
from pathlib import Path
import traceback

import onevision_osft_budget_queue_v3 as previous
import onevision_directional_coeff_report as reporter
import onevision_research_device_entry as device

ROOT=previous.ROOT
STUDY=reporter.REPORT
REPORT=STUDY/'queue'
ENTRY='analysis/onevision_directional_coeff_entry.py'
OLD_EXTEND=previous.extend
read,sha=reporter.read,reporter.sha


def choose(action,inherited,trains,fulls,gpu):
    if inherited or action[0]=='acquired_anchor' or all(fulls):return action
    if gpu is None:return 'wait',None
    if not all(trains):return 'directional_coeff',dict(gpu=gpu,phase='train',view=reporter.VIEWS[trains.index(False)])
    return 'directional_coeff',dict(gpu=gpu,phase='full',view=reporter.VIEWS[fulls.index(False)])


def extend(original,native_spec):
    OLD_EXTEND(original,native_spec)
    spec=read(STUDY/'CPU_PLAN.json')['spec']
    warm=original.prior.cold.previous
    old_next,old_command,old_verify=warm.next_action,warm.command_for,warm.previous.verify_worker
    def next_action(*args):
        inherited=REPORT/'COEFF_INHERITED_WORKER.json'
        if inherited.exists() and not (REPORT/'COEFF_INHERITED_COMPLETE.json').exists():
            item=read(inherited)
            try:
                stat=(Path('/proc')/str(item['worker_pid'])/'stat').read_text().split(') ',1)[1].split()
                ticks=int(stat[19]) if stat[0]!='Z' else None
            except (FileNotFoundError,ProcessLookupError):ticks=None
            if ticks is not None and ticks==item['worker_startticks']:
                return ('wait',None),dict(waiting_for_natural_completion=True,inherited_worker=item)
            assert old_verify(item['kind'],item['target'],args[0],args[3],args[4])
            (REPORT/'COEFF_INHERITED_COMPLETE.json').write_text(json.dumps(dict(verified=True,worker=item))+'\n')
        action,state=old_next(*args)
        if state.get('waiting_for_natural_completion'):return action,state
        assert not list(STUDY.glob('*needs_attention.json'))
        trains=[reporter.complete('train',v,spec) for v in reporter.VIEWS]
        fulls=[reporter.complete('full',v,spec) for v in reporter.VIEWS]
        reporter.main()
        state.update(directional_coeff_train_complete=trains,directional_coeff_full_complete=fulls,
                     all_original_jobs_and_fulls_retained=True,new_method_before_old_fine_causal_controls=True)
        return choose(action,False,trains,fulls,device.select(device.inventory(),os.getuid())),state
    def command(kind,target,specs):
        if kind!='directional_coeff':return old_command(kind,target,specs)
        return [ENTRY,'--gpu',str(target['gpu']),'--phase',target['phase'],'--view',target['view']],STUDY/(target['phase']+'_'+target['view']+'_nohup.log')
    def verify(kind,target,*args):
        if kind!='directional_coeff':return old_verify(kind,target,*args)
        assert reporter.complete(target['phase'],target['view'],spec)
        run=ROOT/read(STUDY/(target['phase']+'_'+target['view']+'_complete.json'))['run']
        receipt=read(run/'physical_device_execution.json')
        assert receipt['physical_GPU']==target['gpu'] and not receipt['inventory'][str(target['gpu'])]['owners']
        reporter.main();return True
    warm.next_action,warm.command_for,warm.previous.verify_worker=next_action,command,verify
    assert command('directional_coeff',dict(gpu=1,phase='train',view=reporter.VIEWS[0]),None)[0]==[ENTRY,'--gpu','1','--phase','train','--view',reporter.VIEWS[0]]
    assert command('osft_budget',dict(gpu=0,phase='train',view=previous.reporter.VIEWS[1]),None)==old_command('osft_budget',dict(gpu=0,phase='train',view=previous.reporter.VIEWS[1]),None)


def main(check=False):
    REPORT.mkdir(parents=True,exist_ok=True)
    auth=read(REPORT/'AUTHORIZATION.json')
    for rel,h in auth['directional_coeff_sources'].items():assert sha(ROOT/rel)==h
    assert sha(STUDY/'PROTOCOL.md')==auth['directional_coeff_protocol_sha256']
    previous.REPORT,previous.extend=REPORT,extend
    previous.main(check)
    if check:
        old=('osft_budget',dict(gpu=0,phase='train',view=previous.reporter.VIEWS[1]))
        assert choose(old,True,[False]*2,[False]*2,1)==old
        assert choose(('acquired_anchor',4),False,[False]*2,[False]*2,1)==('acquired_anchor',4)
        assert choose(old,False,[False]*2,[False]*2,None)==('wait',None)
        assert choose(old,False,[False]*2,[False]*2,1)[1]['view']==reporter.VIEWS[0]
        assert choose(old,False,[True,False],[False]*2,0)[1]['view']==reporter.VIEWS[1]
        assert choose(old,False,[True]*2,[False]*2,1)[1]['phase']=='full'
        assert choose(old,False,[True]*2,[True,False],1)[1]['view']==reporter.VIEWS[1]
        assert choose(old,False,[True]*2,[True]*2,1)==old
        (REPORT/'COEFF_QUEUE_CPU_CHECK.json').write_text(json.dumps(dict(CPU_only=True,checks=8,actual_old_and_new_commands=2,no_GPU_started=True))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    try:main(a.check)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
