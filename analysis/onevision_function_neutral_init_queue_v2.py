"""Preserve original five and live worker; then three compensated diagnostics."""
import argparse
import json
import os
from pathlib import Path
import traceback
import onevision_initialization_source_queue as previous
import onevision_function_neutral_init_report_v2 as reporter
import onevision_research_device_entry as device

ROOT=previous.ROOT
STUDY=ROOT/'summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v2'
REPORT=STUDY/'queue'
ENTRY='analysis/onevision_function_neutral_init_entry_v2.py'
OLD_EXTEND=previous.extend
read,sha=reporter.read,reporter.sha


def choose(action,inherited,originals_done,done,gpu):
    if inherited or action[0]=='acquired_anchor' or not all(originals_done):return action
    if all(done):return action
    if gpu is None:return ('wait',None)
    return ('neutral_initialization',dict(gpu=gpu,index=done.index(False)))


def extend(original,native_spec):
    OLD_EXTEND(original,native_spec)
    spec=read(STUDY/'CPU_PLAN_CHECK.json')['spec'];warm=original.prior.cold.previous
    old_next,old_command,old_verify=warm.next_action,warm.command_for,warm.previous.verify_worker
    def next_action(*args):
        inherited=REPORT/'NEUTRAL_INHERITED_WORKER.json'
        if inherited.exists() and not (REPORT/'NEUTRAL_INHERITED_COMPLETE.json').exists():
            item=read(inherited);p=Path('/proc')/str(item['worker_pid'])
            try:
                stat=(p/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
            except (FileNotFoundError,ProcessLookupError):ticks=None
            if ticks==item['worker_startticks']:
                return ('wait',None),dict(waiting_for_natural_completion=True,neutral_inherited_worker=item)
            assert old_verify(item['kind'],item['target'],args[0],args[3],args[4])
            (REPORT/'NEUTRAL_INHERITED_COMPLETE.json').write_text(json.dumps(dict(verified=True,original_worker=item))+'\n')
        action,state=old_next(*args)
        if state.get('waiting_for_natural_completion'):return action,state
        assert not list(STUDY.glob('*needs_attention.json')),'Preserve fault; no blind retry'
        oldspec=read(previous.STUDY/'CPU_PLAN_CHECK.json')['spec']
        originals=[previous.complete('case',i,oldspec) for i in range(5)]
        done=[reporter.complete(i,spec) for i in range(3)]
        if any(done):reporter.main()
        new=choose(action,False,originals,done,device.select(device.inventory(),os.getuid()))
        state.update(neutral_initialization_complete=done,original_five_complete=originals,
                     all_original_jobs_retained=True,formal_full_deferred_not_deleted=True)
        return new,state
    def command(kind,target,specs):
        if kind!='neutral_initialization':return old_command(kind,target,specs)
        return [ENTRY,'--gpu',str(target['gpu']),'--case-index',str(target['index'])],STUDY/(reporter.CASES[target['index']]+'_nohup.log')
    def verify(kind,target,*args):
        if kind!='neutral_initialization':return old_verify(kind,target,*args)
        assert reporter.complete(target['index'],spec)
        proof=read(STUDY/(reporter.CASES[target['index']]+'_complete.json'))
        receipt=read(ROOT/proof['run']/'physical_device_execution.json')
        assert receipt['physical_GPU']==target['gpu'] and not receipt['admission_inventory'][str(target['gpu'])]['owners']
        reporter.main();return True
    warm.next_action,warm.command_for,warm.previous.verify_worker=next_action,command,verify
    assert command('neutral_initialization',dict(gpu=0,index=2),None)[0]==[ENTRY,'--gpu','0','--case-index','2']
    assert command('initialization_source',dict(gpu=1,phase='case',index=4),None)==old_command('initialization_source',dict(gpu=1,phase='case',index=4),None)


def main(check=False):
    REPORT.mkdir(parents=True,exist_ok=True);auth=read(REPORT/'AUTHORIZATION.json')
    for n,s in auth['neutral_initialization_sources'].items():assert sha(ROOT/n)==s
    assert sha(STUDY/'PROTOCOL.md')==auth['neutral_protocol_sha256']
    spec=read(STUDY/'CPU_PLAN_CHECK.json')['spec']
    for n,s in spec['source_sha256'].items():assert sha(ROOT/n)==s
    previous.REPORT=REPORT;previous.extend=extend
    previous.main(check)
    if check:
        a=('data_factorial',2)
        assert choose(a,True,[True]*5,[False]*3,1)==a
        assert choose(a,False,[True]*4+[False],[False]*3,1)==a
        assert choose(('acquired_anchor',4),False,[True]*5,[False]*3,1)==('acquired_anchor',4)
        assert choose(a,False,[True]*5,[False]*3,None)==('wait',None)
        assert choose(a,False,[True]*5,[False]*3,0)[1]==dict(gpu=0,index=0)
        assert choose(a,False,[True]*5,[True,False,False],1)[1]['index']==1
        assert choose(a,False,[True]*5,[True,True,False],1)[1]['index']==2
        assert choose(a,False,[True]*5,[True]*3,1)==a
        (REPORT/'NEUTRAL_QUEUE_CPU_CHECK.json').write_text(json.dumps(dict(CPU_only=True,checks=8,
            new_and_old_command_checks=2,GPU_started=False))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    try:main(a.check)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
