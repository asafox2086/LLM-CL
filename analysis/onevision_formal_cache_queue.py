"""Admit the common-protocol candidate after current native GPU finishes."""
import argparse
import json
import os
from pathlib import Path
import traceback
import onevision_state_cache_flexible_queue as previous
import onevision_research_device_entry as device
import onevision_formal_cache_report as reporter

ROOT=previous.ROOT
STUDY=ROOT/'summary/onevision_diagnostics_20261016/formal_v2_state_cache_20261018/v1'
REPORT=STUDY/'queue'
ENTRY='analysis/onevision_formal_cache_entry.py'
OLD_EXTEND=previous.extend


def read(p):return json.loads(p.read_text())
def sha(p):return previous.previous.prior.sha(p)


def formal_spec():
    gate=read(STUDY/'CPU_INPUT_CHECK.json');assert gate['passed']
    spec=gate['spec'];smoke=read(STUDY/'CPU_PIPELINE_CHECK.json')
    assert smoke['CPU_only'] and not smoke['real_model_or_QA_used']
    assert smoke['worker_source_sha256']==sha(ROOT/'analysis/onevision_formal_state_cache.py')
    for n,s in spec['source_sha256'].items():assert sha(ROOT/n)==s
    for n,s in spec['shared_source_sha256'].items():assert sha(ROOT/'exp/two_backbone_native_cl_20261017_v2'/n)==s
    family=ROOT/'exp/onevision_formal_state_cache_20261018/v1'
    assert sha(family/'source_states/COMPLETE.json')==spec['source_export_sha256']
    assert sha(family/'SOURCE_REFERENCE155.json')==spec['formal_reference155_sha256']
    assert sha(STUDY/'PROTOCOL.md')==read(REPORT/'AUTHORIZATION.json')['formal_protocol_sha256']
    return spec


def complete(phase,view,spec):
    p=STUDY/(('build' if phase=='build' else 'full_'+view)+'_complete.json')
    if not p.exists():return False
    proof=read(p);assert proof['complete'] and proof['spec']==spec
    run=ROOT/proof['run'];assert run.resolve().is_relative_to((ROOT/'exp/onevision_formal_state_cache_20261018/v1').resolve())
    for rel,s in proof['artifact_sha256'].items():
        p=run/rel;assert p.resolve().is_relative_to(run.resolve()) and sha(p)==s
    if phase=='build':assert read(run/'original_exact_gate.json')['original_formal_SeqFT_exact']
    else:assert proof['full_QA']==21454 and read(run/'candidate_reload_exact.json')['passed']
    reporter.main();return True


def choose(action,inherited,built,full_done,gpu):
    if inherited or action[0]=='acquired_anchor':return action
    if gpu is None:return ('wait',None)
    if not built:return ('formal_cache',dict(gpu=gpu,phase='build',view=None))
    for v,done in zip(['cache_geometry','cache_quality'],full_done):
        if not done:return ('formal_cache',dict(gpu=gpu,phase='full',view=v))
    return action


def extend(original,native_spec):
    OLD_EXTEND(original,native_spec)
    spec=formal_spec();warm=original.prior.cold.previous
    old_next,old_command,old_verify=warm.next_action,warm.command_for,warm.previous.verify_worker
    def next_action(*args):
        # Use a separate inheritance receipt. Ancestors cannot verify the newer
        # flex_research kind; its verifier is only installed after their closure.
        inherited_path=REPORT/'FORMAL_INHERITED_WORKER.json'
        if inherited_path.exists() and not (REPORT/'FORMAL_INHERITED_COMPLETE.json').exists():
            inherited=read(inherited_path);p=Path('/proc')/str(inherited['worker_pid'])
            try:
                s=(p/'stat').read_text();ticks=int(s[s.rfind(')')+2:].split()[19])
            except (FileNotFoundError,ProcessLookupError):ticks=None
            if ticks==inherited['worker_startticks']:
                return ('wait',None),dict(waiting_for_natural_completion=True,formal_inherited_worker=inherited)
            assert old_verify(inherited['kind'],inherited['target'],args[0],args[3],args[4]), 'No complete inherited proof; no retry'
            (REPORT/'FORMAL_INHERITED_COMPLETE.json').write_text(json.dumps(dict(verified=True,original_worker=inherited,no_GPU_interrupt=True))+'\n')
        # Previous inherited-worker and failure checks always run first.
        action,state=old_next(*args)
        if state.get('waiting_for_natural_completion'):return action,state
        assert not list(STUDY.glob('*needs_attention.json'))
        built=complete('build',None,spec)
        done=[complete('full',v,spec) for v in spec['full_candidates']] if built else [False,False]
        gpu=device.select(device.inventory(),os.getuid())
        new=choose(action,False,built,done,gpu)
        state.update(formal_cache_build_complete=built,formal_cache_full_complete=done,
            formal_protocol_FP16=True,research_allowed_GPUs=[0,1],max_our_GPU_cards=3)
        return new,state
    def command(kind,target,specs):
        if kind!='formal_cache':return old_command(kind,target,specs)
        cmd=[ENTRY,'--gpu',str(target['gpu']),'--phase',target['phase']]
        if target['view']:cmd+=['--view',target['view']]
        return cmd,STUDY/(target['phase']+'_'+str(target['view'])+'_nohup.log')
    def verify(kind,target,*args):
        if kind!='formal_cache':return old_verify(kind,target,*args)
        assert complete(target['phase'],target['view'],spec)
        p=STUDY/(('build' if target['phase']=='build' else 'full_'+target['view'])+'_complete.json')
        receipt=read(ROOT/read(p)['run']/'physical_device_execution.json')
        assert receipt['physical_GPU']==target['gpu'] and not receipt['admission_inventory'][str(target['gpu'])]['owners']
        return True
    warm.next_action,warm.command_for,warm.previous.verify_worker=next_action,command,verify
    assert command('formal_cache',dict(gpu=0,phase='build',view=None),None)[0]==[ENTRY,'--gpu','0','--phase','build']
    assert command('data_factorial',0,None)==old_command('data_factorial',0,None)


def checks():
    a=('data_factorial',2);n=0
    assert choose(a,True,False,[False,False],1)==a;n+=1
    assert choose(('acquired_anchor',4),False,False,[False,False],1)==('acquired_anchor',4);n+=1
    assert choose(a,False,False,[False,False],None)==('wait',None);n+=1
    assert choose(a,False,False,[False,False],0)[1]==dict(gpu=0,phase='build',view=None);n+=1
    assert choose(a,False,True,[False,False],1)[1]['view']=='cache_geometry';n+=1
    assert choose(a,False,True,[True,False],1)[1]['view']=='cache_quality';n+=1
    assert choose(a,False,True,[True,True],1)==a;n+=1
    return n


def main(check=False):
    REPORT.mkdir(parents=True,exist_ok=True);auth=read(REPORT/'AUTHORIZATION.json')
    for n,s in auth['formal_new_sources'].items():assert sha(ROOT/n)==s
    formal_spec()
    previous.REPORT=REPORT;previous.extend=extend
    previous.main(check)
    if check:
        (REPORT/'FORMAL_CPU_CHECK.json').write_text(json.dumps(dict(CPU_only=True,checks=checks(),GPU_started=False))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    try:main(a.check)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
