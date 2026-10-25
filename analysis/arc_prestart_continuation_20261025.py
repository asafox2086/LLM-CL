"""One transparent continuation after a proven model-free resource refusal.

All eight scientific files and READY stay unchanged. Only five never-STARTED
phases run. The completed gate trajectory is preserved, not resumed or retrained.
The original failed dispatch is archived explicitly and verified in the final
continuation receipt, alongside the unchanged original verifier result.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,time,traceback

RUN=Path(os.environ['LLMCL_RUN'])
ASSET=Path(os.environ['LLMCL_ASSET'])
REMAINING=('train_original','train_shuffle','eval_gate','eval_original','eval_shuffle')
ATTEMPT_FILES=('train_original.log','train_original_EXIT.json','train_original_FAILURE.json','CONTROLLER_FAILURE.json')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(name,value):
    with (RUN/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v

def audit_sources(permit):
    assert sha(RUN/'READY.json')==permit['ready_sha256']
    ready=read(RUN/'READY.json')
    for name,h in ready['sources'].items():assert sha(RUN/name)==h,name
    assert sha(RUN/'prestart_continuation.py')==permit['supervisor_sha256']
    for name,h in permit['preserved_gate_files'].items():assert sha(RUN/name)==h,name
    return ready

def audit_attempt(permit):
    for name,h in permit['attempt_files'].items():
        assert sha(RUN/'preSTARTED_guard_attempt_20261025'/name)==h,name
    failure=read(RUN/'preSTARTED_guard_attempt_20261025/train_original_FAILURE.json')
    assert 'card=ids.inventory()[self.gpu];assert not card' in failure['traceback']
    assert 'common.py", line 28, in __init__' in failure['traceback']
    old_exit=read(RUN/'preSTARTED_guard_attempt_20261025/train_original_EXIT.json')
    assert old_exit['returncode']==1 and old_exit['pid']==permit['failed_preSTARTED_pid']

def main():
    permit=read(RUN/'PRESTART_CONTINUATION_PERMIT.json')
    assert permit['accepted_once'] and permit['remaining_phases']==list(REMAINING)
    assert permit['absolute_wall_deadline']==permit['first_ACTIVATION_unix']+18300
    ready=audit_sources(permit)
    assert not (RUN/'CONTINUATION_STARTED.json').exists()
    assert all(not (RUN/(phase+'_STARTED.json')).exists() for phase in REMAINING)
    assert read(RUN/'train_gate_EXIT.json')['returncode']==0
    who=read(RUN/'train_gate_STARTED.json')['identity'];proc=Path('/proc')/str(who['pid'])/'stat'
    assert not proc.exists() or proc.read_text().rsplit(') ',1)[1].split()[19]!=str(who['startticks'])
    assert permit['attempt_files'].keys()==dict.fromkeys(ATTEMPT_FILES).keys()
    dest=RUN/'preSTARTED_guard_attempt_20261025';dest.mkdir()
    for name,h in permit['attempt_files'].items():
        path=RUN/name;assert sha(path)==h and not (dest/name).exists()
        path.rename(dest/name);assert sha(dest/name)==h
    audit_attempt(permit)
    write('CONTINUATION_STARTED.json',dict(unix=time.time(),pid=os.getpid(),permit_sha256=sha(RUN/'PRESTART_CONTINUATION_PERMIT.json'),
        original_failed_dispatch_preserved=True,remaining_phases=list(REMAINING),
        absolute_wall_deadline=permit['absolute_wall_deadline'],unit=permit['continuation_unit']))
    fixed=module('arc_fixed_controller_for_launch',RUN/'controller.py')
    ids=module('arc_continuation_ids',ASSET/'exp/paired_image_support_20261024/v1/common.py')
    waits=[]
    for phase in REMAINING:
        assert not (RUN/(phase+'_STARTED.json')).exists()
        audit_sources(permit);audit_attempt(permit)
        wait_started=time.monotonic();checks=[];consecutive=0
        while consecutive<2:
            assert time.time()<permit['absolute_wall_deadline'],'Original absolute deadline expired'
            gpu=ready['phase_GPUs'][phase];card=ids.inventory()[gpu]
            okay=not card['owners'] and card['memory_MiB']<500 and card['utilization']<10
            checks.append(dict(unix=time.time(),GPU=gpu,card=card,passed=okay))
            consecutive=consecutive+1 if okay else 0
            if consecutive<2:time.sleep(min(2,max(.001,permit['absolute_wall_deadline']-time.time())))
        deadline=time.monotonic()+max(.001,permit['absolute_wall_deadline']-time.time())
        wait=dict(phase=phase,seconds=time.monotonic()-wait_started,checks=checks)
        waits.append(wait);write('CONTINUATION_WAIT_'+phase+'.json',wait)
        fixed.launch(phase,ready['phase_GPUs'][phase],deadline)
    elapsed=time.time()-permit['first_ACTIVATION_unix'];assert elapsed<=18300
    phases=('train_gate',)+REMAINING
    ends={p:read(RUN/(p+'_COMPLETE.json')) for p in phases}
    assert sum(x['seconds'] for x in ends.values())<=18000
    audit_sources(permit);audit_attempt(permit)
    write('ALL_EXIT.json',dict(complete=True,ready_sha256=sha(RUN/'READY.json'),controller_seconds=elapsed,
        source_COMPLETE_sha256={p:sha(RUN/(p+'_COMPLETE.json')) for p in phases},
        new_generations=315,new_scores=0,FIT_backwards=3639,current_QA=3639,replay_QA=2658,
        optimizer_updates=261,TEST_QA=0,naive_training_repeated=False,
        preSTARTED_continuation=True,permit_sha256=sha(RUN/'PRESTART_CONTINUATION_PERMIT.json'),
        original_preSTARTED_failure_receipt=permit['observed_failure_sha256'],
        queue_wait_seconds=sum(x['seconds'] for x in waits),original_first_activation_unix=permit['first_ACTIVATION_unix']))
    # Seal the scheduling deviation before the one unchanged original scorer.
    write('CONTINUATION_MODEL_COMPLETE.json',dict(unix=time.time(),permit_sha256=sha(RUN/'PRESTART_CONTINUATION_PERMIT.json'),
        all_exit_sha256=sha(RUN/'ALL_EXIT.json'),archive={n:sha(dest/n) for n in ATTEMPT_FILES},
        original_gate_preserved=True,model_phase_count=6,unstarted_model_phases_run=5,
        first_activation_unix=permit['first_ACTIVATION_unix'],absolute_wall_deadline=permit['absolute_wall_deadline'],
        elapsed_since_first_activation=elapsed,queue_wait_seconds=sum(x['seconds'] for x in waits),
        source_and_READY_unchanged=True,old_preSTARTED_model_QA=0,new_quality_scores_so_far=0))
    assert not (RUN/'VERIFY_STARTED.json').exists() and not (RUN/'VERIFIED.json').exists()
    p=subprocess.run([sys.executable,'-u',str(RUN/'verify.py')],capture_output=True,text=True,
        timeout=min(180,max(.001,permit['absolute_wall_deadline']-time.time())))
    write('VERIFIER_EXIT.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,unix=time.time()))
    assert p.returncode==0 and (RUN/'VERIFIED.json').exists(),p.stderr
    v=read(RUN/'VERIFIED.json');assert v['complete'] and v['new_scores']==300
    audit_sources(permit);audit_attempt(permit)
    assert time.time()<=permit['absolute_wall_deadline']
    write('CONTINUATION_VERIFIED.json',dict(complete=True,unix=time.time(),
        VERIFIED_sha256=sha(RUN/'VERIFIED.json'),permit_sha256=sha(RUN/'PRESTART_CONTINUATION_PERMIT.json'),
        continuation_model_complete_sha256=sha(RUN/'CONTINUATION_MODEL_COMPLETE.json'),
        original_preSTARTED_attempt_archive={n:sha(dest/n) for n in ATTEMPT_FILES},
        original_unit_failed_preserved=True,preSTARTED_model_attempts=0,actual_model_phases=6,
        quality_scores=300,old_scientific_sources_unchanged=True,READY_unchanged=True,
        total_elapsed_since_original_activation=time.time()-permit['first_ACTIVATION_unix'],
        absolute_original_deadline_preserved=True,new_retries_of_started_model_phases=0,
        interpretation='One disclosed model-free resource refusal; five previously unstarted fixed phases completed under the original absolute deadline. Old failure evidence is preserved, not suppressed or rewritten. Original scoring executed exactly once.'))

if __name__=='__main__':
    try:main()
    except BaseException:
        write('CONTINUATION_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),no_retry=True))
        raise
