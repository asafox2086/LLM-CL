"""Finish all three neutral controls, then the fixed TRAIN-feedback experiment."""
import argparse,json,os,traceback
from pathlib import Path
import onevision_function_neutral_init_queue_v2 as previous
import onevision_answer_compatibility_report as reporter
import onevision_research_device_entry as device

ROOT=previous.ROOT
STUDY=reporter.REPORT
REPORT=STUDY/'queue'
ENTRY='analysis/onevision_answer_compatibility_entry.py'
OLD_EXTEND=previous.extend
read,sha=reporter.read,reporter.sha


def choose(action,inherited,neutral_done,built,data_done,full_done,gpu):
    if inherited or action[0]=='acquired_anchor' or not all(neutral_done):return action
    if built and (not all(data_done) or all(full_done)):return action
    if gpu is None:return ('wait',None)
    if not built:return ('answer_compatibility',dict(gpu=gpu,phase='build',view=None))
    return ('answer_compatibility',dict(gpu=gpu,phase='full',view=reporter.VIEWS[full_done.index(False)]))


def extend(original,native_spec):
    OLD_EXTEND(original,native_spec)
    spec=read(STUDY/'CPU_PLAN_CHECK.json')['spec'];warm=original.prior.cold.previous
    old_next,old_command,old_verify=warm.next_action,warm.command_for,warm.previous.verify_worker
    data_spec=original.prior.cold.study_plan()
    def next_action(*args):
        inherited=REPORT/'FEEDBACK_INHERITED_WORKER.json'
        if inherited.exists() and not (REPORT/'FEEDBACK_INHERITED_COMPLETE.json').exists():
            item=read(inherited);p=Path('/proc')/str(item['worker_pid'])
            try:
                s=(p/'stat').read_text();ticks=int(s[s.rfind(')')+2:].split()[19])
            except (FileNotFoundError,ProcessLookupError):ticks=None
            if ticks==item['worker_startticks']:
                return ('wait',None),dict(waiting_for_natural_completion=True,feedback_inherited_worker=item)
            assert old_verify(item['kind'],item['target'],args[0],args[3],args[4])
            (REPORT/'FEEDBACK_INHERITED_COMPLETE.json').write_text(json.dumps(dict(verified=True,original_worker=item))+'\n')
        action,state=old_next(*args)
        if state.get('waiting_for_natural_completion'):return action,state
        assert not list(STUDY.glob('*needs_attention.json')),'Preserve identified fault; no retry'
        neutral_spec=read(previous.STUDY/'CPU_PLAN_CHECK.json')['spec']
        neutral=[previous.reporter.complete(i,neutral_spec) for i in range(3)]
        built=reporter.complete('build',None,spec)
        data_done=[original.prior.cold.study_complete(i,data_spec) for i in range(18)]
        full=[reporter.complete('full',view,spec) for view in reporter.VIEWS]
        reporter.main()
        new=choose(action,False,neutral,built,data_done,full,device.select(device.inventory(),os.getuid()))
        state.update(neutral_controls_complete=neutral,answer_feedback_build_complete=built,
                     answer_feedback_full_complete=full,old_data_controls_complete=data_done,
                     both_feedback_full_deferred_until_old18_complete=True,all_original_jobs_retained=True)
        return new,state
    def command(kind,target,specs):
        if kind!='answer_compatibility':return old_command(kind,target,specs)
        cmd=[ENTRY,'--gpu',str(target['gpu']),'--phase',target['phase']]
        if target['view']:cmd+=['--view',target['view']]
        return cmd,STUDY/(target['phase']+'_'+str(target['view'])+'_nohup.log')
    def verify(kind,target,*args):
        if kind!='answer_compatibility':return old_verify(kind,target,*args)
        assert reporter.complete(target['phase'],target['view'],spec)
        name='build' if target['phase']=='build' else 'full_'+target['view']
        proof=read(STUDY/(name+'_complete.json'));receipt=read(ROOT/proof['run']/'physical_device_execution.json')
        assert receipt['physical_GPU']==target['gpu'] and not receipt['admission_inventory'][str(target['gpu'])]['owners']
        reporter.main();return True
    warm.next_action,warm.command_for,warm.previous.verify_worker=next_action,command,verify
    assert command('answer_compatibility',dict(gpu=0,phase='build',view=None),None)[0]==[ENTRY,'--gpu','0','--phase','build']
    assert command('neutral_initialization',dict(gpu=1,index=2),None)==old_command('neutral_initialization',dict(gpu=1,index=2),None)


def main(check=False):
    REPORT.mkdir(parents=True,exist_ok=True);auth=read(REPORT/'AUTHORIZATION.json')
    for n,s in auth['answer_compatibility_sources'].items():assert sha(ROOT/n)==s
    assert sha(STUDY/'PROTOCOL.md')==auth['answer_compatibility_protocol_sha256']
    spec=read(STUDY/'CPU_PLAN_CHECK.json')['spec']
    for n,s in spec['source_sha256'].items():assert sha(ROOT/n)==s
    previous.REPORT=REPORT;previous.extend=extend
    previous.main(check)
    if check:
        a=('data_factorial',3)
        assert choose(a,True,[True]*3,False,[False]*18,[False]*2,1)==a
        assert choose(a,False,[True,True,False],False,[False]*18,[False]*2,1)==a
        assert choose(('acquired_anchor',4),False,[True]*3,False,[False]*18,[False]*2,1)==('acquired_anchor',4)
        assert choose(a,False,[True]*3,False,[False]*18,[False]*2,None)==('wait',None)
        assert choose(a,False,[True]*3,False,[False]*18,[False]*2,0)[1]==dict(gpu=0,phase='build',view=None)
        assert choose(a,False,[True]*3,True,[False]*18,[False]*2,1)==a
        assert choose(a,False,[True]*3,True,[True]*18,[False]*2,1)[1]['view']=='cache_AnswerF1'
        assert choose(a,False,[True]*3,True,[True]*18,[True,False],1)[1]['view']=='cache_AnswerF1_metric'
        assert choose(a,False,[True]*3,True,[True]*18,[True]*2,1)==a
        (REPORT/'FEEDBACK_QUEUE_CPU_CHECK.json').write_text(json.dumps(dict(CPU_only=True,checks=9,
            new_and_old_command_checks=2,GPU_started=False))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    try:main(a.check)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
