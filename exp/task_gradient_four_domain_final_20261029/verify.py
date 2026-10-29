"""Verify completed four-domain native score metadata; never rescore an answer."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

BINDING_SHA = '4c7a351c3e9873384652ad4b01d3cfbb0b15acb6b3b2613d80e4b1037863dd89'
WORKER_SHA = '7221a74e7553e7bf3a7ef26ad07e86037e56e4e177de63268d3637e4894daceb'
MANIFEST_SHA = '6ffae80b7e290b4bcd7b3ef8df7613c176a344221d28e874125f3faf87bd6798'
CP_SHA = '7a34ff44b996dbdff200d1125b13f4ef6b44328025e4df83f43ec7f30dc56acb'
DOMAINS = ['natural','medical','geography','finance']
METRICS = ['token_f1','exact_match','rougeL']


def check(ok,message):
    if not ok:raise ValueError(message)


def number(v):
    check(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=100,'invalid saved metric')
    return v


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        return None if s[0]=='Z' else dict(pid=int(pid),startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
    except (OSError,ValueError,IndexError):return None


class Files:
    def __init__(self):self.ledger={}
    def raw(self,p,expected=None):
        p=Path(p);st=p.stat();raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
        check(expected is None or expected==h,'SHA mismatch: '+str(p))
        check((st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns),'input changed: '+str(p))
        item=dict(sha256=h,bytes=len(raw),mtime_ns=st.st_mtime_ns)
        check(str(p) not in self.ledger or self.ledger[str(p)]==item,'inconsistent repeated input')
        self.ledger[str(p)]=item;return raw
    def json(self,p,expected=None):return json.loads(self.raw(p,expected))
    def unchanged(self):
        for p,v in self.ledger.items():
            st=Path(p).stat();check((st.st_size,st.st_mtime_ns)==(v['bytes'],v['mtime_ns']),'input changed before result')


def terminal(a,files):
    b=files.json(a.binding,BINDING_SHA);check(b['schema']=='task-gradient-four-domain-final-v1' and b['GPU']==1,'binding identity')
    check(b['worker_sha256']==WORKER_SHA,'frozen worker identity')
    root=Path(b['source_root']);report=a.report;run=a.run
    check(run==root/'exp/task_gradient_four_domain_final_20261029/runs/onevision/pct01/final_s16_v1','unique run')
    check(report==root/'summary/task_gradient_four_domain_final_20261029','supervisor report directory')
    check(not (run/'FAILURE.json').exists() and not (report/'SUPERVISOR_FAILURE.json').exists(),'attempt failed')
    port=Path(b['source_dir']);check(len(b['original_sources'])==29,'original 29 sources')
    for name,h in b['original_sources'].items():files.raw(port/name,h)
    files.raw(root/'exp/task_gradient_four_domain_final_20261029/worker.py',WORKER_SHA)
    ss=files.json(report/'SUPERVISOR_STARTED.json');ws=files.json(report/'WORKER_STARTED.json')
    wx=files.json(report/'WORKER_EXIT.json');se=files.json(report/'SUPERVISOR_COMPLETE.json')
    sx=files.json(a.supervisor_exit)
    check(ss['binding_sha256']==BINDING_SHA and ss['worker_sha256']==WORKER_SHA and ss['GPU']==1,'supervisor start provenance')
    files.raw(a.supervisor_source,ss['supervisor_sha256'])
    check(ss['identity']==ws['supervisor']==se['supervisor']==sx['identity'],'supervisor identity chain')
    check(ws['identity']==wx['identity']==se['worker'] and ws['GPU']==1,'worker identity chain')
    check(ws['command']==ss['command'] and '--execute' in ss['command'],'actual worker command')
    check(wx['returncode']==se['returncode']==sx['returncode']==0 and wx['identity_absent'] and sx['identity_absent'],'actual two process exit0')
    check(sx['binding_sha256']==BINDING_SHA,'external exit receipt binding')
    # External receipt must derive exit0 from systemd/setsid --wait, not the worker's claim.
    check(identity(ws['identity']['pid'])!=ws['identity'] and identity(ss['identity']['pid'])!=ss['identity'],'process still live')
    start=files.json(run/'STARTED.json');end=files.json(run/'COMPLETE.json',se['complete_sha256'])
    check(start['pid']==end['pid']==ws['identity']['pid'],'worker output PID')
    for v in (start,end):check(v['binding_sha256']==BINDING_SHA and v['worker_sha256']==WORKER_SHA and v['GPU']==1,'worker output binding')
    check(end['source_stage']==end['terminal_stage']==16 and end['new_generations']==19331 and end['new_training_updates']==0,'four-domain finite work')
    check(end['manifest_sha256']==MANIFEST_SHA and end['source_checkpoint_sha256']==CP_SHA,'data/prefix provenance')
    check(end['self_contained_final_graph'] and end['teacher_or_adapter_bank'] is False,'final graph declaration')
    fold=files.json(run/'FOLD.json');check(fold['source_stage']==fold['terminal_stage']==16 and fold['original_fold_function'] and fold['training_context_cleared'],'independent terminal16 fold')
    state=fold['method_state'];check(state['completed_stage']==state['initialized_stage']==16 and state['active_stage'] is None and state['deployed'],'retained stage16 counters')
    check(state['main_optimizer_coordinates']==state['structural_AB_coordinates']==0,'dense deployment graph')
    modules={f'model.layers.{i}.self_attn.{q}_proj' for i in range(24) for q in ['q','v']}
    check(len(fold['modules'])==48 and {v['module'] for v in fold['modules']}==modules,'48 original q/v merges')
    for v in fold['modules']:
        check(v['scale']==4 and v['merge_dtype']=='torch.float32' and v['output_dtype']=='torch.float16' and not v['autocast_enabled'] and not v['matmul_allow_tf32'],'original fold arithmetic')
        check(math.isfinite(v['rounding_max_abs']) and math.isfinite(v['rounding_rms']),'finite fold rounding metadata')
    check(end['dense_state']==fold['dense_state']==end['state_after_evaluation'],'same dense graph through final evaluation')
    config=files.json(run/'config.json');check(config['manifest_sha256']==MANIFEST_SHA and config['terminal_stage']==16 and config['source_checkpoint_sha256']==CP_SHA and config['postprocess_binding_sha256']==BINDING_SHA,'saved final config')
    cp=files.json(run/'checkpoints/stage_16.json');latest=files.json(run/'checkpoints/latest.json')
    check(cp==latest==end['checkpoint'] and cp['stage']==16 and cp['manifest_sha256']==MANIFEST_SHA and cp['full_architecture_and_runtime'],'saved final checkpoint metadata')
    check(cp['path']=='checkpoints/stage_16.pt' and (run/cp['path']).stat().st_size==cp['bytes'],'final checkpoint exists with declared bytes')
    manifest=files.json(b['manifest'],MANIFEST_SHA)
    check([t['task'] for t in manifest['tasks']]==list(range(1,17)) and {t['domain'] for t in manifest['tasks']}==set(DOMAINS),'all original sixteen tasks')
    check(sum(t['files']['test']['count'] for t in manifest['tasks'])==19331,'complete four-domain TEST')
    expected={f'evaluations/stage_16/T{t:02d}_test_predictions{s}' for t in range(1,17) for s in ['.jsonl','.scores.json','.datasets.json']}
    check(set(end['evaluation_files'])==expected,'complete sixteen prediction/score/dataset triples')
    return b,manifest,end,cp


def existing_scores(a,b,manifest,end,files):
    rows=[];seen=0
    for task in manifest['tasks']:
        t=task['task'];f=task['files']['test'];raw=files.raw(Path(b['source_root'])/f['path'],f['sha256'])
        source=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
        base=f'evaluations/stage_16/T{t:02d}_test_predictions'
        pred=[json.loads(line) for line in files.raw(a.run/(base+'.jsonl'),end['evaluation_files'][base+'.jsonl']).decode('utf-8').splitlines()]
        saved=files.json(a.run/(base+'.scores.json'),end['evaluation_files'][base+'.scores.json'])
        datasets=files.json(a.run/(base+'.datasets.json'),end['evaluation_files'][base+'.datasets.json'])
        check(len(source)==len(pred)==f['count']==saved['count'] and len({v['id'] for v in source})==len(source),'complete task identity denominator')
        by_dataset={}
        for src,p in zip(source,pred):
            check(p['id']==src['id'] and p['references']==src['answers'] and p['metadata']==src.get('metadata',{}) and p['generation_cap']==task['max_new_tokens'] and p['native_label']==task.get('native_label'),'same ordered original inputs/refs/caps')
            check(isinstance(p['prediction'],str),'raw output string')
            ds=src.get('dataset',src['id'].split(':',1)[0]);check(p['dataset']==ds,'dataset identity')
            values=by_dataset.setdefault(ds,[]);values.append(p)
            for metric in METRICS:number(p['scores'][metric])
        means={k:sum(p['scores'][k] for p in pred)/len(pred) for k in METRICS}
        check(all(saved[k]==means[k] for k in METRICS),'saved task aggregate differs from existing per-ID scores')
        check(saved['exact_match_correct']==sum(p['scores']['exact_match']==100 for p in pred),'saved exact-match count')
        check(set(datasets)==set(by_dataset),'dataset aggregate inventory')
        for ds,ps in by_dataset.items():
            v=datasets[ds];check(v['count']==len(ps) and v['exact_match_correct']==sum(p['scores']['exact_match']==100 for p in ps),'dataset count')
            check(all(v[k]==sum(p['scores'][k] for p in ps)/len(ps) for k in METRICS),'dataset saved score aggregation')
        rows.append(dict(task=t,domain=task['domain'],native_label=task.get('native_label'),QA=len(pred),F1=saved['token_f1'],exact_match=saved['exact_match'],rougeL=saved['rougeL']))
        seen+=len(pred)
    check(seen==19331,'total existing answer records')
    domains={}
    for d in DOMAINS:
        rs=[v for v in rows if v['domain']==d];n=sum(v['QA'] for v in rs)
        domains[d]=dict(QA=n,F1=sum(v['F1']*v['QA'] for v in rs)/n)
    return dict(final_native16=rows,domains=domains,domain_macro_final=sum(domains[d]['F1'] for d in DOMAINS)/4,
        AP16=sum(v['F1'] for v in rows)/16,QA_weighted_final=sum(v['F1']*v['QA'] for v in rows)/19331,
        original_AP19=None,records_checked=seen,new_scoring_calls=0,
        score_scope='Copied existing native per-ID scores; aggregate/input consistency checked, metric correctness not independently rescored')


def write(p,v):
    with p.open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--supervisor-exit',type=Path,required=True);ap.add_argument('--supervisor-source',type=Path,required=True)
    ap.add_argument('--write',action='store_true');a=ap.parse_args()
    check(all(p.is_absolute() for p in [a.binding,a.run,a.report,a.supervisor_exit,a.supervisor_source]),'absolute paths')
    files=Files();b,manifest,end,cp=terminal(a,files)
    if not a.write:print(json.dumps(dict(terminal_metadata_validated=True,predictions_read=False,new_scores=0)));return
    names=['FINAL_VERIFY_STARTED.json','FINAL_VERIFY_FAILURE.json','FINAL_VERIFIED.json','FINAL_RESULT.json']
    check(not any((a.report/n).exists() for n in names),'one independent existing-score consumption only')
    began=time.monotonic();own=sha(__file__)
    write(a.report/names[0],dict(unix=time.time(),pid=os.getpid(),verifier_sha256=own,binding_sha256=BINDING_SHA,new_score_calls=0))
    try:
        result=existing_scores(a,b,manifest,end,files)
        files.unchanged();check(sha(__file__)==own,'verifier changed')
        result.update(status='FOUR_DOMAIN_NATIVE_SCORE_METADATA_VERIFIED_DEPLOYMENT_PENDING',
            method='TaskGradReuse',backbone='onevision',condition='pct01',terminal_stage=16,
            binding_sha256=BINDING_SHA,verifier_sha256=own,worker_complete_sha256=sha(a.run/'COMPLETE.json'),
            final_checkpoint=cp,final_checkpoint_full_SHA_rechecked=False,independent_dense_reload_pending=True,
            independent_fixed_window_resource_pending=True,new_models=0,new_generations=0,
            limits=['S16 TRAIN-prefix equivalence is source-audited, not a fresh sixteen-task training process.',
                    'The original run evaluated some traffic inputs but no traffic TRAIN was used through S16.',
                    'Final F1 is not acquired F1, a forgetting metric, pure direction causality or a strong-baseline replacement proof.'])
        write(a.report/'FINAL_RESULT.json',result)
        write(a.report/'FINAL_VERIFIED.json',dict(unix=time.time(),status=result['status'],result_sha256=sha(a.report/'FINAL_RESULT.json'),
            verifier_sha256=own,binding_sha256=BINDING_SHA,source_ledger=files.ledger,existing_answer_records_checked=19331,
            new_score_calls=0,new_models=0,new_generations=0,seconds=time.monotonic()-began,
            independent_reload_and_resource_pending=True))
        print(json.dumps({'verified':str(a.report/'FINAL_VERIFIED.json'),'new_scores':0}))
    except BaseException:
        write(a.report/'FINAL_VERIFY_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),verifier_sha256=own,new_score_calls=0,no_automatic_retry=True));raise


if __name__=='__main__':main()
