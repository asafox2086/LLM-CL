"""Read sealed TaskGradReuse/Seq/PAS/OSFT acquisition aggregates; never score.

--root ABS_SCIENCE_ROOT --binding ABS_JSON --references ABS_JSON --output ABS_JSON
references is a list of exactly three objects: {method, root, proof, sha256};
method is SeqFT_LoRA, PASs_MoE or OSFT; proof is relative to its absolute root.
No future result SHA is hardcoded: the frozen binding pins the verifier, whose
successful VERIFIED receipt seals RESULT and its original formal proof.
Reads JSON metadata and aggregate .scores.json only, never QA/prediction/CP.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import time

DOMAINS = ('natural','medical','geography','finance','traffic')
REFERENCES = {'SeqFT_LoRA','PASs_MoE','OSFT'}
MANIFEST = 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
ENGINE_SHA = 'd7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a'
VERIFIER = 'analysis/task_gradient_formal_verify_20261028.py'
REPORT = 'summary/task_gradient_formal_20261028'


def check(ok,message):
    if not ok:raise ValueError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def scalar(value):
    check(isinstance(value,(int,float)) and not isinstance(value,bool)
          and math.isfinite(value) and 0<=value<=100,'invalid sealed F1')
    return value


def close(a,b,label):check(math.isclose(a,b,rel_tol=0,abs_tol=1e-10),label)


def relative(root,rel):
    p=PurePosixPath(rel)
    check(not p.is_absolute() and '..' not in p.parts and '\\' not in rel,'unsafe relative metadata path')
    return root/rel  # Read-only frozen data may be linked outside an overlay.


class Metadata:
    def __init__(self):self.ledger={}

    def raw(self,path,expected=None):
        path=Path(path);before=path.stat();raw=path.read_bytes();after=path.stat()
        stamp=(before.st_size,before.st_mtime_ns)
        check(stamp==(after.st_size,after.st_mtime_ns),'metadata changed during read')
        h=digest(raw);check(expected is None or h==expected,'sealed SHA mismatch: '+str(path))
        row=dict(sha256=h,bytes=len(raw),mtime_ns=before.st_mtime_ns)
        check(str(path) not in self.ledger or self.ledger[str(path)]==row,'metadata changed between reads')
        self.ledger[str(path)]=row
        return raw

    def json(self,path,expected=None):return json.loads(self.raw(path,expected))

    def unchanged(self):
        for path,row in self.ledger.items():
            st=Path(path).stat()
            check((st.st_size,st.st_mtime_ns)==(row['bytes'],row['mtime_ns']),'input changed before output')


def read_method(io,root,proof,method,manifest):
    c=proof['candidate'];run=c['run'];sources=proof['source_sha256']
    check(c['method']==method and c['backbone']=='onevision' and c['collector_sha256']==ENGINE_SHA,
          'same-backbone sealed original-engine candidate')
    check(proof['CPU_only'] and proof['primary_metric']=='existing raw Answer F1'
          and proof['rescored_cells']==74 and proof['final_QA']==21454
          and proof['positive_finite_training_stages']==19,'complete original formal proof')
    check(sources[MANIFEST]==MANIFEST_SHA,'same original manifest declaration')
    cfgrel=run+'/config.json'
    check(sources[cfgrel]==proof['config_sha256'],'config proof link')
    config=io.json(relative(root,cfgrel),proof['config_sha256'])
    check(config['method']==method and config['backbone']=='onevision' and config['condition']=='pct01'
          and config['manifest_sha256']==MANIFEST_SHA and config['seed']==42
          and config['epochs']==1 and config['batch_size']==16 and config['micro_batch_size']==1
          and config['train_fraction']==.01 and config['test_fraction']==1
          and not config['preflight_only'],'original formal setting')
    check(len(proof['final_native19'])==19 and set(proof['domains'])==set(DOMAINS),'complete final scalars')
    rows=[]
    for task in manifest['tasks']:
        t=task['task'];test=task['files']['test'];train=task['files']['train']
        check(sources[test['path']]==test['sha256'] and sources[train['path']]==train['sha256'],
              'same TRAIN/TEST SHA declarations; files not opened')
        check(config['task_counts'][task['task_id']]['test']==test['count']
              and config['task_counts'][task['task_id']]['train']==train['count'],'config task count')
        predrel=f'evaluations/stage_{t:02d}/T{t:02d}_test_predictions.jsonl'
        pred=run+'/'+predrel
        check(sources[pred]==proof['prediction_sha256'][predrel],'sealed acquired prediction declaration')
        score=pred.replace('.jsonl','.scores.json')
        saved=io.json(relative(root,score),sources[score])
        check(saved['count']==test['count'] and isinstance(saved['count'],int),'aggregate QA denominator')
        acquired=scalar(saved['token_f1']);final=scalar(proof['final_native19'][t-1])
        if t==19:close(acquired,final,'T19 acquired/final scalar')
        rows.append(dict(task=t,task_id=task['task_id'],domain=task['domain'],
            native_label=task['native_label'],TEST_QA=test['count'],TRAIN_QA=train['count'],
            TEST_sha256=test['sha256'],acquired_F1=acquired,final_F1=final,
            acquired_aggregate_sha256=sources[score],acquired_aggregate_path=str(relative(root,score))))
    domains={}
    for d in DOMAINS:
        group=[r for r in rows if r['domain']==d];n=sum(r['TEST_QA'] for r in group)
        a=sum(r['acquired_F1']*r['TEST_QA'] for r in group)/n
        f=sum(r['final_F1']*r['TEST_QA'] for r in group)/n
        check(proof['domains'][d]['QA']==n,'final domain denominator')
        close(f,scalar(proof['domains'][d]['F1']),'final task/domain consistency')
        domains[d]=dict(TEST_QA=n,acquired_F1=a,final_F1=f)
    final_ap=scalar(proof['aggregate']['AP']);final_macro=scalar(proof['domain_macro_final'])
    close(sum(r['final_F1'] for r in rows)/19,final_ap,'original final AP19')
    close(sum(domains[d]['final_F1'] for d in DOMAINS)/5,final_macro,'original final domain macro')
    return dict(candidate=c,tasks=rows,domains=domains,
        acquired_AP19=sum(r['acquired_F1'] for r in rows)/19,
        acquired_domain_macro_F1=sum(domains[d]['acquired_F1'] for d in DOMAINS)/5,
        original_final_AP19=final_ap,final_domain_macro_F1=final_macro,
        checkpoint_sha256_from_existing_proof=proof['checkpoint_sha256'])


def differences(candidate,reference):
    tasks=[]
    for a,b in zip(candidate['tasks'],reference['tasks']):
        for k in ('task','task_id','domain','TEST_QA','TEST_sha256'):check(a[k]==b[k],'paired task identity')
        tasks.append(dict(task=a['task'],domain=a['domain'],native_label=a['native_label'],
            TEST_QA=a['TEST_QA'],candidate_minus_reference_acquired_pp=a['acquired_F1']-b['acquired_F1'],
            candidate_minus_reference_final_pp=a['final_F1']-b['final_F1']))
    domains={d:dict(candidate_minus_reference_acquired_pp=candidate['domains'][d]['acquired_F1']-reference['domains'][d]['acquired_F1'],
        candidate_minus_reference_final_pp=candidate['domains'][d]['final_F1']-reference['domains'][d]['final_F1']) for d in DOMAINS}
    return dict(tasks=tasks,domains=domains,
        acquired_AP19_pp=candidate['acquired_AP19']-reference['acquired_AP19'],
        acquired_domain_macro_pp=candidate['acquired_domain_macro_F1']-reference['acquired_domain_macro_F1'],
        original_final_AP19_pp=candidate['original_final_AP19']-reference['original_final_AP19'],
        final_domain_macro_pp=candidate['final_domain_macro_F1']-reference['final_domain_macro_F1'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','binding','references','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    check(all(getattr(a,k).is_absolute() for k in ('root','binding','references','output')),'absolute CLI paths')
    check(a.root.is_dir() and not a.output.exists(),'existing science root/exclusive output')
    tick=time.monotonic();io=Metadata();root=a.root;bh=digest(io.raw(a.binding));binding=io.json(a.binding,bh)
    check(binding['schema']=='task-gradient-formal-v1' and binding['method']=='TaskGradReuse'
          and binding['manifest_sha256']==MANIFEST_SHA,'frozen formal binding identity')
    verifier_sha=binding['inherited_sources'][VERIFIER]
    io.raw(root/VERIFIER,verifier_sha)
    verified=io.json(root/REPORT/'VERIFIED.json')
    check(verified['status']=='VERIFIED_ONCE_RESOURCES_PENDING'
          and verified['verifier_sha256']==verifier_sha and verified['engine_sha256']==ENGINE_SHA
          and verified['binding_sha256']==bh and verified['original_formal_cells']==74
          and verified['existing_answer_score_checks']==80316 and verified['extra_acquisition_scores']==0
          and verified['new_models']==verified['new_generations']==0,'completed unique formal verification')
    result=io.json(root/REPORT/'RESULT.json',verified['result_sha256'])
    check(result['source_sha256']==verifier_sha and result['binding_sha256']==bh
          and result['method']=='TaskGradReuse' and result['backbone']=='onevision'
          and result['condition']=='pct01','sealed result identity')
    complete=io.json(root/'run_v1/FORMAL_COMPLETE.json',verified['formal_complete_sha256'])
    check(complete['binding_sha256']==bh and complete['source_sha256']==binding['sources']
          and complete['active_worker'] is None and all(x['returncode']==0 for x in complete['phases']),
          'same completed formal model source')
    check(result['formal_complete_sha256']==verified['formal_complete_sha256'],'same result/terminal receipt')
    manifest=io.json(root/MANIFEST,MANIFEST_SHA)
    check([t['task'] for t in manifest['tasks']]==list(range(1,20))
          and sum(t['files']['test']['count'] for t in manifest['tasks'])==21454,'all original tasks')
    methods={'TaskGradReuse':read_method(io,root,result['original_formal_proof'],'TaskGradReuse',manifest)}
    close(methods['TaskGradReuse']['acquired_AP19'],result['acquired']['AP19'],'candidate existing acquired AP19')
    close(methods['TaskGradReuse']['acquired_domain_macro_F1'],result['acquired']['domain_macro_F1'],
          'candidate existing acquired domain macro')
    specs=io.json(a.references)
    check(isinstance(specs,list) and len(specs)==3 and {s['method'] for s in specs}==REFERENCES,'exact three fixed references')
    for spec in specs:
        rr=Path(spec['root']);check(rr.is_absolute() and rr.is_dir(),'reference absolute root')
        proof=io.json(relative(rr,spec['proof']),spec['sha256'])
        methods[spec['method']]=read_method(io,rr,proof,spec['method'],manifest)
    comparisons={m:differences(methods['TaskGradReuse'],methods[m]) for m in sorted(REFERENCES)}
    io.unchanged()
    output=dict(schema='task-gradient-formal-acquisition-v1',status='SEALED_AGGREGATES_ONLY',methods=methods,
        candidate_minus_reference=comparisons,binding_sha256=bh,verifier_sha256=verifier_sha,
        source_script_sha256=digest(Path(__file__).read_bytes()),source_metadata=io.ledger,
        definitions=dict(original_final_AP19='mean_t R[19,t], the original aggregate.AP',
            acquired_AP19='mean_t R[t,t]; different times, not original aggregate.AP',
            acquired_domain_macro='within-domain TEST-QA-weighted R[t,t], then equal five-domain mean',
            final_domain_macro='within-domain TEST-QA-weighted R[19,t], then equal five-domain mean'),
        limits=['Different method trajectories; acquired differences include previous transfer and current learning.',
                'No causal attribution solely to directions, capacity or expert routing; no forgetting decomposition.',
                'One formal seed; task/domain differences are descriptive, not significance or equivalence tests.',
                'No resource inference: this consumer does not load weights or measure GPU/CPU memory.'],
        activity=dict(new_scores=0,new_models=0,new_generations=0,QA_or_prediction_rows_read=0,
                      checkpoint_payload_reads=0,existing_acquired_aggregate_files=76),
        elapsed_seconds_before_output=time.monotonic()-tick)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(a.output),sha256=digest(a.output.read_bytes()),new_scores=0)))


if __name__=='__main__':main()
