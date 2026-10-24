"""Consume sealed aggregate scores once; no prediction reading or scoring."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import os
import time

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/final_weight_reduction_20261024/osft_initial_gap_check_v1'
PROFILE=ROOT/'summary/final_weight_reduction_20261024/response_failure_profile_v1/EVIDENCE.json'
ACQUISITION=ROOT/'summary/onevision_diagnostics_20261016/osft_acquisition_20261018/EVIDENCE.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(name,value):
    path=REPORT/name
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def contrast(seq,osft):
    initial=osft['initial']-seq['initial']
    acquired=osft['acquired']-seq['acquired']
    final=osft['final']-seq['final']
    return dict(SeqFT=seq,OSFT=osft,initial_gap_pp=initial,acquired_gap_pp=acquired,
        acquired_gap_minus_initial_gap_pp=acquired-initial,
        final_gap_pp=final,final_gap_minus_initial_gap_pp=final-initial,
        SeqFT_trajectory_change_to_acquisition_pp=seq['acquired']-seq['initial'],
        OSFT_trajectory_change_to_acquisition_pp=osft['acquired']-osft['initial'])


def main():
    admission=json.loads((REPORT/'ADMISSION.json').read_text())
    assert admission['decision']=='accept_single_sealed_aggregate_contrast_only'
    assert admission['consumer_sha256']==sha(Path(__file__))
    assert admission['question_sha256']==sha(REPORT/'QUESTION.md')
    for n,digest in admission['input_sha256'].items():assert sha(ROOT/n)==digest,n
    assert not any((REPORT/n).exists() for n in ('STARTED.json','EVIDENCE.json','EXECUTION.json'))
    write('STARTED.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),
        admission_sha256=sha(REPORT/'ADMISSION.json'),no_model_QA_predictions_or_scoring=True))
    begin=time.perf_counter()
    source=json.loads(PROFILE.read_text());assert source['complete'] and source['original_scores_copied_not_recomputed']
    prior=json.loads(ACQUISITION.read_text())
    assert source['new_scores']==source['new_models']==source['new_bootstraps']==0
    methods=source['methods'];names=('SeqFT_LoRA','OSFT');phases=('initial','acquired','final')
    assert all(set(methods[n]['task_profile'])=={str(i) for i in range(1,20)} for n in names)
    tasks=[]
    for task in range(1,20):
        rows=[methods[n]['task_profile'][str(task)] for n in names]
        counts={row[p]['QA'] for row in rows for p in phases};assert len(counts)==1
        values=[]
        for row in rows:
            v={p:row[p]['existing_F1_mean'] for p in phases}
            assert all(math.isfinite(x) and 0<=x<=100 for x in v.values());values.append(v)
        tasks.append(dict(task=task,QA=counts.pop(),**contrast(*values)))
    assert sum(t['QA'] for t in tasks)==21454
    means=[{p:sum(t[key][p] for t in tasks)/19 for p in phases} for key in ('SeqFT','OSFT')]
    task_equal=contrast(*means)
    for name,v in zip(names,means):
        assert abs(v['acquired']-prior['methods'][name]['acquired19_mean'])<1e-9
        assert abs(v['final']-prior['methods'][name]['AP19'])<1e-9
    domains={}
    assert set(methods[names[0]]['profile']['initial'])==set(methods[names[1]]['profile']['initial'])
    for domain in methods[names[0]]['profile']['initial']:
        values=[];counts=set()
        for name in names:
            rows={p:methods[name]['profile'][p][domain] for p in phases}
            counts.update(row['QA'] for row in rows.values())
            values.append({p:row['existing_F1_mean'] for p,row in rows.items()})
        assert len(counts)==1
        domains[domain]=dict(QA=counts.pop(),**contrast(*values))
    write('EVIDENCE.json',dict(complete=True,input_sha256=admission['input_sha256'],
        profile_proof_sha256={n:methods[n]['proof_sha256'] for n in names},
        consumer_sha256=sha(Path(__file__)),task_equal=task_equal,tasks=tasks,domains=domains,
        decision='positive_observed_trajectory_change_difference' if task_equal['acquired_gap_minus_initial_gap_pp']>0
            else 'nonpositive_observed_trajectory_change_difference',
        scope='Existing completed formal trajectories; acquired minus S0 includes prior-task transfer and current-stage effects. This is not an adjusted causal effect or direct knowledge measure.',
        no_claim_of_equal_initial_functions=True,no_new_original_scores_or_bootstraps=True,
        no_TEST_based_method_or_hyperparameter_selection=True,new_models=0,new_QA=0,new_predictions=0))
    seconds=time.perf_counter()-begin
    write('EXECUTION.json',dict(complete=True,pid=os.getpid(),consumer_seconds=seconds,
        input_files=2,raw_prediction_files=0,model_tensor_QA_score_calls=0,
        evidence_sha256=sha(REPORT/'EVIDENCE.json'),historical_model_costs_unchanged=True))
    print(json.dumps(dict(evidence_sha256=sha(REPORT/'EVIDENCE.json'),consumer_seconds=seconds,
        task_equal=task_equal,domains=domains),ensure_ascii=False))


if __name__=='__main__':main()
