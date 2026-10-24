"""Consume new verified scalar measurements once; no model or original score."""
from pathlib import Path
from statistics import fmean
import hashlib,json,time

def main():
    begin=time.perf_counter();root=Path(__file__).resolve().parents[1]
    folder=root/'summary/final_weight_reduction_20261024/paired_image_support_diagnostic_v1'
    output=folder/'MODEL_OUTCOME.json';assert not output.exists()
    path=folder/'MODEL_VERIFIED.json';proof=json.loads(path.read_bytes())
    assert proof['total_NLL_requests']==488 and proof['QA']==61 and proof['new_F1_scores']==proof['new_bootstrap']==0
    states=('qv_before','qv_after','osft_before','osft_after');domains={}
    for domain,summary in proof['domains'].items():
        rows=[r for r in proof['results'] if r['domain']==domain]
        domains[domain]=dict(summary=summary,state_NLL_means={s:{k:fmean(r['state_NLL'][s][k] for r in rows)
            for k in ('original_NLL','wrong_NLL','C')} for s in states})
    costs={}
    for state in states:
        c=json.loads((folder/(state+'_COMPLETE.json')).read_bytes());ex=json.loads((folder/(state+'_ACTUAL_EXIT.json')).read_bytes())
        costs[state]=dict(model_recorded_RUN_seconds=c['model_RUN_seconds'],observed_exit_bound_seconds=ex['observed_elapsed_bound_seconds'],
            NLL_requests=c['NLL_requests'],sampled_process_peaks_bytes=c['observed_sample_peaks_bytes'])
    value=dict(verified_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Only existing new verified scalar means/costs; no re-evaluation, score, pairing or bootstrap.',
        domains=domains,five_domain_equal_macro=proof['five_domain_equal_macro'],
        candidate_direction_checks=proof['candidate_direction_checks'],all_candidate_directions=proof['all_candidate_directions'],
        states_cost=costs,sum_recorded_RUN_seconds=sum(c['model_recorded_RUN_seconds'] for c in costs.values()),
        sum_observed_exit_bound_seconds=proof['total_model_observed_exit_bound_seconds'],
        transport=json.loads((folder/'MODEL_TRANSPORT_EXECUTION.json').read_bytes()),
        metadata_freeze=json.loads((folder/'MODEL_FREEZE_EXECUTION.json').read_bytes()),
        new_model_or_QA_requests=0,new_F1_or_bootstrap=0,seconds=time.perf_counter()-begin)
    with output.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(outcome_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),seconds=value['seconds'],domains=domains),ensure_ascii=False))

if __name__=='__main__':main()
