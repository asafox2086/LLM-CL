"""Consume the one sealed comparison. No tensors, QA text, score or resampling."""
import hashlib,json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/final_weight_reduction_20261024/gate_response_reuse_v1'
WORK=ROOT/'exp/gate_response_reuse_20261024/v1/private'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())


def main():
    tick=time.monotonic();target=REPORT/'OUTCOME.json';assert not target.exists()
    v=read(REPORT/'VERIFIED.json');assert v['complete'] and v['TEST_QA']==0
    assert v['ready_sha256']==sha(REPORT/'READY.json')
    complete={}
    for case in ('init','Static','Reuse'):
        p=REPORT/(case+'_COMPLETE.json');assert sha(p)==v['case_complete_sha256'][case]
        complete[case]=read(p);assert complete[case]['ready_sha256']==v['ready_sha256']
    quality=dict(parent=v['parent_copied_verified_scalars'],**v['quality'])
    static={x['id']:x for x in quality['Static']['per_id']}
    reuse={x['id']:x for x in quality['Reuse']['per_id']}
    assert set(static)==set(reuse) and len(static)==100
    paired=dict(higher=sum(reuse[i]['F1']>static[i]['F1'] for i in static),
                lower=sum(reuse[i]['F1']<static[i]['F1'] for i in static),
                same=sum(reuse[i]['F1']==static[i]['F1'] for i in static))
    details={};resources={}
    for case in ('Static','Reuse'):
        c=complete[case];p=WORK/case/'fit_summary.json'
        assert sha(p)==c['files'][case+'/fit_summary.json']
        fit=read(p);assert len(fit['layers'])==24
        after=[x['held_feature_after'] for x in fit['layers']];before=[x['held_feature_before'] for x in fit['layers']]
        details[case]=dict(complete_weights=c['layout']['weights'],ordinary_buffers=c['layout']['ordinary_buffers'],
            final_checkpoint=c['checkpoint'],final_trainable=c['layout']['trainable'],local_optimizer_updates=c['updates'],
            fit_sampled_states=c['fit_states'],fit_state_presentations=c['fit_states']*4*24,
            held_feature_mean_before=sum(before)/24,held_feature_mean_after=sum(after)/24,
            held_feature_scope='Equal-layer mean of24 normalized squared-error ratios; not global MSE or AnswerF1.',
            layers_held_feature_error_lower=sum(a<b for a,b in zip(after,before)),
            model_RUN_seconds=c['seconds'],complete_layout_and_strict_reload=True)
    for case in ('init','Static','Reuse'):
        p=WORK/case/'resources.jsonl';assert sha(p)==complete[case]['files'][case+'/resources.jsonl']
        rr=[json.loads(x) for x in p.read_text().splitlines()]
        assert all(x['pid']==complete[case]['worker']['pid'] for x in rr)
        keys=['VmRSS_bytes','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_process_nvml_bytes']
        resources[case]=dict(rows=len(rr),worker=complete[case]['worker'],
            sampled_max={k:max(x[k] for x in rr if x.get(k) is not None) if any(x.get(k) is not None for x in rr) else None for k in keys},
            scope='Whole observed process trajectory including loading/fitting/save/reload/evaluation; not a controlled inference peak or speed comparison.')
    domain_delta={d:quality['Reuse']['domains'][d]-quality['Static']['domains'][d] for d in quality['Static']['domains']}
    versus_parent={case:{d:quality[case]['domains'][d]-quality['parent']['domains'][d] for d in quality['parent']['domains']} for case in ('Static','Reuse')}
    passed=v['primary_point_pass']
    parent_floor=all(quality['Reuse']['domains'][d]>=value for d,value in quality['parent']['domains'].items())
    assert passed==(parent_floor and quality['Reuse']['macro']>quality['Static']['macro'])
    outcome=dict(complete=True,source_VERIFIED_sha256=sha(REPORT/'VERIFIED.json'),consumer_sha256=sha(Path(__file__)),
        quality=quality,Reuse_minus_Static_macro_pp=quality['Reuse']['macro']-quality['Static']['macro'],
        Reuse_minus_Static_domain_pp=domain_delta,versus_unadapted_parent_domain_pp=versus_parent,paired_reuse_vs_static=paired,
        details=details,resources=resources,teacher_capture_QA=complete['init']['capture_QA'],
        teacher_sampled_payload_bytes=complete['init']['payload_bytes'],new_DEV_generations=200,TEST_QA=0,
        primary_point_pass=passed,
        parent_floor_pass=parent_floor,
        decision='retain_compact_starting_carrier_for_next_necessity_review_only' if passed else 'close_fixed_gate_reuse_and_calibrated_control_conditions',
        costs=dict(model_RUN_seconds_sum=sum(c['seconds'] for c in complete.values()),
            successful_exit_observed_seconds_sum=v['observed_model_seconds_sum'],
            overlap='RUN and observed exits overlap, do not add. CPU admission/transport/verification/history additional.'),
        scientific_scope='Whole-model starting-function comparison after S1-only local fitting. No causal claim about cold-start/OSFT/PAS or future adaptation. No novel framework established.',
        limitations=['Original100DEV repeatedly explored, one seed,20QA per domain89known groups.',
            'Static has21504 more full weight coefficients; near-matched upper budget, not identical counts.',
            'Parent is unadapted and weak; retaining it does not establish strong trained baseline performance.',
            'The local block targets use teacher inputs; held feature fidelity does not certify compressed whole-decoder trajectories.',
            'Relative contributions of G/U correspondence, input drift, optimization and numerical paths were not isolated.',
            'No equivalence, noninferiority, stability, significance, seed uncertainty, new bootstrap or automatic full confirmation.'])
    outcome['consumer_seconds_before_write']=time.monotonic()-tick
    with target.open('x',encoding='utf8') as f:json.dump(outcome,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(primary_point_pass=passed,macro={k:x['macro'] for k,x in quality.items()},
        Reuse_minus_Static_pp=outcome['Reuse_minus_Static_macro_pp'],paired=paired,outcome_sha256=sha(target),
        consumer_seconds=outcome['consumer_seconds_before_write'])))


if __name__=='__main__':main()
