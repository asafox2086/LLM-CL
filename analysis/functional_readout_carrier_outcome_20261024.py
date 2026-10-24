"""Consume the already sealed new proof once; no Torch, QA, scoring or checkpoint reads."""
import hashlib,json,math,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/final_weight_reduction_20261024/functional_readout_carrier_v1'
WORK=ROOT/'exp/functional_readout_carrier_20261024/v1/private'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    assert not p.exists();p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    assert not (REPORT/'OUTCOME.json').exists();tick=time.monotonic()
    proof=json.loads((REPORT/'VERIFIED.json').read_bytes());assert proof['complete']
    quality=proof['quality'];parent=proof['parent_copied_verified_scalars'];selected=proof['retained_carrier']
    a={row['id']:row['F1'] for row in quality['WeightD']['per_id']}
    b={row['id']:row['F1'] for row in quality['ResponseD']['per_id']};assert a.keys()==b.keys()
    costs={}
    for case in ('init','WeightD','ResponseD'):
        comp=json.loads((REPORT/(case+'_COMPLETE.json')).read_bytes());exit=json.loads((REPORT/(case+'_EXIT.json')).read_bytes())
        assert sha(REPORT/(case+'_COMPLETE.json'))==proof['case_complete_sha256'][case]
        rows=[json.loads(x) for x in (WORK/case/'resources.jsonl').read_text().splitlines()]
        costs[case]=dict(RUN_seconds=comp['seconds'],exit_observation_seconds=exit['observed_elapsed_seconds'],
            sample_count=len(rows),whole_phase_sampled_RSS_bytes_max=max(r['VmRSS_bytes'] for r in rows),
            not_controlled_inference_peak_or_speed=True)
        for name in ('layout','checkpoint','teacher_QA','moment_QA','diagnostic_QA','moment_tensor_bytes'):
            if name in comp:costs[case][name]=comp[name]
    diag=json.loads((WORK/'init/local_diagnostic.json').read_bytes())
    outcome=dict(source_VERIFIED_sha256=sha(REPORT/'VERIFIED.json'),consumer_sha256=sha(Path(__file__)),
        quality=quality,parent_copied_verified_scalars=parent,feasible_by_five_domain_point_floor=proof['feasible_by_five_domain_point_floor'],
        retained_carrier=selected,delta_ResponseD_minus_WeightD_pp=quality['ResponseD']['macro']-quality['WeightD']['macro'],
        paired_id_counts=dict(higher=sum(b[k]>a[k] for k in a),lower=sum(b[k]<a[k] for k in a),same=sum(b[k]==a[k] for k in a)),
        local_diagnostic=diag,costs=costs,total_model_RUN_seconds=sum(v['RUN_seconds'] for v in costs.values()),
        total_exit_observation_seconds=sum(v['exit_observation_seconds'] for v in costs.values()),
        cost_note='RUN and exit observations overlap; CPU admission, verification, transport, implementation and history are separate.48 teacher decoder passes,200 DEV generations,0 updates.',
        not_continual_learning_OSFT_causality_novelty_or_equivalence=True,consumer_seconds_before_write=time.monotonic()-tick)
    write(REPORT/'OUTCOME.json',outcome)
    write(REPORT/'DECISION.json',dict(action='retain_starting_carrier_for_separate_adaptation_necessity_review' if selected else 'close_fixed_D_only_rank200_point',
        retained_carrier=selected,source_OUTCOME_sha256=sha(REPORT/'OUTCOME.json'),
        no_automatic_training_confirmation_or_Ours=True,no_rank_LR_QA_recovery_or_time_rescue=True,goal_achieved=False))
    lines=['# Compact readout starting-function result','',
        'Original G/U and all other weights remain intact; only24 D projections were physically replaced. Prior art supplies tools, rather than a veto or automatic originality.',
        '', '| State | Five-domain Answer F1 macro | Final full weight coefficients |', '|---|---:|---:|',
        f"| Unadapted parent | {parent['macro']:.10f} | 893675552 |"]
    for arm in ('WeightD','ResponseD'):lines.append(f"| {arm} | {quality[arm]['macro']:.10f} | 816728096 |")
    lines+=['',f'Retained carrier: {selected or "none; fixed point closed"}. All domain point floors and100 ID scalars are in OUTCOME.json.',
        '', 'Local diagnostics use parent inputs. They do not establish whole-decoder fidelity, preserved knowledge, finite-budget learning, OSFT/PAS mechanism or equivalence.',
        'A retained known tool is only a usable start for a separate continual-readout question. Neither arm passing closes this D-only cost point without rotating to more matrices.',
        'Explored DEV, one seed,20QA/domain,89 known groups; no new holdout or statistical stability claim. No official TEST or trained1313 teacher.',
        '', f"Model RUN sum {outcome['total_model_RUN_seconds']:.6f}s; successful exit-observation sum {outcome['total_exit_observation_seconds']:.6f}s overlaps it. All48 teacher passes and200 DEV generations counted; no optimizer updates.",
        'Final complete weights, ordinary buffers and checkpoint bytes are separate. Whole-phase resource samples are not controlled inference peaks or speed.', '']
    path=REPORT/'README.md';assert not path.exists();path.write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(retained_carrier=selected,OUTCOME_sha256=sha(REPORT/'OUTCOME.json'),seconds=outcome['consumer_seconds_before_write'])))


if __name__=='__main__':main()
