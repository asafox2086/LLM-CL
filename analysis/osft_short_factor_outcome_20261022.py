"""Once-only consumption of completed ShortFixed proof; no rescoring or bootstrap."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/edge_deployment_20261019/osft_short_factor_control_20261022/v2'
WORK = ROOT/'exp/osft_short_factor_control_20261022/v2/private'
SCREEN = ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    return json.loads(Path(p).read_text())

def main():
    started = time.monotonic()
    assert not (REPORT/'OUTCOME.json').exists()
    v = load(REPORT/'VERIFIED.json'); c = load(REPORT/'COMPLETE.json'); ready = load(REPORT/'READY.json')
    assert sha(REPORT/'VERIFIED.json') == '6289cfecb65f989e75ab8e031e9ee85700a4b18b5005d5ef0203b0789ea9d1fc'
    assert v['complete'] and v['complete_sha256'] == sha(REPORT/'COMPLETE.json')
    assert v['ready_sha256'] == c['ready_sha256'] == sha(REPORT/'READY.json')
    assert v['quality'] == c['quality'] and v['checkpoint'] == c['checkpoint']
    assert v['TEST_QA'] == 0 and v['updates'] == 87 and not v['automatic_full_confirmation']
    for name, h in ready['sources'].items():
        assert sha(ROOT/name) == h, name
    for p in [SCREEN/'SCREEN_COMPLETE.json', SCREEN/'SPLIT.json', REPORT/'CPU_GATE.json', REPORT/'V1_FAILURE_SEAL.json']:
        assert ready['evidence'][p.relative_to(ROOT).as_posix()] == sha(p)
    refs = load(SCREEN/'SCREEN_COMPLETE.json')['all_four_results']
    anchors = {name:refs[name] for name in ('OSFT_all', 'LoRA_qv')}
    threshold = max(x['domain_macro'] for x in anchors.values())
    quality_pass = v['quality']['domain_macro'] > threshold
    assert v['primary_quality_pass'] == quality_pass
    assert v['paired']['replicates'] == 10000 and v['paired']['seed'] == 20261022
    assert list(v['paired']['comparisons']) == ['OSFT_all', 'LoRA_qv']

    # Only copy already-verified scalar scores; do not call the scorer, models, or paired bootstrap.
    prediction_inputs = {}
    def scores(proof):
        out = {}
        for name, h in proof['prediction_sha256'].items():
            p = ROOT/name; assert sha(p) == h
            prediction_inputs[name] = h
            for line in p.read_text().splitlines():
                row = json.loads(line); i = row['id']; f = row['scores']['token_f1']
                assert i not in out and math.isfinite(f) and 0 <= f <= 100
                out[i] = f
        assert len(out) == 100
        return out
    all_scores = {'ShortFixed': scores(v['quality']), **{n:scores(p) for n,p in anchors.items()}}
    split = load(SCREEN/'SPLIT.json')
    identity = {i:(g['domain'], gid) for gid,g in split['groups'].items() if g['dev'] for i in g['ids']}
    assert len(identity) == 100 and all(set(s) == set(identity) for s in all_scores.values())
    qa = []
    for i,(domain,gid) in sorted(identity.items()):
        qa.append(dict(id_sha256=hashlib.sha256(i.encode()).hexdigest(), domain=domain,
            group_sha256=hashlib.sha256(gid.encode()).hexdigest(),
            scores={n:s[i] for n,s in all_scores.items()},
            differences_pp={n:all_scores['ShortFixed'][i]-s[i] for n,s in all_scores.items() if n!='ShortFixed'}))
    counts = {}
    for n,pair in v['paired']['comparisons'].items():
        counts[n] = {kind:sum(t[kind] for t in pair['QA_comparison'].values()) for kind in ('higher','lower','equal')}
        assert sum(counts[n].values()) == 100
        assert math.isclose(pair['macro_difference_pp'], v['quality']['domain_macro']-anchors[n]['domain_macro'], abs_tol=1e-10)
        for d in DOMAINS:
            delta = [r['differences_pp'][n] for r in qa if r['domain']==d]
            assert len(delta)==20 and math.isclose(sum(delta)/20,pair['domain_difference_pp'][d],abs_tol=1e-10)
            assert dict(higher=sum(x>0 for x in delta),lower=sum(x<0 for x in delta),equal=sum(x==0 for x in delta))==pair['QA_comparison'][d]

    optimizer = load(WORK/'OPTIMIZER_EVIDENCE.json'); stages = load(WORK/'SHORT_STAGE_EVIDENCE.json')
    assert sha(WORK/'OPTIMIZER_EVIDENCE.json') == c['optimizer_evidence_sha256']
    assert sha(WORK/'SHORT_STAGE_EVIDENCE.json') == c['short_stage_evidence_sha256']
    assert stages == c['short_stage_records'] and len(stages)==19 and len(optimizer)==87
    assert c['frozen_base_digest_before'] == c['frozen_base_digest_after']
    actual = []
    for plan,s in zip(ready['training_plan'], stages):
        assert plan['stage']==s['stage'] and s['updates']==plan['updates']
        assert s['short_frozen_exact'] and s['short_before']==s['short_after']
        w=s['first_step']; assert len(w['matrices'])==168
        rows=[o for o in optimizer if o['stage']==s['stage']]
        assert len(rows)==plan['updates']
        actual.append(dict(stage=s['stage'],updates=s['updates'],short_frozen_exact=True,
            first_step_squared_change=w['squared_change'],first_step_squared_before=w['squared_before'],
            changed_matrices=sum(t['changed_coordinates']>0 for t in w['matrices']),
            changed_weight_coordinates=sum(t['changed_coordinates'] for t in w['matrices']),
            after_step_observation_seconds=w['after_step_observation_seconds']))
    assert all(o['gradient_bytes']==357940224 and o['moment_bytes']==715880448 and o['parameter_tensors']==336
        and o['short_gradient_None'] and o['optimizer_parameter_IDs_match'] and o['all_moments_CPU'] for o in optimizer)
    old_grad, old_mom = 455064576, 910129152
    cost_pass=all(o['gradient_bytes']<old_grad and o['moment_bytes']<old_mom for o in optimizer)
    assert cost_pass == v['measured_gradient_and_moment_cost_pass']
    assert v['possible_fresh_FIT_PAS_reference_followup'] == (quality_pass and cost_pass)
    events=[json.loads(s) for s in (WORK/'logs/events.jsonl').read_text().splitlines()]
    assert len(events)==87 and sha(WORK/'logs/events.jsonl')==c['events_sha256']
    resources=[json.loads(s) for s in (WORK/'resources.jsonl').read_text().splitlines()]
    def phase(name):
        found=[r for r in resources if r['phase']==name];assert len(found)==1;return found[0]
    checkpoints=[load(p) for p in sorted((WORK/'checkpoints').glob('stage_??.json'))]
    assert [p['stage'] for p in checkpoints]==list(range(1,20)) and checkpoints[-1]==c['checkpoint']
    costs=dict(gradient_bytes_per_update=357940224,CPU_two_moment_bytes_per_update=715880448,
        original_source_implied_gradient_bytes=old_grad,original_source_implied_two_moment_bytes=old_mom,
        byte_reduction_fraction=1-357940224/old_grad,original_trainable_coordinates=113766144,
        trainable_coordinates=89485056,frozen_short_coordinates=24281088,all_USV_storage_unchanged=True,
        model_RUN_seconds=phase('run_finished')['unix']-phase('process_before_model')['unix'],
        TRAIN_interval_seconds=phase('training_after')['unix']-phase('training_before')['unix'],
        update_body_seconds=sum(e['seconds'] for e in events),checkpoint_metadata_seconds=sum(p['seconds'] for p in checkpoints),
        logical_checkpoint_bytes=sum(p['bytes'] for p in checkpoints),
        update_GPU_allocated_recorded_peak_bytes=max(e['gpu_peak_allocated_bytes'] for e in events),
        first_step_after_observation_seconds=sum(s['after_step_observation_seconds'] for s in actual),
        clipped_updates=sum(e['gradient_l2_preclip']>e['gradient_clip_threshold'] for e in events),
        memory_summary=c['memory_summary'],CPU_admission_seconds=load(REPORT/'CPU_GATE.json')['seconds'],
        CPU_verification_and_two_paired_intervals_seconds=v['CPU_verification_seconds'],
        v1_failed_attempt_file_mtime_interval_seconds=55.936747,
        v1_cost_scope='19 FIT gates 38 NLL + 38 short, first 16 FIT forward/backward; no optimizer, CP or DEV. mtime interval is not a completed RUN. Original resources/failure seal retained.',
        scope='RUN excludes early imports/checks; TRAIN includes first18 saves, both before/after weight observations, excludes final save and terminal frozen-base digest. Update/observation/save clocks overlap and must not be added. Logical bytes not physical IO or deployment. No fresh controlled speed/peak comparison.')
    unit=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',
        'llmcl-onevision-osft-short-factor-control-20261022-v2.service','-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],text=True).splitlines())
    assert unit==dict(MainPID='0',ExecMainStatus='0',ActiveState='inactive')
    assert load(REPORT/'queue/state.json')['phase']=='complete_verified'
    inputs=[REPORT/n for n in ('VERIFIED.json','COMPLETE.json','READY.json','CPU_GATE.json','V1_FAILURE_SEAL.json')]
    inputs += [SCREEN/'SCREEN_COMPLETE.json',SCREEN/'SPLIT.json',WORK/'OPTIMIZER_EVIDENCE.json',WORK/'SHORT_STAGE_EVIDENCE.json',
        WORK/'resources.jsonl',WORK/'logs/events.jsonl',*sorted((WORK/'checkpoints').glob('stage_??.json'))]
    out=dict(complete=True,quality=v['quality'],anchors={n:dict(domain_macro=p['domain_macro'],domains=p['domains']) for n,p in anchors.items()},
        paired=v['paired'],QA_total_counts=counts,all_100_QA_scalar_pairs=qa,primary_quality_pass=quality_pass,
        measured_gradient_and_moment_cost_pass=cost_pass,fixed_quality_threshold=threshold,
        action='retain_feasibility_for_separate_fresh_FIT_PAS_reference_necessity_review' if quality_pass and cost_pass else 'close_fixed_recipe',
        stage_actual_weight_evidence=actual,costs=costs,service_terminal=unit,automatic_new_model_or_full_confirmation=False,
        frozen_scientific_source_unchanged=True,source_sha256=sha(__file__),
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},prediction_sha256=prediction_inputs,
        new_model_or_GPU_or_QA_generation_or_bootstrap_or_score_requests=0,
        report_CPU_seconds=time.monotonic()-started)
    (REPORT/'OUTCOME.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(outcome_sha256=sha(REPORT/'OUTCOME.json'),primary_quality_pass=quality_pass,cost_pass=cost_pass,
        costs={k:val for k,val in costs.items() if k!='memory_summary'},stage_summary=actual,CPU_seconds=out['report_CPU_seconds'])))

if __name__=='__main__':
    main()
