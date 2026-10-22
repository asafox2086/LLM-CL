"""Frozen FIT data and one independent ShortFixed-minus-fresh-PAS comparison."""
import ast
from collections import Counter
import importlib.util
import json
import math
import random
import re
from pathlib import Path
from core import ROOT,PORT,REPORT,WORK,study,sha

DOMAINS=('natural','medical','geography','finance','traffic')
CANDIDATE=ROOT/'summary/edge_deployment_20261019/osft_short_factor_control_20261022/v2'
CANDIDATE_WORK=ROOT/'exp/osft_short_factor_control_20261022/v2/private'

def data():
    path,m,original=study.original_train();split,fit,dev=study.split_train(m,original)
    assert split==json.loads((study.REPORT/'SPLIT.json').read_text())
    assert (split['FIT_QA'],split['DEV_QA'])==(1213,100)
    plan=[];mapped={}
    import hashlib
    for t in m['tasks']:
        k=t['task'];mapped[k,'train']=fit[k];mapped[k,'dev']=dev[k]
        order=sorted(fit[k],key=lambda r:hashlib.sha256(f"42:{k}:{r['id']}".encode()).digest())
        t['training_order_sha256']=hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
        t['files']={'train':{'count':len(fit[k])},'dev':{'count':len(dev[k])}}
        plan.append(dict(stage=k,QA=len(fit[k]),updates=math.ceil(len(fit[k])/16)))
    assert sum(p['updates'] for p in plan)==87
    return path,m,original,split,fit,dev,mapped,plan

def check_ready(base=True):
    v=json.loads((REPORT/'READY.json').read_text())
    for n,h in v['sources'].items():assert sha(ROOT/n)==h,n
    for n,h in v['evidence'].items():assert sha(ROOT/n)==h,n
    if base:
        for n,h in v['base_files'].items():assert sha(ROOT/n)==h,n
    assert v['training_ast_sha256']==study.training_ast()[1]
    assert v['worker_budget_seconds']==14400 and v['conservative_cumulative_budget_seconds']==21600
    assert (v['FIT_QA'],v['DEV_QA'],v['updates'],v['TEST_QA'])==(1213,100,87,0)
    return v

def scorer():
    t=ast.parse((PORT/'runtime_utils.py').read_text());ns=dict(re=re,Counter=Counter)
    exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ('score','tokens')],type_ignores=[]),'<original-score>','exec'),ns)
    return ns['score']

def read_predictions(root,m,dev,expected_hashes=None):
    score=scorer();scores={};domains={d:[] for d in DOMAINS};files={}
    for t in m['tasks']:
        k=t['task']
        if not dev[k]:continue
        p=root/'evaluations/stage_19'/f'T{k:02d}_dev_predictions.jsonl';rel=p.relative_to(ROOT).as_posix();h=sha(p)
        if expected_hashes is not None:assert expected_hashes[rel]==h
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        assert [r['id'] for r in rows]==[r['id'] for r in dev[k]]
        for r,g in zip(rows,dev[k]):
            assert r['references']==g['answers'] and r['generation_cap']==t['max_new_tokens']
            assert r['scores']==score(r['prediction'],g['answers'])
            f=r['scores']['token_f1'];assert math.isfinite(f) and 0<=f<=100
            scores[r['id']]=f;domains[t['domain']].append(f)
        files[rel]=h
    assert len(scores)==100 and all(len(v)==20 for v in domains.values())
    stats={d:dict(F1=sum(v)/len(v),QA=len(v)) for d,v in domains.items()}
    return dict(domains=stats,domain_macro=sum(v['F1'] for v in stats.values())/5,prediction_sha256=files),scores

def paired(candidate,reference,split):
    p=ROOT/'analysis/osft_increment_paired_analysis_20261022.py'
    spec=importlib.util.spec_from_file_location('fresh_pas_paired_math',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    groups=mod.known_groups(split);rng=random.Random(20261022);values={d:[] for d in (*DOMAINS,'macro')}
    for _ in range(10000):
        dd={}
        for d in DOMAINS:
            chosen=[rng.randrange(len(groups[d])) for _ in groups[d]];ids=[i for j in chosen for i in groups[d][j][1]]
            dd[d]=sum(candidate[i]-reference[i] for i in ids)/len(ids);values[d].append(dd[d])
        values['macro'].append(sum(dd.values())/5)
    ids_by_domain={d:[i for _,ii in groups[d] for i in ii] for d in DOMAINS}
    difference={d:sum(candidate[i]-reference[i] for i in ii)/len(ii) for d,ii in ids_by_domain.items()}
    import hashlib
    return dict(primary='ShortFixed_minus_fresh_FIT_PASs',domain_difference_pp=difference,
        macro_difference_pp=sum(difference.values())/5,
        QA_comparison={d:mod.comparison([candidate[i]-reference[i] for i in ii]) for d,ii in ids_by_domain.items()},
        all_QA_differences=[dict(id_sha256=hashlib.sha256(i.encode()).hexdigest(),domain=d,ShortFixed=candidate[i],fresh_PAS=reference[i],difference_pp=candidate[i]-reference[i]) for d,ii in ids_by_domain.items() for i in ii],
        exploratory_group_interval_pp={d:[mod.percentile(v,.025),mod.percentile(v,.975)] for d,v in values.items()},
        replicates=10000,seed=20261022,reused_DEV_not_new_holdout=True,
        not_equivalence_significance_or_training_seed_uncertainty=True,old_two_comparisons_not_repeated=True)

def candidate_scores(m,dev,proof):
    """Copy old sealed scores for this new pair; never rescore the completed candidate."""
    values={}
    for task in m['tasks']:
        k=task['task']
        if not dev[k]:continue
        path=CANDIDATE_WORK/'evaluations/stage_19'/f'T{k:02d}_dev_predictions.jsonl'
        assert sha(path)==proof['quality']['prediction_sha256'][path.relative_to(ROOT).as_posix()]
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert [r['id'] for r in rows]==[g['id'] for g in dev[k]]
        for row,gold in zip(rows,dev[k]):
            assert row['references']==gold['answers']
            f=row['scores']['token_f1'];assert math.isfinite(f) and 0<=f<=100
            assert row['id'] not in values;values[row['id']]=f
    assert len(values)==100
    return values

def verify():
    ready=check_ready(False);c=json.loads((REPORT/'COMPLETE.json').read_text())
    assert c['complete'] and c['TEST_QA']==0 and c['ready_sha256']==sha(REPORT/'READY.json')
    assert c['source_sha256']==sha(Path(__file__).with_name('run.py'))
    _,m,_,split,_,dev,_,plan=data()
    quality,reference=read_predictions(WORK,m,dev,c['quality']['prediction_sha256']);assert quality==c['quality']
    assert sha(WORK/c['checkpoint']['path'])==c['checkpoint']['sha256']
    assert c['trainable_digest_before_DEV']==c['trainable_digest_after_DEV']
    assert c['learner_digest_before_DEV']==c['learner_digest_after_DEV']
    assert c['frozen_base_digest_before']==c['frozen_base_digest_after']
    cfg=c['config'];assert cfg==json.loads((WORK/'config.json').read_text()) and sha(WORK/'config.json')==c['config_sha256']
    assert cfg['method']=='PASs_MoE' and cfg['learning_rate']==1e-4 and cfg['fresh_base']
    assert (cfg['FIT_QA'],cfg['DEV_QA'],cfg['updates'],cfg['TEST_QA'])==(1213,100,87,0)
    assert cfg['ready_sha256']==sha(REPORT/'READY.json') and cfg['source_sha256']==c['source_sha256']
    assert cfg['manifest_sha256']==ready['manifest_sha256'] and cfg['split_sha256']==ready['split_sha256']
    assert cfg['target_modules']==168 and cfg['host_optimizer'] and not cfg['save_checkpoints_for_author']
    assert (cfg['rank'],cfg['alpha'],cfg['dropout'],cfg['experts'],cfg['gate_rank'])==(128,256,.05,2,16)
    gate=json.loads((WORK/'NATIVE_GATE.json').read_text())
    assert sha(WORK/'NATIVE_GATE.json')==c['native_gate_sha256']
    assert gate['complete'] and len(gate['gates'])==19 and gate['NLL_paths']==gate['short_generations']==38 and gate['usage_unpolluted']
    train=json.loads((WORK/'TRAIN_COMPLETE.json').read_text());assert train['complete'] and train['updates']==87 and train['TEST_QA']==0
    assert sha(WORK/'CALLBACK_EVIDENCE.json')==c['callback_evidence_sha256']
    callbacks=json.loads((WORK/'CALLBACK_EVIDENCE.json').read_text());assert callbacks==c['callback_records'] and len(callbacks)==19
    assert sha(WORK/'OPTIMIZER_EVIDENCE.json')==c['optimizer_evidence_sha256']
    optimizer=json.loads((WORK/'OPTIMIZER_EVIDENCE.json').read_text());assert len(optimizer)==c['optimizer_witness_commits']==87
    assert cfg['optimizer_witness_durable_every_step'] and cfg['worker_budget_seconds']==14400 and cfg['conservative_cumulative_budget_seconds']==21600
    layout=json.loads((WORK/'LAYOUT.json').read_text());assert sha(WORK/'LAYOUT.json')==c['layout_sha256']
    assert layout['target_modules']==168 and layout['trainable_coordinates']==cfg['trainable_coordinates']
    events=[json.loads(s) for s in (WORK/'logs/events.jsonl').read_text().splitlines()]
    assert len(events)==87 and sha(WORK/'logs/events.jsonl')==c['events_sha256']
    rawfiles={}
    for p,r in zip(plan,callbacks):
        k=p['stage'];assert r['stage']==k
        assert r['recording_true']==r['recording_false']==r['post_backward']==p['QA'] and r['after_update']==p['updates']
        assert r['prior_importance_modules']==(0 if k==1 else 168)
        assert r['recorded_usage_callbacks']>0 and r['current_importance_modules']==168
        assert r['aggregated_modules']==dict(_I_agg=168,_A_ref=168,_B_ref=168) and r['reference_tensors_CPU']
        assert len(r['tau_at_end'])==168 and all(t==.7 for t in r['tau_at_end'])
        validity=json.loads((WORK/'validity'/f'stage_{k:02d}.json').read_text())
        assert validity['passed'] and validity['actual_parameters_changed'] and validity['updates']==validity['positive_lr_updates']==p['updates']
        ev=[e for e in events if e['stage']==k]
        assert [e['step'] for e in ev]==list(range(1,p['updates']+1)) and ev[-1]['qa_seen']==ev[-1]['total_qa']==p['QA']
        oo=[o for o in optimizer if o['stage']==k];assert [o['step'] for o in oo]==list(range(1,p['updates']+1))
        for o in oo:
            assert o['optimizer_parameter_IDs_match'] and o['all_moments_CPU'] and o['trainable_coordinates']==cfg['trainable_coordinates']
            assert o['parameter_tensors']==cfg['parameter_tensors'] and o['gradient_bytes']>0 and o['two_moment_bytes']>0
        for e in ev:
            assert all(math.isfinite(e[x]) for x in ['loss','gradient_l2_preclip','gradient_l2_postclip'])
            assert all(math.isfinite(lr) and lr>0 for lr in e['learning_rate_used'])
            pth=WORK/'gradients'/f"S{k:02d}_{e['step']:03d}.json";raw=json.loads(pth.read_text())
            assert raw['raw'] and set(raw['raw'])<=set(layout['parameters'])
            assert all(t['finite'] and math.isfinite(t['norm']) for t in raw['raw'].values())
            assert math.isfinite(raw['preclip_total_norm']) and 0<raw['clip_coefficient']<=1
            rawfiles[pth.relative_to(ROOT).as_posix()]=sha(pth)
        pth=WORK/'gradients'/f'S{k:02d}_first_raw.pt';rawfiles[pth.relative_to(ROOT).as_posix()]=sha(pth)
    assert len(rawfiles)==106 and rawfiles==c['gradient_sha256']
    assert len(c['importance_snapshot_sha256'])==19
    for n,h in c['importance_snapshot_sha256'].items():assert sha(ROOT/n)==h
    old=json.loads((CANDIDATE/'VERIFIED.json').read_text());assert old['complete'] and old['primary_quality_pass'] and old['measured_gradient_and_moment_cost_pass']
    # Only this new pair consumes the candidate scores again; old paired analyses are not called.
    candidate_quality=old['quality'];candidate=candidate_scores(m,dev,old)
    assert candidate_quality['domain_macro']==42.30132909166463
    comparison=paired(candidate,reference,split)
    threshold=max(quality['domain_macro'],40.78380056394307,39.870438970139254)
    candidate_pass=candidate_quality['domain_macro']>threshold
    return dict(complete=True,quality=quality,paired=comparison,ShortFixed_quality=candidate_quality,
        candidate_exceeds_all_fixed_quality_anchors=candidate_pass,
        action='consider_separate_full_confirmation_necessity_review' if candidate_pass else 'close_current_ShortFixed_candidate',
        fixed_quality_threshold=threshold,candidate_actual_state_byte_gate_previously_passed=True,
        FIT_QA=1213,DEV_QA=100,updates=87,TEST_QA=0,checkpoint=c['checkpoint'],
        ready_sha256=sha(REPORT/'READY.json'),complete_sha256=sha(REPORT/'COMPLETE.json'),source_sha256=sha(__file__),
        candidate_VERIFIED_sha256=sha(CANDIDATE/'VERIFIED.json'),new_model_requests=0,automatic_full_confirmation=False)
