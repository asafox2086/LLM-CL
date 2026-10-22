"""Frozen one-condition FIT/DEV short-factor comparison; never official TEST."""
import ast
from collections import Counter
import importlib.util
import json
import math
import random
import re
from pathlib import Path
from core import ROOT,PILOT,PORT,REPORT,WORK,study,sha,write

DOMAINS=('natural','medical','geography','finance','traffic')

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
    assert v['trainable_coordinates']==89485056 and v['short_side_frozen'] is True
    return v

def scorer():
    t=ast.parse((PORT/'runtime_utils.py').read_text())
    ns=dict(re=re,Counter=Counter)
    exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ('score','tokens')],type_ignores=[]),'<original-score>','exec'),ns)
    return ns['score']

def read_predictions(root,m,dev,expected_hashes=None):
    score=scorer();scores={};domains={d:[] for d in DOMAINS};files={}
    for t in m['tasks']:
        k=t['task']
        if not dev[k]:continue
        p=root/'evaluations/stage_19'/f'T{k:02d}_dev_predictions.jsonl';rel=p.relative_to(ROOT).as_posix()
        h=sha(p)
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

def paired(new,old,split):
    # Reuse only previously reviewed math helpers; do not run/rewrite the old six comparisons.
    p=ROOT/'analysis/osft_increment_paired_analysis_20261022.py'
    spec=importlib.util.spec_from_file_location('allocation_paired_math',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    groups=mod.known_groups(split);rng=random.Random(20261022)
    values={c:{d:[] for d in (*DOMAINS,'macro')} for c in old}
    for _ in range(10000):
        dd={c:{} for c in old}
        for d in DOMAINS:
            chosen=[rng.randrange(len(groups[d])) for _ in groups[d]]
            ids=[i for j in chosen for i in groups[d][j][1]]
            for c,o in old.items():dd[c][d]=sum(new[i]-o[i] for i in ids)/len(ids)
        for c in old:
            for d in DOMAINS:values[c][d].append(dd[c][d])
            values[c]['macro'].append(sum(dd[c].values())/5)
    result={}
    for c,o in old.items():
        ids={d:[i for _,ii in groups[d] for i in ii] for d in DOMAINS}
        result[c]=dict(domain_difference_pp={d:sum(new[i]-o[i] for i in ii)/len(ii) for d,ii in ids.items()},
            QA_comparison={d:mod.comparison([new[i]-o[i] for i in ii]) for d,ii in ids.items()},
            exploratory_group_interval_pp={d:[mod.percentile(v,.025),mod.percentile(v,.975)] for d,v in values[c].items()})
        result[c]['macro_difference_pp']=sum(result[c]['domain_difference_pp'].values())/5
    return dict(primary='short_fixed_minus_OSFT_all',secondary='short_fixed_minus_LoRA_qv',comparisons=result,
        replicates=10000,seed=20261022,not_equivalence_or_significance_or_training_seed_uncertainty=True,
        reused_DEV_not_new_holdout=True,automatic_full_confirmation=False)

def verify():
    ready=check_ready(False);complete=json.loads((REPORT/'COMPLETE.json').read_text())
    assert complete['complete'] and complete['ready_sha256']==sha(REPORT/'READY.json')
    assert complete['TEST_QA']==0
    assert complete['source_sha256']==sha(Path(__file__).with_name('run.py'))
    _,m,_,split,_,dev,_,plan=data()
    quality,scores=read_predictions(WORK,m,dev,complete['quality']['prediction_sha256'])
    assert quality==complete['quality']
    assert sha(WORK/complete['checkpoint']['path'])==complete['checkpoint']['sha256']
    assert complete['trainable_digest_before_DEV']==complete['trainable_digest_after_DEV']
    assert complete['config']['trainable_coordinates']==89485056 and complete['config']['target_modules']==168
    assert complete['config']==json.loads((WORK/'config.json').read_text())
    assert sha(WORK/'config.json')==complete['config_sha256']
    cfg=complete['config']
    assert cfg['learning_rate']==1e-5 and cfg['unfreeze_rank_ratio']==.25
    assert cfg['short_side_frozen'] is True and cfg['host_optimizer'] and cfg['offload_activations']
    assert (cfg['FIT_QA'],cfg['DEV_QA'],cfg['TEST_QA'],cfg['updates'])==(1213,100,0,87)
    assert cfg['ready_sha256']==sha(REPORT/'READY.json') and cfg['source_sha256']==complete['source_sha256']
    assert cfg['manifest_sha256']==ready['manifest_sha256'] and cfg['split_sha256']==ready['split_sha256']
    assert complete['frozen_base_digest_before']==complete['frozen_base_digest_after']
    train=json.loads((WORK/'TRAIN_COMPLETE.json').read_text())
    assert train['complete'] and train['short_stage_records']==complete['short_stage_records']
    gate=json.loads((WORK/'NATIVE_GATE.json').read_text())
    assert sha(WORK/'NATIVE_GATE.json')==complete['native_gate_sha256']
    assert gate['complete'] and len(gate['gates'])==19 and gate['NLL_paths']==gate['short_generations']==38 and gate['TEST_QA']==0
    events=[json.loads(s) for s in (WORK/'logs/events.jsonl').read_text().splitlines()]
    assert len(events)==87 and sha(WORK/'logs/events.jsonl')==complete['events_sha256']
    rawfiles={}
    for p in plan:
        k=p['stage'];v=json.loads((WORK/'validity'/f'stage_{k:02d}.json').read_text())
        assert v['passed'] and v['actual_parameters_changed'] and v['updates']==v['positive_lr_updates']==p['updates']
        ev=[e for e in events if e['stage']==k]
        assert [e['step'] for e in ev]==list(range(1,p['updates']+1))
        assert ev[-1]['qa_seen']==ev[-1]['total_qa']==p['QA']
        for e in ev:
            assert all(math.isfinite(e[x]) for x in ['loss','gradient_l2_preclip','gradient_l2_postclip'])
            assert all(math.isfinite(lr) and lr>0 for lr in e['learning_rate_used'])
            path=WORK/'gradients'/f"S{k:02d}_{e['step']:03d}.json"
            raw_record=json.loads(path.read_text());raw=raw_record['raw']
            assert len(raw)==336 and all(g['finite'] and math.isfinite(g['norm']) for g in raw.values())
            assert math.isfinite(raw_record['preclip_total_norm']) and raw_record['preclip_total_norm']>=0
            assert 0<raw_record['clip_coefficient']<=1
            rawfiles[path.relative_to(ROOT).as_posix()]=sha(path)
        path=WORK/'gradients'/f'S{k:02d}_first_raw.pt';rawfiles[path.relative_to(ROOT).as_posix()]=sha(path)
    assert rawfiles==complete['gradient_sha256'] and len(rawfiles)==106
    refs=json.loads((study.REPORT/'SCREEN_COMPLETE.json').read_text())['all_four_results'];old={}
    for c in ['OSFT_all','LoRA_qv']:
        v=refs[c];q,s=read_predictions(ROOT/v['checkpoint_root'],m,dev,v['prediction_sha256'])
        assert q['domains']==v['domains'] and q['domain_macro']==v['domain_macro'];old[c]=s
    records=complete['short_stage_records'];assert len(records)==19
    assert sha(WORK/'SHORT_STAGE_EVIDENCE.json')==complete['short_stage_evidence_sha256']
    assert json.loads((WORK/'SHORT_STAGE_EVIDENCE.json').read_text())==records
    assert sha(WORK/'OPTIMIZER_EVIDENCE.json')==complete['optimizer_evidence_sha256']
    optimizer=json.loads((WORK/'OPTIMIZER_EVIDENCE.json').read_text());assert len(optimizer)==87
    for p,r in zip(plan,records):
        assert r['stage']==p['stage'] and r['updates']==p['updates']
        assert r['short_frozen_exact'] and r['short_before']==r['short_after']
        w=r['first_step'];assert w['stage']==p['stage'] and len(w['matrices'])==168
        assert all(math.isfinite(w[x]) and w[x]>=0 for x in ['squared_change','squared_before'])
        assert all(math.isfinite(t[x]) and t[x]>=0 for t in w['matrices'] for x in ['squared_change','squared_before'])
        oo=[o for o in optimizer if o['stage']==p['stage']]
        assert [o['step'] for o in oo]==list(range(1,p['updates']+1))
        for o in oo:
            assert o['parameter_tensors']==336 and o['gradient_bytes']==4*89485056 and o['moment_bytes']==8*89485056
            assert o['all_moments_CPU'] and o['short_gradient_None'] and o['optimizer_parameter_IDs_match']
    layout=json.loads((WORK/'LAYOUT.json').read_text())
    assert layout['trainable_coordinates']==89485056 and layout['frozen_short_coordinates']==24281088
    assert layout['S_low_coordinates']==28416 and layout['target_modules']==168
    assert len({r['name'] for r in layout['modules']})==168
    for r in layout['modules']:
        m,n=r['shape'];rank=min(m,n)-int(min(m,n)*.75)
        assert r['frozen']==('V_low' if m>=n else 'U_low')
        assert r['trainable']==rank*(max(m,n)+1) and r['frozen_short']==rank*min(m,n)
    assert sha(WORK/'LAYOUT.json')==complete['layout_sha256']
    comparisons=paired(scores,old,split)
    primary=quality['domain_macro']>max(refs[c]['domain_macro'] for c in ['OSFT_all','LoRA_qv'])
    cost=all(o['gradient_bytes']<455064576 and o['moment_bytes']<910129152 for o in optimizer)
    return dict(complete=True,quality=quality,paired=comparisons,FIT_QA=1213,DEV_QA=100,updates=87,TEST_QA=0,
        complete_sha256=sha(REPORT/'COMPLETE.json'),ready_sha256=sha(REPORT/'READY.json'),
        source_sha256=sha(__file__),checkpoint=complete['checkpoint'],new_model_requests=0,
        mechanism_only_no_full_candidates=True,original_screen_and_full_confirmation_unchanged=True,
        primary_quality_pass=primary,measured_gradient_and_moment_cost_pass=cost,
        possible_fresh_FIT_PAS_reference_followup=primary and cost,automatic_full_confirmation=False)
