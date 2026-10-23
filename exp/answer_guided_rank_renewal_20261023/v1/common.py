"""Verify only the two new flows; old quality anchors are not recomputed."""
import ast,importlib.util,json,math,random,re,hashlib
from collections import Counter
from pathlib import Path
from core import ROOT,PORT,REPORT,WORK,study,sha,write,FINAL,TRAINABLE,RANK,EXTRA,CASES
DOMAINS=('natural','medical','geography','finance','traffic')

def data():
    path,m,original=study.original_train()
    split=json.loads((study.REPORT/'SPLIT.json').read_text())
    assert (split['FIT_QA'],split['DEV_QA'])==(1213,100)
    fit={};dev={};mapped={};plan=[];groups={}
    for gid,g in split['groups'].items():
        for i in g['ids']:groups[i]=gid
    for t in m['tasks']:
        k=t['task'];fs=set(split['fit_ids'][str(k)]);ds=set(split['dev_ids'][str(k)])
        assert not fs&ds
        fit[k]=[r for r in original[k] if r['id'] in fs];dev[k]=[r for r in original[k] if r['id'] in ds]
        assert [r['id'] for r in fit[k]]==split['fit_ids'][str(k)]
        assert [r['id'] for r in dev[k]]==split['dev_ids'][str(k)]
        mapped[k,'train']=fit[k];mapped[k,'dev']=dev[k]
        order=sorted(fit[k],key=lambda r:hashlib.sha256(f"42:{k}:{r['id']}".encode()).digest())
        t['training_order_sha256']=hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
        t['files']={'train':{'count':len(fit[k])},'dev':{'count':len(dev[k])}}
        plan.append(dict(stage=k,QA=len(fit[k]),updates=math.ceil(len(fit[k])/16)))
    assert sum(p['updates'] for p in plan)==87
    return path,m,split,fit,dev,mapped,plan,groups

def check_ready(base=True):
    v=json.loads((REPORT/'READY.json').read_text())
    for key in ('sources','evidence'):
        for n,h in v[key].items():assert sha(ROOT/n)==h,n
    if base:
        for n,h in v['base_files'].items():assert sha(ROOT/n)==h,n
    assert v['training_ast_sha256']==study.training_ast()[1]
    assert v['rank']==RANK and v['temporary_rank']==EXTRA and v['cases']==list(CASES)
    assert v['worker_budget_seconds']==7200 and v['aggregate_model_budget_seconds']==14400
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
    groups=mod.known_groups(split);rng=random.Random(20261023);values={d:[] for d in (*DOMAINS,'macro')}
    for _ in range(10000):
        dd={}
        for d in DOMAINS:
            chosen=[rng.randrange(len(groups[d])) for _ in groups[d]];ids=[i for j in chosen for i in groups[d][j][1]]
            dd[d]=sum(candidate[i]-reference[i] for i in ids)/len(ids);values[d].append(dd[d])
        values['macro'].append(sum(dd.values())/5)
    ids_by_domain={d:[i for _,ii in groups[d] for i in ii] for d in DOMAINS}
    difference={d:sum(candidate[i]-reference[i] for i in ii)/len(ii) for d,ii in ids_by_domain.items()}
    import hashlib
    return dict(primary='Answer_minus_Spectral',domain_difference_pp=difference,
        macro_difference_pp=sum(difference.values())/5,
        QA_comparison={d:mod.comparison([candidate[i]-reference[i] for i in ii]) for d,ii in ids_by_domain.items()},
        all_QA_differences=[dict(id_sha256=hashlib.sha256(i.encode()).hexdigest(),domain=d,factor=candidate[i],anchor=reference[i],difference_pp=candidate[i]-reference[i]) for d,ii in ids_by_domain.items() for i in ii],
        exploratory_group_interval_pp={d:[mod.percentile(v,.025),mod.percentile(v,.975)] for d,v in values.items()},
        replicates=10000,seed=20261023,reused_DEV_not_new_holdout=True,
        not_equivalence_significance_or_training_seed_uncertainty=True,old_two_comparisons_not_repeated=True)

def verify_case(case,ready,m,dev,plan):
    root=WORK/case;cp=REPORT/'cases'/case/'COMPLETE.json';c=json.loads(cp.read_text())
    assert c['complete'] and c['case']==case and c['TEST_QA']==0
    assert math.isfinite(c['model_RUN_seconds']) and 0<c['model_RUN_seconds']<=7200
    started=json.loads((REPORT/'cases'/case/'STARTED.json').read_text());assert c['worker_pid']==started['pid']
    assert c['ready_sha256']==sha(REPORT/'READY.json')
    q,scores=read_predictions(root,m,dev,c['quality']['prediction_sha256']);assert q==c['quality']
    cfg=c['config'];assert cfg==json.loads((root/'config.json').read_text())
    assert (cfg['FIT_QA'],cfg['DEV_QA'],cfg['updates'],cfg['TEST_QA'])==(1213,100,87,0)
    assert (cfg['rank'],cfg['temporary_rank'],cfg['learning_rate'],cfg['scale'],cfg['dropout'])==(200,16,1e-4,4,.1)
    assert cfg['final_coefficients']==FINAL and cfg['trainable_coordinates']==TRAINABLE
    assert cfg['worker_budget_seconds']==7200 and cfg['aggregate_model_budget_seconds']==14400
    assert cfg['source_sha256']==sha(Path(__file__).with_name('run.py'))==c['source_sha256']
    assert cfg['manifest_sha256']==ready['manifest_sha256'] and cfg['split_sha256']==ready['split_sha256']
    assert sha(root/c['checkpoint']['path'])==c['checkpoint']['sha256']
    assert c['full_checkpoint_reload'] and c['all_weight_digest_before_DEV']==c['all_weight_digest_after_DEV']
    assert c['stable_digest_before_TRAIN']==c['stable_digest_after_TRAIN']
    lay=c['final_layout'];assert lay['coefficients']==FINAL and lay['trainable']==0
    assert not any(n.endswith(('.gate_proj.weight','.up_proj.weight','.down_proj.weight','.A','.B','.gate')) for n in lay['parameters'])
    factors=[v for n,v in lay['parameters'].items() if n.endswith(('.left','.right'))]
    assert len(factors)==144 and all(not f['trainable'] and f['dtype']=='torch.float16' and 200 in f['shape'] for f in factors)
    assert sha(root/'INITIAL_BINDING.json')==c['initial_binding_sha256']
    assert sha(root/'INITIAL_LAYOUT.json')==c['initial_layout_sha256']
    initial=json.loads((root/'INITIAL_BINDING.json').read_text());assert initial['initial_sha256']==sha(WORK/'INITIAL.json') and initial['untrained']
    for n,h in c['file_sha256'].items():assert sha(ROOT/n)==h,n
    gate=json.loads((root/'NATIVE_GATE.json').read_text())
    assert sha(root/'NATIVE_GATE.json')==c['native_gate_sha256']
    assert gate['complete'] and len(gate['gates'])==19 and gate['NLL_paths']==gate['short_generations']==38
    assert json.loads((root/'TRAIN_COMPLETE.json').read_text())['updates']==87
    events=[json.loads(s) for s in (root/'logs/events.jsonl').read_text().splitlines()]
    assert len(events)==87 and sha(root/'logs/events.jsonl')==c['events_sha256']
    consolidations=[];keys=set()
    for row in plan:
        k=row['stage'];v=json.loads((root/'validity'/f'stage_{k:02d}.json').read_text())
        assert v['passed'] and v['actual_parameters_changed'] and v['positive_lr_updates']==v['updates']==row['updates']
        es=[e for e in events if e['stage']==k]
        assert [e['step'] for e in es]==list(range(1,row['updates']+1)) and es[-1]['qa_seen']==es[-1]['total_qa']==row['QA']
        for e in es:
            assert all(math.isfinite(e[x]) for x in ('loss','gradient_l2_preclip','gradient_l2_postclip'))
            assert all(math.isfinite(lr) and lr>0 for lr in e['learning_rate_used'])
            p=root/'gradients'/f"S{k:02d}_{e['step']:03d}.json";g=json.loads(p.read_text())['raw']
            assert len(g)==144 and all(z['finite'] and math.isfinite(z['norm']) for z in g.values())
            assert all(n.endswith(('.A','.B')) for n in g)
            o=json.loads((root/'optimizer'/f"S{k:02d}_{e['step']:03d}.json").read_text())
            assert o['parameter_IDs_match'] and o['coordinates']==TRAINABLE and o['gradient_bytes']==TRAINABLE*4 and o['two_moment_bytes']==TRAINABLE*8
            assert o['moment_devices']==['cpu']
        z=json.loads((root/'consolidation'/f'S{k:02d}.json').read_text());consolidations.append(z)
        assert z['case']==case and z['stage']==k and z['count_before']==FINAL+TRAINABLE and z['count_after']==FINAL and z['final_trainable']==0
        assert z['stage_frozen_digest_before_TRAIN']==z['stage_frozen_digest_after_TRAIN']
        assert z['current_QA']<=2 and z['old_QA']<=8 and z['gradient_QA']<=10 and z['candidate_forward_QA']<=20
        assert z['current_payload_bytes']<=16*1024**2 and z['old_payload_bytes']<=32*1024**2
        assert z['bank_after_QA']<=8 and z['bank_after_bytes']<=32*1024**2
        assert len(z['spectral_masks'])==72 and all(a==list(range(200)) for a in z['spectral_masks'].values())
        if case=='Spectral':assert z['selected']=='spectral' and z['gradient_QA']==z['candidate_forward_QA']==0 and z['answer_masks'] is None
        else:
            if z['gradient_QA']:
                assert len(z['answer_masks'])==72 and len(z['gate_stats'])==72
                assert all(len(a)==200 and len(set(a))==200 and min(a)>=0 and max(a)<216 for a in z['answer_masks'].values())
                for name,stats in z['gate_stats'].items():
                    costs=stats['deletion_costs'];assert len(costs)==216 and all(math.isfinite(x) for x in costs)
                    assert stats['signed_score_sha256']==hashlib.sha256(json.dumps(costs).encode()).hexdigest()
                    assert z['answer_masks'][name]==sorted(sorted(range(216),key=lambda i:(-costs[i],i))[:200])
                assert z['mask_different']==any(z['answer_masks'][n]!=z['spectral_masks'][n] for n in z['spectral_masks'])
            if z['mask_different']:
                assert z['candidate_forward_QA']==2*z['gradient_QA']
                assert all(math.isfinite(z[a]) for a in ('parent_gate_NLL','spectral_NLL','answer_NLL'))
                assert z['selected']==('answer' if z['answer_NLL']<z['spectral_NLL'] else 'spectral')
            else:assert z['selected']=='spectral' and z['candidate_forward_QA']==0
        saved=json.loads((root/'layouts'/f'S{k:02d}.json').read_text());assert saved['coefficients']==FINAL and saved['trainable']==0
        assert all(z[key]['actual_CPU_tensor_storage_union_bytes']<=176*1024**2 for key in ('memory_before','memory_after'))
        meta=json.loads((root/'checkpoints'/f'stage_{k:02d}.json').read_text());assert meta['stage']==k and meta['full_architecture_and_runtime']
    assert len(list((root/'gradients').iterdir()))==106 and len(list((root/'optimizer').iterdir()))==87
    expected={p.relative_to(ROOT).as_posix() for folder in ('gradients','optimizer','consolidation','layouts','validity') for p in (root/folder).iterdir()}
    expected.update(p.relative_to(ROOT).as_posix() for p in [root/'resources.jsonl',root/'memory_summary.json',*sorted((root/'checkpoints').glob('*.json'))])
    assert expected==set(c['file_sha256'])
    resources=[json.loads(s) for s in (root/'resources.jsonl').read_text().splitlines()]
    assert resources and all(z['pid']==c['worker_pid'] and z['visible_gpu']=='1' for z in resources)
    assert any(z['phase']=='run_finished' for z in resources)
    assert json.loads((root/'memory_summary.json').read_text())==c['memory_summary']
    return dict(quality=q,initial_binding=initial,complete_sha256=sha(cp),
        gradient_QA=sum(z['gradient_QA'] for z in consolidations),candidate_forward_QA=sum(z['candidate_forward_QA'] for z in consolidations),
        different_masks=sum(z['mask_different'] for z in consolidations),answer_selected=sum(z['selected']=='answer' for z in consolidations),
        fallback_counts=Counter(z['fallback'] for z in consolidations),model_RUN_seconds=c['model_RUN_seconds']),scores

def verify():
    assert not (REPORT/'VERIFIED.json').exists(),'Only one new verification and new pair'
    ready=check_ready(False);_,m,split,_,dev,_,plan,_=data();proofs={};scores={}
    for case in CASES:proofs[case],scores[case]=verify_case(case,ready,m,dev,plan)
    assert proofs['Spectral']['initial_binding']==proofs['Answer']['initial_binding']
    assert sum(v['model_RUN_seconds'] for v in proofs.values())<=14400
    assert proofs['Answer']['gradient_QA']<=190 and proofs['Answer']['candidate_forward_QA']<=380
    pair=paired(scores['Answer'],scores['Spectral'],split)
    eligible=(FINAL<=.75*893675552 and proofs['Answer']['quality']['domain_macro']>max(ready['reference_best_macro'],proofs['Spectral']['quality']['domain_macro']))
    return dict(complete=True,cases=proofs,new_pair=pair,possible_followup_only=eligible,
        old_best_legal_FIT_only_macro=ready['reference_best_macro'],final_coefficients=FINAL,base_coefficients=893675552,
        source_sha256=sha(__file__),ready_sha256=sha(REPORT/'READY.json'),TEST_QA=0,
        automatic_full_confirmation=False,not_new_holdout_or_novelty_or_stable_gain=True,old_F1_and_pairs_not_repeated=True)

if __name__=='__main__':
    import time
    tick=time.monotonic();v=verify();v['CPU_verification_seconds']=time.monotonic()-tick
    write(REPORT/'VERIFIED.json',v)
    print(json.dumps(dict(verified=True,verified_sha256=sha(REPORT/'VERIFIED.json'))))
