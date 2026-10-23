"""Only new factor-pilot quality is scored; old anchors copy bound sealed scores."""
import ast,importlib.util,json,math,random,re,hashlib
from collections import Counter
from pathlib import Path
from core import ROOT,PORT,REPORT,WORK,study,sha,write,FINAL,TRAINABLE
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
    return dict(primary='physical_factor_minus_anchor',domain_difference_pp=difference,
        macro_difference_pp=sum(difference.values())/5,
        QA_comparison={d:mod.comparison([candidate[i]-reference[i] for i in ii]) for d,ii in ids_by_domain.items()},
        all_QA_differences=[dict(id_sha256=hashlib.sha256(i.encode()).hexdigest(),domain=d,factor=candidate[i],anchor=reference[i],difference_pp=candidate[i]-reference[i]) for d,ii in ids_by_domain.items() for i in ii],
        exploratory_group_interval_pp={d:[mod.percentile(v,.025),mod.percentile(v,.975)] for d,v in values.items()},
        replicates=10000,seed=20261022,reused_DEV_not_new_holdout=True,
        not_equivalence_significance_or_training_seed_uncertainty=True,old_two_comparisons_not_repeated=True)

def check_ready(base=True):
    v=json.loads((REPORT/'READY.json').read_text())
    for k in ['sources','evidence']:
        for n,h in v[k].items():assert sha(ROOT/n)==h,n
    if base:
        for n,h in v['base_files'].items():assert sha(ROOT/n)==h,n
    assert v['training_ast_sha256']==study.training_ast()[1]
    assert v['worker_budget_seconds']==7200 and v['rank']==540
    return v

def anchor_scores(m,dev,proof):
    values={};root=ROOT/proof['checkpoint_root']
    for task in m['tasks']:
        k=task['task']
        if not dev[k]:continue
        path=root/'evaluations/stage_19'/f'T{k:02d}_dev_predictions.jsonl'
        assert sha(path)==proof['prediction_sha256'][path.relative_to(ROOT).as_posix()]
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        assert [r['id'] for r in rows]==[r['id'] for r in dev[k]]
        for row,gold in zip(rows,dev[k]):
            assert row['references']==gold['answers']
            f=row['scores']['token_f1'];assert math.isfinite(f) and 0<=f<=100
            assert row['id'] not in values;values[row['id']]=f
    assert len(values)==100
    return values

def verify():
    assert not (REPORT/'VERIFIED.json').exists(),'Unique new verification only'
    ready=check_ready(False);c=json.loads((REPORT/'COMPLETE.json').read_text())
    assert c['complete'] and c['ready_sha256']==sha(REPORT/'READY.json') and c['TEST_QA']==0
    _,m,_,split,_,dev,_,plan=data()
    quality,scores=read_predictions(WORK,m,dev,c['quality']['prediction_sha256']);assert quality==c['quality']
    assert sha(WORK/c['checkpoint']['path'])==c['checkpoint']['sha256']
    assert c['trainable_digest_before_DEV']==c['trainable_digest_after_DEV']
    assert c['all_weight_digest_before_DEV']==c['all_weight_digest_after_DEV']==c['full_checkpoint_reload']['after_digest']
    assert c['frozen_digest_before_TRAIN']==c['frozen_digest_after_TRAIN']
    cfg=c['config'];assert cfg==json.loads((WORK/'config.json').read_text())
    assert sha(WORK/'config.json')==c['config_sha256']
    assert (cfg['FIT_QA'],cfg['DEV_QA'],cfg['updates'],cfg['TEST_QA'])==(1213,100,87,0)
    assert cfg['rank']==540 and cfg['fresh_base'] and cfg['learning_rate']==1e-4
    assert cfg['worker_budget_seconds']==7200 and cfg['method']=='PhysicalFFN_SVD540_qv'
    assert cfg['source_sha256']==sha(Path(__file__).with_name('run.py'))==c['source_sha256']
    assert cfg['ready_sha256']==sha(REPORT/'READY.json')
    assert cfg['manifest_sha256']==ready['manifest_sha256'] and cfg['split_sha256']==ready['split_sha256']
    lay=json.loads((WORK/'LAYOUT.json').read_text());assert sha(WORK/'LAYOUT.json')==c['layout_sha256']
    assert lay['weight_coefficients']==FINAL+TRAINABLE and lay['trainable_coordinates']==TRAINABLE
    assert not any(n.endswith(('.gate_proj.weight','.up_proj.weight','.down_proj.weight')) for n in lay['parameters'])
    factors=[v for n,v in lay['parameters'].items() if n.endswith(('.left','.right'))]
    assert len(factors)==144 and all(not v['trainable'] and v['dtype']=='torch.float16' for v in factors)
    assert c['full_checkpoint_reload']['passed'] and c['full_checkpoint_reload']['layout']==lay
    assert c['full_checkpoint_reload']['before_digest']==c['full_checkpoint_reload']['after_digest']
    compression=json.loads((WORK/'COMPRESSION.json').read_text())
    assert sha(WORK/'COMPRESSION.json')==c['compression_sha256'] and len(compression['matrices'])==72
    assert all(r['rank']==540 and sorted(r['original_shape'])==[896,4864] for r in compression['matrices'])
    assert sum(r['original_coefficients']-r['factor_coefficients'] for r in compression['matrices'])==89837568
    gate=json.loads((WORK/'NATIVE_GATE.json').read_text())
    assert sha(WORK/'NATIVE_GATE.json')==c['native_gate_sha256']
    assert gate['complete'] and len(gate['gates'])==19 and gate['NLL_paths']==gate['short_generations']==38
    assert json.loads((WORK/'TRAIN_COMPLETE.json').read_text())['updates']==87
    events=[json.loads(s) for s in (WORK/'logs/events.jsonl').read_text().splitlines()]
    assert len(events)==87 and sha(WORK/'logs/events.jsonl')==c['events_sha256']
    gradfiles={};optfiles={}
    for row in plan:
        k=row['stage'];v=json.loads((WORK/'validity'/f'stage_{k:02d}.json').read_text())
        assert v['passed'] and v['actual_parameters_changed'] and v['positive_lr_updates']==v['updates']==row['updates']
        es=[e for e in events if e['stage']==k]
        assert [e['step'] for e in es]==list(range(1,row['updates']+1)) and es[-1]['qa_seen']==es[-1]['total_qa']==row['QA']
        for e in es:
            assert all(math.isfinite(e[x]) for x in ['loss','gradient_l2_preclip','gradient_l2_postclip'])
            assert all(math.isfinite(lr) and lr>0 for lr in e['learning_rate_used'])
            p=WORK/'gradients'/f"S{k:02d}_{e['step']:03d}.json";g=json.loads(p.read_text())['raw']
            assert g and all(v['finite'] and math.isfinite(v['norm']) for v in g.values())
            assert set(g)=={n for n,v in lay['parameters'].items() if v['trainable']}
            gradfiles[p.relative_to(ROOT).as_posix()]=sha(p)
            p=WORK/'optimizer'/f"S{k:02d}_{e['step']:03d}.json";o=json.loads(p.read_text())
            assert o['parameter_IDs_match'] and o['coordinates']==TRAINABLE
            assert o['gradient_bytes']==TRAINABLE*4 and o['two_moment_bytes']==TRAINABLE*8
            optfiles[p.relative_to(ROOT).as_posix()]=sha(p)
        p=WORK/'gradients'/f'S{k:02d}_first_raw.pt';gradfiles[p.relative_to(ROOT).as_posix()]=sha(p)
    assert gradfiles==c['gradient_sha256'] and len(gradfiles)==106
    assert optfiles==c['optimizer_sha256'] and len(optfiles)==87
    refs=json.loads((study.REPORT/'SCREEN_COMPLETE.json').read_text())['all_four_results']
    comparisons={n:paired(scores,anchor_scores(m,dev,refs[n]),split) for n in ['LoRA_qv','OSFT_all']}
    return dict(complete=True,quality=quality,comparisons=comparisons,
        possible_research_followup_only=quality['domain_macro']>refs['LoRA_qv']['domain_macro'],
        final_folded_weight_coefficients=FINAL,unfolded_weight_coefficients=FINAL+TRAINABLE,
        complete_sha256=sha(REPORT/'COMPLETE.json'),ready_sha256=sha(REPORT/'READY.json'),
        automatic_full_confirmation=False,not_new_algorithm_or_stable_gain=True,TEST_QA=0,
        new_pairs_only=True,old_F1_not_recomputed=True)

if __name__=='__main__':
    import time
    start=time.monotonic();proof=verify();proof['CPU_verification_seconds']=time.monotonic()-start
    write(REPORT/'VERIFIED.json',proof)
    print(json.dumps(dict(verified=True,verified_sha256=sha(REPORT/'VERIFIED.json'))))
