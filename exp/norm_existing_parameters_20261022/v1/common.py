"""Frozen one-condition FIT/DEV existing-norm comparison; never official TEST."""
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
    assert v['trainable_coordinates']==43008 and v['method']=='RMSNorm_only'
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
    return dict(primary='RMSNorm_only_macro_strictly_above_both_anchors',secondary='paired_against_both_anchors',comparisons=result,
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
    assert complete['config']['trainable_coordinates']==43008 and complete['config']['target_modules']==48
    assert complete['config']==json.loads((WORK/'config.json').read_text())
    assert sha(WORK/'config.json')==complete['config_sha256']
    cfg=complete['config']
    assert cfg['learning_rate']==1e-4 and cfg['method']=='RMSNorm_only'
    assert not cfg['host_optimizer'] and not cfg['offload_activations']
    assert cfg['master_dtype']=='float32' and cfg['forward_weight_dtype']=='float16'
    assert cfg['frozen_digest_before_TRAIN']==cfg['frozen_digest_after_TRAIN']
    assert complete['forward_digest_before_DEV']==complete['forward_digest_after_DEV']
    assert (cfg['FIT_QA'],cfg['DEV_QA'],cfg['TEST_QA'],cfg['updates'])==(1213,100,0,87)
    assert cfg['ready_sha256']==sha(REPORT/'READY.json') and cfg['source_sha256']==complete['source_sha256']
    assert cfg['manifest_sha256']==ready['manifest_sha256'] and cfg['split_sha256']==ready['split_sha256']
    train=json.loads((WORK/'TRAIN_COMPLETE.json').read_text())
    assert train['complete'] and train['updates']==87 and train['update_evidence_sha256']==complete['update_evidence_sha256']
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
            raw=json.loads(path.read_text())['raw'];assert raw and all(g['finite'] and math.isfinite(g['norm']) for g in raw.values())
            rawfiles[path.relative_to(ROOT).as_posix()]=sha(path)
        path=WORK/'gradients'/f'S{k:02d}_first_raw.pt';rawfiles[path.relative_to(ROOT).as_posix()]=sha(path)
    assert rawfiles==complete['gradient_sha256'] and len(rawfiles)==106
    refs=json.loads((study.REPORT/'SCREEN_COMPLETE.json').read_text())['all_four_results'];old={}
    for c in ['LoRA_qv','OSFT_all']:
        v=refs[c];q,s=read_predictions(ROOT/v['checkpoint_root'],m,dev,v['prediction_sha256'])
        assert q['domains']==v['domains'] and q['domain_macro']==v['domain_macro'];old[c]=s
    # Load only our two tiny private gain snapshots, not the model or QA tensors.
    import torch
    from core import original_norm_source
    assert original_norm_source()==ready['original_norm_source']
    snapshots={}
    for name in ['initial_gamma.pt','final_gamma.pt']:
        p=WORK/name;assert sha(p)==complete['gamma_sha256'][name]
        snapshots[name]=torch.load(p,map_location='cpu',weights_only=True)
    first,last=snapshots.values();assert first.keys()==last.keys() and len(first)==48
    assert sum(v['master'].numel() for v in first.values())==43008
    for snapshot in snapshots.values():
        for name,value in snapshot.items():
            assert name.endswith(('.input_layernorm.weight','.post_attention_layernorm.weight'))
            assert value['master'].dtype==torch.float32 and value['forward'].dtype==torch.float16
            assert torch.equal(value['master'].half(),value['forward'])
            assert all(torch.isfinite(t).all() for t in value.values())
    delta={kind:dict(changed=sum(int((first[n][kind]!=last[n][kind]).sum()) for n in first),
        l2=float(torch.cat([(last[n][kind].float()-first[n][kind].float()).reshape(-1) for n in first]).norm()))
        for kind in ['master','forward']}
    assert delta==complete['final_gain_change']
    assert sha(WORK/'gain_updates.json')==complete['update_evidence_sha256']
    updates=json.loads((WORK/'gain_updates.json').read_text());assert len(updates)==87
    for p in plan:
        rows=[r for r in updates if r['stage']==p['stage']]
        assert [r['step'] for r in rows]==list(range(1,p['updates']+1))
        for r in rows:
            assert r['updates']==p['updates']
            for label in ['from_previous']+(['from_stage_start'] if r['step']==p['updates'] else []):
                for kind in ['master','forward']:
                    v=r[label][kind];assert v['total']==43008 and 0<=v['changed']<=43008 and math.isfinite(v['l2'])
    assert not torch.cuda.is_initialized()
    comparisons=paired(scores,old,split)
    threshold=max(refs[c]['domain_macro'] for c in ['LoRA_qv','OSFT_all'])
    return dict(complete=True,quality=quality,paired=comparisons,FIT_QA=1213,DEV_QA=100,updates=87,TEST_QA=0,
        complete_sha256=sha(REPORT/'COMPLETE.json'),ready_sha256=sha(REPORT/'READY.json'),
        source_sha256=sha(__file__),checkpoint=complete['checkpoint'],new_model_requests=0,
        possible_followup_fresh_PAS_reference=quality['domain_macro']>threshold,
        fixed_anchor_threshold=threshold,final_gain_change=delta,actual_forward_update_evidence=updates,
        not_whole_function_change_proof=True,automatic_full_confirmation=False,
        original_screen_and_full_confirmation_unchanged=True)

if __name__=='__main__':
    import time
    begin=time.monotonic();proof=verify();proof['CPU_verification_seconds']=time.monotonic()-begin
    write(REPORT/'VERIFIED.json',proof)
    print(json.dumps(dict(verified=True,verified_sha256=sha(REPORT/'VERIFIED.json'))))
