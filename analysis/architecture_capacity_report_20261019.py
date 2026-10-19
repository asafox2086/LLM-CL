"""Read-only sealed capacity outcomes; no TEST-driven selection or GPU use."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/architecture_capacity/v1'
WORK=ROOT/'exp/architecture_capacity_20261019/v1/private'
sys.path.insert(0,str(ROOT/'exp/architecture_capacity_20261019/v1'))
from capacity_core import CASES


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def write(p,v):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)


def main():
    paths=list(REPORT.glob('*_train_COMPLETE.json'))+list(REPORT.glob('*_full_COMPLETE.json'))
    markers={p.name:sha(p) for p in paths}
    prior=REPORT/'VERIFIED_RESULTS.json'
    if prior.exists() and read(prior)['completion_markers']==markers:
        print('No newly sealed outcomes; no repeated scan');return
    ready=read(REPORT/'READY.json')
    assert all(sha(ROOT/p)==h for p,h in ready['sources'].items())
    port=ROOT/'exp/two_backbone_native_cl_20261017_v2'
    # Compile the original scoring functions only; importing the trainer is unnecessary.
    import ast,re
    from collections import Counter
    tree=ast.parse((port/'runtime_utils.py').read_text())
    ns=dict(re=re,Counter=Counter)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['tokens','score']]
    assert len(nodes)==2
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<frozen-AnswerF1>','exec'),ns)
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest_path)==ready['manifest_sha256'];manifest=read(manifest_path)
    seen_path=ROOT/'summary/onevision_diagnostics_20261016/learned_reuse/CPU_PLAN_CHECK.json'
    seen_spec=read(seen_path)['spec'];seen={i for ids in seen_spec['diagnostic_ids'].values() for i in ids}
    assert len(seen)==500
    outcomes={}
    for case in CASES:
        train_file=REPORT/(case+'_train_COMPLETE.json')
        if not train_file.exists():continue
        train=read(train_file);run=WORK/case
        assert train['complete'] and train['TRAIN_QA']==1313 and train['updates']==93 and train['stages']==19
        assert train['ready_sha256']==sha(REPORT/'READY.json') and train['source_sha256']==sha(ROOT/'exp/architecture_capacity_20261019/v1/run.py')
        events=[json.loads(s) for s in (run/'logs/events.jsonl').read_text().splitlines()]
        assert len(events)==93
        assert all(math.isfinite(e['loss']) and math.isfinite(e['gradient_l2_preclip']) and all(x>0 for x in e['learning_rate_used']) for e in events)
        updates=0
        for stage in range(1,20):
            v=read(run/'validity'/f'stage_{stage:02d}.json')
            expected=math.ceil(manifest['tasks'][stage-1]['files']['train']['count']/16)
            assert v['passed'] and v['actual_parameters_changed'] and v['updates']==v['positive_lr_updates']==expected
            updates+=expected
        assert updates==93
        cp=train['checkpoint'];assert sha(ROOT/cp['path'])==cp['sha256']
        cfg=read(run/'config.json');assert cfg['capacity']['experts']==CASES[case]['experts'] and cfg['capacity']['total_rank']==CASES[case]['total_rank']
        outcome=dict(train_verified=True,full_verified=False,checkpoint_sha256=cp['sha256'],
            capacity=cfg['capacity'],parameter_inventory=cfg['inventory'],TEST_QA=None,AP19=None,domains=None,
            observed_training_memory=read(run/'memory_summary.json'))
        full_file=REPORT/(case+'_full_COMPLETE.json')
        if full_file.exists():
            full=read(full_file);reload=read(run/'FRESH_RELOAD.json')
            assert full['complete'] and full['TEST_QA']==21454 and full['ready_sha256']==sha(REPORT/'READY.json')
            assert full['checkpoint_sha256']==cp['sha256'] and full['source_sha256']==train['source_sha256']
            assert reload['complete'] and reload['all_adapter_parameters_exact'] and reload['nineteen_TRAIN_NLL_exact']
            assert reload['checkpoint_sha256']==cp['sha256']
            per_task=[];domain={};scopes={'prior_diagnostic_500':[],'remaining_20954':[]};files={};all_ids=set()
            for task in manifest['tasks']:
                meta=task['files']['test'];gold_path=ROOT/meta['path'];assert sha(gold_path)==meta['sha256']
                gold=[json.loads(s) for s in gold_path.read_text().splitlines()]
                p=run/'evaluations/stage_19'/f"T{task['task']:02d}_test_predictions.jsonl"
                records=[json.loads(s) for s in p.read_text().splitlines()]
                assert len(records)==len(gold)==meta['count'];scores=[]
                for record,row in zip(records,gold):
                    assert record['id']==row['id'] and record['references']==row['answers']
                    value=ns['score'](record['prediction'],row['answers'])['token_f1']
                    assert value==record['scores']['token_f1']
                    assert row['id'] not in all_ids;all_ids.add(row['id'])
                    scores.append(value);domain.setdefault(task['domain'],[]).append(value)
                    scopes['prior_diagnostic_500' if row['id'] in seen else 'remaining_20954'].append(value)
                per_task.append(sum(scores)/len(scores));files[str(p.relative_to(ROOT))]=sha(p)
            assert len(all_ids)==21454 and seen<=all_ids and len(scopes['prior_diagnostic_500'])==500 and len(scopes['remaining_20954'])==20954
            domains={d:dict(F1=sum(s)/len(s),QA=len(s)) for d,s in domain.items()}
            ap=sum(per_task)/19;macro=sum(d['F1'] for d in domains.values())/len(domains)
            assert ap==full['AP19'] and domains==full['domains'] and macro==full['domain_macro']
            outcome.update(full_verified=True,TEST_QA=21454,AP19=ap,domains=domains,domain_macro=macro,
                FWT=None,BWT=None,per_scope_QA_micro_F1={k:dict(QA=len(v),F1=sum(v)/len(v)) for k,v in scopes.items()},
                prediction_files_sha256=files,not_deployment_profile=True)
        outcomes[case]=outcome
    write(prior,dict(completion_markers=markers,verified=outcomes,source_sha256=sha(__file__),
        scoring_source_sha256=sha(port/'runtime_utils.py'),prior_diagnostic_ids_source_sha256=sha(seen_path),
        no_TEST_selection=True,no_deployment_advantage_claim=True))
    lines=['# 架构扩展必要性：已封存结果','',
        '六组容量消融；空格表示未完整封存，不是0分。原Answer F1，AP19沿用原定义，未补测前缀的FWT/BWT不计算。',
        '', '| 条件 | 专家数 | 总rank | 自然 | 医学 | 遥感 | 金融 | 交通 | AP19 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case,spec in CASES.items():
        o=outcomes.get(case,{});scores=[]
        for d in ['natural','medical','geography','finance','traffic']:
            scores.append(f"{o['domains'][d]['F1']:.3f}" if o.get('full_verified') else '')
        ap=f"{o['AP19']:.3f}" if o.get('full_verified') else ''
        lines.append('| '+' | '.join([case,str(spec['experts']),str(spec['total_rank'])]+scores+[ap])+' |')
    lines += ['', '仅原／少专家同rank的配对检验分工；同rank／小rank配对检验容量。数值相近不自动证明等价、结构无用或一般性；单seed／单骨干范围保留。',
        '', '完整包、CPU／GPU内存与速度未验证前，不宣称部署优势，不写正式Ours。训练内存不当作部署内存。']
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(sealed_training=sum(o['train_verified'] for o in outcomes.values()),
        sealed_full=sum(o['full_verified'] for o in outcomes.values()),performance_NOT_inferred_from_admission=True)))


if __name__=='__main__':main()
