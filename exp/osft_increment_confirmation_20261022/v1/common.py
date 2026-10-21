"""One fixed full-data confirmation; inherited kernels stay immutable."""
import ast
from collections import Counter
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT/'exp/osft_increment_screen_20261019/v1'
spec = importlib.util.spec_from_file_location('confirmation_inherited_study', PILOT/'study.py')
study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
PORT = study.PORT
sha, write = study.sha, study.write
REPORT = ROOT/'summary/edge_deployment_20261019/osft_increment_confirmation_20261022/v1'
WORK = Path(__file__).parent/'private'
SCREEN = study.REPORT
CASE = 'LoRA_all'
ANCHORS = {
 'PASs_MoE': {'macro':46.619976395158304,'AP19':46.40616871326078,
  'domains':dict(zip(['natural','medical','geography','finance','traffic'],[70.69669151505542,44.20167545468118,55.73770491803279,25.223205725465,37.24060436255713]))},
 'OSFT': {'macro':45.93743821034059,'AP19':46.35669274628726,
  'domains':dict(zip(['natural','medical','geography','finance','traffic'],[73.90668242541788,43.89408925018375,51.7093162734906,23.981576965682596,36.195526136928116]))}}

def score_function():
    tree=ast.parse((PORT/'runtime_utils.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['score','tokens']]
    ns=dict(re=re,Counter=Counter)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-score>','exec'),ns)
    return ns['score']

def original_data():
    path,m,train=study.original_train()
    assert [t['task'] for t in m['tasks']]==list(range(1,20))
    lockpath=path.parent.parent/'split_lock.json'
    lock=json.loads(lockpath.read_text())['conditions']['pct01']
    assert sha(path)==lock['manifest_sha256']
    assert sha(path.parent/'frozen_ids.json')==lock['frozen_ids_sha256']
    assert m['formal_ready'] and not m['assets_pending'] and m['condition']=='pct01'
    assert m['experiment_id']=='five_domain_native_cl_20261016_sparse'
    for t in m['tasks']:
        for f in t['files'].values(): assert f['sha256']==lock['files'][f['path']]
    assert len({r['id'] for rows in train.values() for r in rows})==1313
    plan=[]
    import hashlib
    for t in m['tasks']:
        rows=train[t['task']]
        ordered=sorted(rows,key=lambda r:hashlib.sha256(f"42:{t['task']}:{r['id']}".encode()).digest())
        assert hashlib.sha256(''.join(r['id']+'\n' for r in ordered).encode()).hexdigest()==t['training_order_sha256']
        plan.append(dict(task=t['task'],QA=len(rows),updates=math.ceil(len(rows)/16)))
    assert sum(t['updates'] for t in plan)==93
    return path,m,train,plan,lockpath

def load_test_after_training(m,train):
    assert (WORK/'TRAIN_COMPLETE.json').exists(), 'No official TEST before completed full training'
    data={}; allids=[]; trainids={r['id'] for rows in train.values() for r in rows}
    for t in m['tasks']:
        f=t['files']['test'];p=ROOT/f['path'];assert sha(p)==f['sha256']
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        assert len(rows)==f['count']
        allids.extend(r['id'] for r in rows);data[t['task'],'test']=rows
    assert len(allids)==21454 and len(set(allids))==21454 and trainids.isdisjoint(allids)
    return data

def check_ready(include_base=True):
    ready=json.loads((REPORT/'READY.json').read_text())
    for p,h in ready['sources'].items(): assert sha(ROOT/p)==h,p
    for p,h in ready['evidence'].items(): assert sha(ROOT/p)==h,p
    if include_base:
        for p,h in ready['base_files'].items(): assert sha(ROOT/p)==h,p
    assert ready['train_AST_sha256']==study.training_ast()[1]
    assert ready['anchors']==ANCHORS
    return ready

def prediction_summary(m,data):
    scorer=score_function();domains={};tasks=[];files={};seen=[]
    for t in m['tasks']:
        p=WORK/'evaluations/stage_19'/f"T{t['task']:02d}_test_predictions.jsonl"
        rows=[json.loads(s) for s in p.read_text().splitlines()];gold=data[t['task'],'test']
        assert [r['id'] for r in rows]==[r['id'] for r in gold]
        values=[]
        for row,g in zip(rows,gold):
            assert row['references']==g['answers'] and row['generation_cap']==t['max_new_tokens']
            assert row['scores']==scorer(row['prediction'],g['answers'])
            values.append(row['scores']['token_f1']);seen.append(row['id'])
        domains.setdefault(t['domain'],[]).extend(values)
        tasks.append(dict(task=t['task'],domain=t['domain'],F1=sum(values)/len(values),QA=len(values)))
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    assert len(seen)==len(set(seen))==21454 and len(domains)==5
    domains={d:dict(F1=sum(v)/len(v),QA=len(v)) for d,v in domains.items()}
    macro=sum(v['F1'] for v in domains.values())/5
    return dict(tasks=tasks,domains=domains,domain_macro=macro,AP19=sum(t['F1'] for t in tasks)/19,
        FWT=None,BWT=None,prediction_sha256=files,TEST_QA=21454)

def verify():
    ready=check_ready(False);v=json.loads((REPORT/'COMPLETE.json').read_text())
    assert v['complete'] and v['ready_sha256']==sha(REPORT/'READY.json')
    assert v['source_sha256']==sha(Path(__file__).with_name('run.py'))
    assert v['TRAIN_QA']==1313 and v['updates']==93 and v['coverage']=='final_only_19_tasks_not_74_cells'
    _,m,train,plan,_=original_data();data=load_test_after_training(m,train)
    assert prediction_summary(m,data)==v['quality']
    assert sha(WORK/v['checkpoint']['path'])==v['checkpoint']['sha256']
    import math
    events=[json.loads(s) for s in (WORK/'logs/events.jsonl').read_text().splitlines()]
    assert len(events)==93 and len(list((WORK/'gradients').glob('S*_first_raw.pt')))==19
    rawfiles={}
    for expected in plan:
        stage=expected['task'];valid=json.loads((WORK/'validity'/f'stage_{stage:02d}.json').read_text())
        assert valid['passed'] and valid['actual_parameters_changed']
        assert valid['updates']==valid['positive_lr_updates']==expected['updates']
        ev=[e for e in events if e['stage']==stage];assert len(ev)==expected['updates']
        assert ev[-1]['qa_seen']==ev[-1]['total_qa']==expected['QA']
        assert [e['step'] for e in ev]==list(range(1,len(ev)+1))
        for e in ev:
            assert math.isfinite(e['loss']) and math.isfinite(e['gradient_l2_preclip'])
            assert math.isfinite(e['gradient_l2_postclip']) and e['gradient_l2_postclip']>=0
            assert all(math.isfinite(lr) and lr>0 for lr in e['learning_rate_used'])
            p=WORK/'gradients'/f"S{stage:02d}_{e['step']:03d}.json"
            raw=json.loads(p.read_text())['raw'];assert raw and all(g['finite'] and math.isfinite(g['norm']) for g in raw.values())
            rawfiles[p.relative_to(ROOT).as_posix()]=sha(p)
        p=WORK/'gradients'/f'S{stage:02d}_first_raw.pt';rawfiles[p.relative_to(ROOT).as_posix()]=sha(p)
    assert v['trainable_digest_before_TEST']==v['trainable_digest_after_TEST']
    assert v['config']['trainable_parameters']==4399104 and v['config']['target_modules']==168
    assert sha(WORK/'logs/events.jsonl')==v['events_sha256']
    assert rawfiles==v['gradient_sha256']
    return dict(complete=True,quality=v['quality'],checkpoint=v['checkpoint'],
        completion_sha256=sha(REPORT/'COMPLETE.json'),ready_sha256=sha(REPORT/'READY.json'),
        source_sha256=sha(__file__),TRAIN_QA=1313,updates=93,gradient_files=len(rawfiles),
        final_only_not_original_74_cell_baseline=True,new_model_requests=0,
        primary_pass=v['quality']['domain_macro']>max(a['macro'] for a in ANCHORS.values()))
