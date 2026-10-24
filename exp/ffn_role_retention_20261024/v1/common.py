"""Frozen data and one final original-score consumer. No model import here."""
import ast,copy,hashlib,json,math,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).parent
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/final_weight_reduction_20261024/ffn_role_retention_v1'
WORK=HERE/'private'
BASE=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
SPLIT=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
MANIFEST=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
ANCHOR=ROOT/'summary/edge_deployment_20261019/position_coordinate_control_20261022/v1/VERIFIED.json'
CASES=('init','KeepGate','KeepUp')
BUDGETS={'init':1800,'KeepGate':3600,'KeepUp':3600}
SOURCES=('core.py','common.py','worker.py','freeze.py','dispatch.py')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        uid=int(next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')))
        return dict(pid=pid,startticks=s[19],uid=uid,pgid=int(s[2]))
    except FileNotFoundError:return None


def data():
    assert sha(MANIFEST)=='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    assert sha(SPLIT)=='d8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
    m=json.loads(MANIFEST.read_bytes());s=json.loads(SPLIT.read_bytes());fit={};dev={};plan={}
    for t in m['tasks']:
        tid=t['task'];f=t['files']['train'];p=ROOT/f['path'];assert sha(p)==f['sha256']
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        by={r['id']:r for r in rows};assert len(by)==f['count']
        fit[tid]=[by[i] for i in s['fit_ids'][str(tid)]];dev[tid]=[by[i] for i in s['dev_ids'][str(tid)]]
        order=sorted(fit[tid],key=lambda r:hashlib.sha256(f"42:{tid}:{r['id']}".encode()).digest())
        t['training_order_sha256']=hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
        plan[str(tid)]=[r['id'] for r in order]
    assert sum(map(len,fit.values()))==1213 and sum(map(len,dev.values()))==100
    assert sum(math.ceil(len(x)/16) for x in fit.values())==87
    return m,s,fit,dev,plan


def train_ast():
    tree=ast.parse((PORT/'run.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='stage')
    cutoff=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='wanted' for t in n.targets))
    prefix=copy.deepcopy(loop.body[:cutoff-1]);assert isinstance(prefix[-1],ast.If)
    code=ast.fix_missing_locations(ast.Module(body=[ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_stage(stage)').body,orelse=[])],type_ignores=[]))
    return code,hashlib.sha256(ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False).encode()).hexdigest()


def ready():
    r=json.loads((REPORT/'READY.json').read_bytes())
    assert r['uid']==os.getuid() and r['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for p,h in r['files'].items():assert sha(ROOT/p)==h,p
    return r


def verify():
    assert not (REPORT/'VERIFY_STARTED.json').exists() and not (REPORT/'VERIFIED.json').exists()
    r=ready();write(REPORT/'VERIFY_STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    assert sha(ANCHOR)=='acaee0246525532d99318f3798fa0116abd968fa8476a9815284593473fecb56'
    anchor=json.loads(ANCHOR.read_bytes())['quality']['domain_macro'];assert anchor==43.79198740853657
    m,s,fit,dev,plan=data();by_id={x['id']:(t,x) for t in m['tasks'] for x in dev[t['task']]}
    from collections import Counter
    import re
    tree=ast.parse((PORT/'runtime_utils.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')]
    env=dict(Counter=Counter,re=re);exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-score-AST','exec'),env)
    rows={};proofs={};elapsed=0;selected_gpu=None
    init_meta=json.loads((WORK/'init/initializations.json').read_bytes())
    for case in CASES:
        complete=json.loads((REPORT/(case+'_COMPLETE.json')).read_bytes());exit_=json.loads((REPORT/(case+'_EXIT.json')).read_bytes())
        assert complete['complete'] and exit_['returncode']==0 and exit_['observed_elapsed_seconds']<=BUDGETS[case]
        assert complete['ready_sha256']==sha(REPORT/'READY.json') and complete['worker']==exit_['worker']
        start=json.loads((REPORT/(case+'_STARTED.json')).read_bytes())
        assert all(start[k]==complete['worker'][k] for k in complete['worker'])
        assert start['boot_id']==r['boot_id'] and start['ready_sha256']==sha(REPORT/'READY.json')
        assert exit_['identity_after'] is None and complete['seconds']<=BUDGETS[case]
        selected_gpu=start['GPU'] if selected_gpu is None else selected_gpu
        assert start['GPU']==selected_gpu
        elapsed+=exit_['observed_elapsed_seconds']
        if case=='init':
            expected={'init/parent.jsonl','init/initializations.json','init/resources.jsonl','KeepGate_CP0.pt','KeepUp_CP0.pt'}
            assert {x['path'] for x in init_meta['checkpoints']}=={'KeepGate_CP0.pt','KeepUp_CP0.pt'}
            for meta in init_meta['checkpoints']:
                assert meta['sha256']==complete['files'][meta['path']] and meta['bytes']==(WORK/meta['path']).stat().st_size
        else:
            names={'initial.jsonl','final.jsonl','final_complete.pt','result_proof.json','resources.jsonl','optimizer.jsonl','logs/events.jsonl','run_status.json'}
            names|={f'validity/stage_{i:02d}.json' for i in range(1,20)}
            names|={f'stage_adapters/{i:02d}.pt' for i in range(1,20)}
            names|={f'gradients/{i:03d}.pt' for i in range(1,88)}
            expected={case+'/'+n for n in names}
            proof=json.loads((WORK/case/'result_proof.json').read_bytes());meta=proof['final_checkpoint']
            assert meta['path']==case+'/final_complete.pt' and meta['sha256']==complete['files'][meta['path']]
            assert meta['bytes']==(WORK/meta['path']).stat().st_size
            assert set(proof['counts'])==set(map(str,range(1,20))) and sum(proof['counts'].values())==87
            for i in range(1,20):
                validity=json.loads((WORK/case/f'validity/stage_{i:02d}.json').read_bytes())
                assert validity.get('passed',validity.get('valid',False)),validity
        assert set(complete['files'])==expected
        actual={str(p.relative_to(WORK)) for p in (WORK/case).rglob('*') if p.is_file()}
        assert actual=={n for n in expected if n.startswith(case+'/')}
        for p,h in complete['files'].items():assert sha(WORK/p)==h,p
        resources=[json.loads(x) for x in (WORK/case/'resources.jsonl').read_text().splitlines()]
        assert resources and resources[-1]['phase']=='complete'
        assert all(x['pid']==complete['worker']['pid'] and x['visible_gpu']==str(selected_gpu) for x in resources)
        if case!='init':
            assert complete['updates']==87 and complete['stages']==19 and complete['FIT_QA']==1213
            assert complete['frozen_before']==complete['frozen_after']
            assert complete['final_layout']['weights']['coefficients']==816728096
            assert complete['final_layout']==complete['reload_layout'] and complete['final_digest']==complete['reload_digest']
            assert complete['qv_initial_sha256']==json.loads((REPORT/'KeepGate_COMPLETE.json').read_bytes())['qv_initial_sha256']
            witnesses=[json.loads(x) for x in (WORK/case/'optimizer.jsonl').read_text().splitlines()]
            assert len(witnesses)==87 and all(w['positive_lr'] and w['parameter_count']==540672 and w['finite'] for w in witnesses)
            assert len(list((WORK/case/'gradients').glob('*.pt')))==87 and len(list((WORK/case/'stage_adapters').glob('*.pt')))==19
            actual=[i for w in witnesses for i in w['ids']];expected=[i for t in m['tasks'] for i in plan[str(t['task'])]];assert actual==expected
        for phase in (('parent',) if case=='init' else ('initial','final')):
            p=WORK/case/(phase+'.jsonl');records=[json.loads(x) for x in p.read_text().splitlines()]
            expected_ids=[x['id'] for t in m['tasks'] for x in dev[t['task']]]
            assert [x['id'] for x in records]==expected_ids
            dom={};scalars=[]
            for v in records:
                t,gold=by_id[v['id']];assert v['references']==gold['answers'] and v['images']=={n:s['image_sha256'][n] for n in gold['images']}
                assert v['generation_cap']==t['max_new_tokens'] and 0<len(v['output_token_ids'])<=t['max_new_tokens']
                assert v['finite_score_checks']==len(v['output_token_ids'])
                assert v['worker']==complete['worker'] and v['input_sha256']
                f1=env['score'](v['prediction'],gold['answers'])['token_f1'];dom.setdefault(t['domain'],[]).append(f1)
                scalars.append(dict(id=v['id'],F1=f1,domain=t['domain']))
            means={k:sum(v)/len(v) for k,v in dom.items()};assert len(means)==5 and all(len(v)==20 for v in dom.values())
            rows[case+'_'+phase]=dict(domains=means,macro=sum(means.values())/5,per_id=scalars,prediction_sha256=sha(p))
        proofs[case]=sha(REPORT/(case+'_COMPLETE.json'))
    assert elapsed<=9000
    g=rows['KeepGate_final']['macro'];u=rows['KeepUp_final']['macro']
    write(REPORT/'VERIFIED.json',dict(complete=True,rows=rows,case_complete_sha256=proofs,ready_sha256=sha(REPORT/'READY.json'),
         old_wideqv_anchor=anchor,old_wideqv_proof_sha256=sha(ANCHOR),primary_point_pass=g>u and g>anchor,model_elapsed_observed_sum=elapsed,
         conditions='explored DEV, single seed, point screening only; not causal trainability or novel-framework proof',TEST_QA=0))
