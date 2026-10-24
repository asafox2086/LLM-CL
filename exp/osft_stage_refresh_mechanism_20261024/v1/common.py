"""Frozen S1 selection, parent-proof binding and unique original-score consumer."""
import ast,hashlib,json,math,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).parent
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
BASE=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
REPORT=ROOT/'summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1'
WORK=HERE/'private'
SPLIT=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
MANIFEST=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
CASES=('prefix','Fixed','Refresh')
BUDGETS={'prefix':3600,'Fixed':7200,'Refresh':7200}
TRAINABLE=113766144
SOURCES=('core.py','common.py','worker.py','freeze.py','dispatch.py')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        uid=int(next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')))
        return dict(pid=pid,startticks=s[19],uid=uid,pgid=int(s[2]))
    except FileNotFoundError:return None


def h(s):return hashlib.sha256(s.encode()).hexdigest()


def metadata():
    assert sha(MANIFEST)=='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    assert sha(SPLIT)=='d8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
    return json.loads(MANIFEST.read_bytes()),json.loads(SPLIT.read_bytes())


def data():
    import copy
    m,s=metadata();m=copy.deepcopy(m);fit={};dev={};plan=[]
    for task in m['tasks']:
        k=task['task'];p=ROOT/task['files']['train']['path'];assert sha(p)==task['files']['train']['sha256']
        wanted=set(s['fit_ids'][str(k)])|set(s['dev_ids'][str(k)])
        rows={r['id']:r for r in map(json.loads,p.read_text().splitlines()) if r['id'] in wanted}
        assert set(rows)==wanted
        fit[k]=sorted([rows[i] for i in s['fit_ids'][str(k)]],key=lambda r:h(f"42:{k}:{r['id']}"))
        dev[k]=[rows[i] for i in s['dev_ids'][str(k)]]
        task['training_order_sha256']=h(''.join(row['id']+'\n' for row in fit[k]))
        plan.append(dict(stage=k,FIT=len(fit[k]),updates=math.ceil(len(fit[k])/16),DEV=len(dev[k])))
    assert sum(map(len,fit.values()))==1213 and sum(map(len,dev.values()))==100 and sum(x['updates'] for x in plan)==87
    assert len(fit[1])==327 and len(dev[1])==20 and plan[0]['updates']==21
    return m,s,fit,dev,plan


def training_prefix():
    tree=ast.parse((PORT/'run.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='stage')
    cut=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='wanted' for t in n.targets))
    nodes=loop.body[:cut]
    return nodes,h(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False))


def ready():
    r=json.loads((REPORT/'READY.json').read_bytes())
    assert r['uid']==os.getuid() and r['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for n,v in r['files'].items():assert sha(ROOT/n)==v,n
    assert r['training_ast_sha256']==training_prefix()[1]
    return r


def verify():
    assert not (REPORT/'VERIFY_STARTED.json').exists() and not (REPORT/'VERIFIED.json').exists()
    for case in CASES:
        c=json.loads((REPORT/(case+'_COMPLETE.json')).read_bytes());e=json.loads((REPORT/(case+'_EXIT.json')).read_bytes())
        assert c['complete'] and e['returncode']==0 and e['identity_after'] is None
    r=ready();write(REPORT/'VERIFY_STARTED.json',dict(unix=time.time(),pid=os.getpid()))
    m,s,fit,dev,plan=data();truth={row['id']:(task,row) for task in m['tasks'] for row in dev[task['task']]}
    from collections import Counter
    import re
    tree=ast.parse((PORT/'runtime_utils.py').read_text());env=dict(Counter=Counter,re=re)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-score-AST','exec'),env)
    quality={};all_pred={};complete_sha={};total=0;gpu=None;shared=None;physical_updates=0;train_inputs={}
    def score_rows(rows):
        domains={};per=[]
        for row in rows:
            task,original=truth[row['id']]
            assert row['references']==original['answers'] and row['generation_cap']==task['max_new_tokens']
            assert row['images']=={n:s['image_sha256'][n] for n in original['images']}
            assert row['output_token_ids'] and row['finite_score_checks']>0
            value=env['score'](row['prediction'],row['references'])['token_f1']
            domains.setdefault(task['domain'],[]).append(value);per.append(dict(id=row['id'],domain=task['domain'],F1=value))
        return per,domains
    def aggregate(per,domains):
        assert len(per)==100 and len(domains)==5 and all(len(x)==20 for x in domains.values())
        values={d:sum(v)/len(v) for d,v in domains.items()}
        return dict(macro=sum(values.values())/5,domains=values,per_id=per)
    for case in CASES:
        c=json.loads((REPORT/(case+'_COMPLETE.json')).read_bytes());e=json.loads((REPORT/(case+'_EXIT.json')).read_bytes())
        start=json.loads((REPORT/(case+'_STARTED.json')).read_bytes())
        assert e['worker']==c['worker'] and all(start[k]==c['worker'][k] for k in c['worker'])
        assert c['ready_sha256']==start['ready_sha256']==sha(REPORT/'READY.json')
        assert c['seconds']<=BUDGETS[case] and e['observed_elapsed_seconds']<=BUDGETS[case]
        total+=e['observed_elapsed_seconds'];gpu=start['GPU'] if gpu is None else gpu;assert start['GPU']==gpu
        for n,v in c['files'].items():assert sha(WORK/n)==v,n
        assert c['unchanged_before']==c['unchanged_after']
        assert c['layout']['weights']['coefficients']==990913568
        assert c['layout']['trainable']==(TRAINABLE if case=='prefix' else 0)
        complete_sha[case]=sha(REPORT/(case+'_COMPLETE.json'))
        stages=[1] if case=='prefix' else list(range(2,20))
        witnesses=[json.loads(p.read_bytes()) for p in sorted((WORK/case/'witness').glob('*.json'))]
        assert len(witnesses)==c['actual_updates']==sum(plan[k-1]['updates'] for k in stages)
        physical_updates+=len(witnesses)
        prepared=[json.loads(x) for x in (WORK/case/'train_inputs.jsonl').read_text().splitlines()]
        train_inputs[case]=prepared
        assert [x['id'] for x in prepared]==[row['id'] for k in stages for row in fit[k]]
        assert len(prepared)==c['actual_FIT']==sum(len(fit[k]) for k in stages)
        for k in stages:
            w=[x for x in witnesses if x['stage']==k]
            assert [x['step'] for x in w]==list(range(1,plan[k-1]['updates']+1))
            assert all(x['finite'] and x['positive_lr'] and x['actual_step'] and x['coordinates']==TRAINABLE and
                x['gradient_bytes']==4*TRAINABLE and x['cpu_moments_bytes']==8*TRAINABLE and len(x['parameter_ids'])==504 for x in w)
            validity=json.loads((WORK/case/f'validity/stage_{k:02d}.json').read_bytes());assert validity['passed'] and validity['updates']==len(w)==validity['positive_lr_updates']
            state=json.loads((WORK/case/f'stages/S{k:02d}.json').read_bytes());assert state['digest_after_train']==state['digest_after_acquired']
            assert state['method_stage']==k
            if case!='prefix':
                boundary=json.loads((WORK/case/f'boundaries/S{k:02d}.json').read_bytes())
                assert boundary['prior_method_stage']==k-1 and boundary['modules']==168
                assert boundary['actual_refactor_calls']==(168 if case=='Refresh' else 0)
                if case=='Fixed':assert all(x['actual_FP16_changed_coordinates']==0 for x in boundary['matrices'])
        resources=[json.loads(x) for x in (WORK/case/'resources.jsonl').read_text().splitlines()]
        assert resources and resources[0]['phase']=='before_model' and resources[-1]['phase']=='complete'
        assert all(x['pid']==c['worker']['pid'] for x in resources)
        acquired=[json.loads(x) for x in (WORK/case/'acquired.jsonl').read_text().splitlines()]
        assert [x['id'] for x in acquired]==[row['id'] for task in m['tasks'] if task['task'] in stages for row in dev[task['task']]]
        all_pred[case]={'acquired':acquired};per,ds=score_rows(acquired)
        if case=='prefix':
            shared_scored=(per,ds)
            initial=[json.loads(x) for x in (WORK/case/'initial.jsonl').read_text().splitlines()]
            assert [x['id'] for x in initial]==[row['id'] for task in m['tasks'] for row in dev[task['task']]]
            pp,dd=score_rows(initial);quality['initial']=aggregate(pp,dd)
            all_pred[case]['initial']=initial
            shared=dict(checkpoint=c['checkpoint'],digest=c['shared_S1_digest'],rng_sha256=c['shared_rng_sha256'],rng_digest=c['shared_rng_digest'])
            assert c['checkpoint']['sha256']==c['files']['prefix/shared_S1.pt']
            assert c['shared_rng_sha256']==c['files']['prefix/rng.pt']
            continue
        assert c['loaded_shared_S1']==shared and c['restored_method_stage']==1
        assert c['final_digest']==c['reload_digest']==c['evaluation_digest_before']==c['evaluation_digest_after']
        def logical(value):
            return {kind:{k:v for k,v in item.items() if k not in ('backing_storage_bytes','unused_backing_bytes')}
                    if isinstance(item,dict) else item for kind,item in value.items()}
        assert logical(c['layout'])==logical(c['reload_layout']) and c['checkpoint']['sha256']==c['files'][case+'/final_complete.pt']
        final=[json.loads(x) for x in (WORK/case/'final.jsonl').read_text().splitlines()]
        assert [x['id'] for x in final]==[row['id'] for task in m['tasks'] for row in dev[task['task']]]
        # Shared S1 answers are scored once, then the verified scalar rows are copied below.
        all_pred[case]['final']=final
        pp,dd=score_rows(final);quality[case]={'final':aggregate(pp,dd),'branch_acquired_per_id':per,'branch_acquired_domains':ds}
    shared_per,shared_domains=shared_scored
    assert train_inputs['Fixed']==train_inputs['Refresh']
    for case in CASES[1:]:
        q=quality[case];pp=shared_per+q.pop('branch_acquired_per_id');dd=q.pop('branch_acquired_domains')
        for domain,values in shared_domains.items():dd.setdefault(domain,[]).extend(values)
        q['acquired']=aggregate(pp,dd)
    initial_by={x['id']:x for x in all_pred['prefix']['initial']}
    for phase in all_pred.values():
        for rows in phase.values():
            for row in rows:
                old=initial_by[row['id']]
                assert row['input_sha256']==old['input_sha256'] and row['eos_token_id']==old['eos_token_id']
    assert physical_updates==153 and total<=sum(BUDGETS.values())
    aq=quality['Refresh']['acquired']['macro']-quality['Fixed']['acquired']['macro']
    fq=quality['Refresh']['final']['macro']-quality['Fixed']['final']['macro']
    action='boundary_package_has_acquisition_point_support' if aq>0 else 'no_acquisition_point_support'
    if aq<=0 and fq<=0:action='prefer_omitting_package_at_these_point_outcomes'
    elif (aq>0 and fq<0) or (aq<0 and fq>0):action='acquisition_retention_tradeoff_no_universal_winner'
    write(REPORT/'VERIFIED.json',dict(complete=True,quality=quality,acquired_Refresh_minus_Fixed_pp=aq,final_Refresh_minus_Fixed_pp=fq,
        decision=action,case_complete_sha256=complete_sha,ready_sha256=sha(REPORT/'READY.json'),actual_optimizer_updates=153,
        actual_FIT_exposures=2099,new_generation_count=480,observed_model_seconds_sum=total,
        no_compression_or_pure_subspace_causality_claim=True,TEST_QA=0))


