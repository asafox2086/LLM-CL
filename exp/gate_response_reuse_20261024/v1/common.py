"""Frozen S1 selection, parent-proof binding and unique original-score consumer."""
import ast,hashlib,json,math,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).parent
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
BASE=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
REPORT=ROOT/'summary/final_weight_reduction_20261024/gate_response_reuse_v1'
WORK=HERE/'private'
SPLIT=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
MANIFEST=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
REFERENCE=ROOT/'parent_reference'
CASES=('init','Static','Reuse')
BUDGETS={'init':1200,'Static':2400,'Reuse':2400}
COUNTS={'Static':816728096,'Reuse':816706592}
SOURCES=('core.py','common.py','worker.py','freeze.py','dispatch.py')
CAP=335544320


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


def selection():
    m,s=metadata();task=m['tasks'][0];assert task['task']==1
    p=ROOT/task['files']['train']['path'];assert sha(p)==task['files']['train']['sha256']
    fit=set(s['fit_ids']['1']);rows={r['id']:r for r in map(json.loads,p.read_text().splitlines()) if r['id'] in fit}
    assert len(rows)==327
    groups=[]
    for gid,g in s['groups'].items():
        ids=[i for i in g['ids'] if i in rows]
        if ids:
            assert not g['dev']
            rep=min(ids,key=lambda i:h('42:gate-response-representative:'+i))
            groups.append(dict(group=gid,id=rep))
    assert len(groups)==325
    groups.sort(key=lambda x:h('42:gate-response-group:'+x['group']))
    chosen=[dict(**x,role='fit' if j<32 else 'check') for j,x in enumerate(groups[:48])]
    return task,rows,chosen


def dev_data():
    m,s=metadata();dev={}
    for task in m['tasks']:
        p=ROOT/task['files']['train']['path'];assert sha(p)==task['files']['train']['sha256']
        wanted=s['dev_ids'][str(task['task'])];ii=set(wanted)
        by={r['id']:r for r in map(json.loads,p.read_text().splitlines()) if r['id'] in ii}
        assert set(by)==ii;dev[task['task']]=[by[i] for i in wanted]
    assert sum(map(len,dev.values()))==100
    return m,s,dev


def parent_reference():
    p=REFERENCE/'VERIFIED.json'
    assert sha(p)=='da6715c564bfe96bdd0a3ef49bba2ebab01860d5b6a013aaaf634a2a35e3496f'
    proof=json.loads(p.read_bytes());comp=json.loads((REFERENCE/'init_COMPLETE.json').read_bytes())
    assert sha(REFERENCE/'init_COMPLETE.json')==proof['case_complete_sha256']['init']
    assert sha(REFERENCE/'initializations.json')==comp['files']['init/initializations.json']
    assert sha(REFERENCE/'parent.jsonl')==comp['files']['init/parent.jsonl']
    assert sha(REFERENCE/'READY.json')==proof['ready_sha256']
    init=json.loads((REFERENCE/'initializations.json').read_bytes())
    oldready=json.loads((REFERENCE/'READY.json').read_bytes())
    # Runtime/input files and original processor/config must match the earlier parent.
    for name,expected in oldready['files'].items():
        if name.startswith('exp/two_backbone_native_cl_20261017_v2/') or name.startswith('model/'):
            assert sha(ROOT/name)==expected,name
    predictions=[json.loads(x) for x in (REFERENCE/'parent.jsonl').read_text().splitlines()]
    assert len(predictions)==100 and len({x['id'] for x in predictions})==100
    return proof['rows']['init_parent'],{x['id']:x for x in predictions},init


def ready():
    r=json.loads((REPORT/'READY.json').read_bytes())
    assert r['uid']==os.getuid() and r['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for name,expected in r['files'].items():assert sha(ROOT/name)==expected,name
    return r


def verify():
    assert not (REPORT/'VERIFY_STARTED.json').exists() and not (REPORT/'VERIFIED.json').exists()
    for case in CASES:
        finished=json.loads((REPORT/(case+'_COMPLETE.json')).read_bytes())
        exited=json.loads((REPORT/(case+'_EXIT.json')).read_bytes())
        assert finished['complete'] and exited['returncode']==0 and exited['identity_after'] is None
    r=ready();write(REPORT/'VERIFY_STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    parent,parent_pred,parent_init=parent_reference()
    m,s,dev=dev_data();truth={row['id']:(task,row) for task in m['tasks'] for row in dev[task['task']]}
    from collections import Counter
    import re
    tree=ast.parse((PORT/'runtime_utils.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')]
    env=dict(Counter=Counter,re=re);exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-score-AST','exec'),env)
    total=0;gpu=None;quality={};complete_sha={}
    for case in CASES:
        c=json.loads((REPORT/(case+'_COMPLETE.json')).read_bytes());e=json.loads((REPORT/(case+'_EXIT.json')).read_bytes())
        start=json.loads((REPORT/(case+'_STARTED.json')).read_bytes())
        assert c['complete'] and e['returncode']==0 and e['identity_after'] is None
        assert e['worker']==c['worker'] and all(start[k]==c['worker'][k] for k in c['worker'])
        assert c['ready_sha256']==sha(REPORT/'READY.json')==start['ready_sha256']
        assert c['seconds']<=BUDGETS[case] and e['observed_elapsed_seconds']<=BUDGETS[case]
        total+=e['observed_elapsed_seconds'];gpu=start['GPU'] if gpu is None else gpu;assert start['GPU']==gpu
        required={'features.pt','initial_factors.pt','capture_index.json','parent_binding.json','resources.jsonl'} if case=='init' else {'updates.jsonl','predictions.jsonl','fit_summary.json','final_complete.pt','resources.jsonl'}
        assert set(c['files'])=={case+'/'+n for n in required}
        for name,expected in c['files'].items():assert sha(WORK/name)==expected,name
        resources=[json.loads(x) for x in (WORK/case/'resources.jsonl').read_text().splitlines()]
        assert resources and all(x['pid']==c['worker']['pid'] for x in resources)
        assert resources[0]['phase']=='before_model' and resources[-1]['phase']=='complete'
        complete_sha[case]=sha(REPORT/(case+'_COMPLETE.json'))
        if case=='init':
            assert c['parent_digest']==parent_init['parent_digest'] and c['capture_QA']==48 and c['payload_bytes']<=CAP
            index=json.loads((WORK/'init/capture_index.json').read_bytes())
            _,_,selected=selection();assert [x['selection'] for x in index['rows']]==selected
            assert all(0<len(x['positions'])<=64 and len(x['layers'])==24 for x in index['rows'])
            assert sum(x['payload_bytes'] for x in index['rows'])==c['payload_bytes']
            continue
        assert c['layout']['weights']['coefficients']==COUNTS[case] and c['layout']==c['reload_layout']
        assert c['final_digest']==c['reload_digest']==c['evaluation_digest_before']==c['evaluation_digest_after']
        assert c['frozen_before']==c['frozen_after'] and c['layout']['trainable']==0
        summary=json.loads((WORK/case/'fit_summary.json').read_bytes())
        assert summary['checkpoint']==c['checkpoint'] and summary['fit_states']==c['fit_states'] and summary['updates']==c['updates']
        assert [x['layer'] for x in summary['layers']]==list(range(24)) and all(x['parameters_changed'] for x in summary['layers'])
        assert c['checkpoint']['sha256']==c['files'][case+'/final_complete.pt'] and c['checkpoint']['bytes']==(WORK/case/'final_complete.pt').stat().st_size
        updates=[json.loads(x) for x in (WORK/case/'updates.jsonl').read_text().splitlines()]
        assert len(updates)==c['updates']==24*4*math.ceil(c['fit_states']/128)
        for i in range(24):
            uu=[x for x in updates if x['layer']==i];assert len(uu)==4*math.ceil(c['fit_states']/128)
            assert [(x['epoch'],x['offset'],x['states']) for x in uu]==[(ep,st,min(128,c['fit_states']-st)) for ep in range(1,5) for st in range(0,c['fit_states'],128)]
            assert all(x['finite'] and x['positive_lr'] and x['actual_step'] and x['gradient_bytes']==4*x['trainable'] and x['cpu_moments_bytes']==8*x['trainable'] for x in uu)
            expected=200*5760 if case=='Static' else 199*5760+4864
            assert all(x['trainable']==expected and len(x['parameter_ids'])==(2 if case=='Static' else 3) for x in uu)
        rows=[json.loads(x) for x in (WORK/case/'predictions.jsonl').read_text().splitlines()]
        order=[row['id'] for task in m['tasks'] for row in dev[task['task']]]
        assert [x['id'] for x in rows]==order
        per=[];domains={}
        for row in rows:
            task,original=truth[row['id']];old=parent_pred[row['id']]
            assert row['references']==old['references']==original['answers']
            assert row['input_sha256']==old['input_sha256'] and row['generation_cap']==old['generation_cap']==task['max_new_tokens']
            assert row['eos_token_id']==old['eos_token_id'] and row['output_token_ids'] and row['finite_score_checks']>0
            assert row['images']=={n:s['image_sha256'][n] for n in original['images']}
            value=env['score'](row['prediction'],original['answers'])['token_f1']
            per.append(dict(id=row['id'],F1=value,domain=task['domain']))
            domains.setdefault(task['domain'],[]).append(value)
        assert len(domains)==5 and all(len(v)==20 for v in domains.values())
        ds={k:sum(v)/len(v) for k,v in domains.items()};quality[case]=dict(domains=ds,macro=sum(ds.values())/5,per_id=per)
    assert total<=6000
    point=quality['Reuse']['macro']>quality['Static']['macro'] and all(quality['Reuse']['domains'][d]>=v for d,v in parent['domains'].items())
    write(REPORT/'VERIFIED.json',dict(complete=True,quality=quality,parent_copied_verified_scalars=parent,
        parent_VERIFIED_sha256=sha(REFERENCE/'VERIFIED.json'),parent_state_digest=parent_init['parent_digest'],
        primary_point_pass=point,case_complete_sha256=complete_sha,ready_sha256=sha(REPORT/'READY.json'),
        observed_model_seconds_sum=total,TEST_QA=0,not_continual_learning_or_novelty_proof=True))
