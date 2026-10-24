"""Static contracts and the only result scorer for the fixed carrier."""
import ast, hashlib, json, math, os, re, sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).parent
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/final_weight_reduction_20261024/shared_fullspan_carrier_v1'
WORK=HERE/'private'
BASE=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
SCREEN=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'
BUDGET=3600
CACHE_LIMIT=8*1024**3
SOURCES=('core.py','common.py','worker.py','freeze.py','dispatch.py')
ORIGINAL_SPLIT_SHA='d8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
ORIGINAL_MANIFEST_SHA='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
ORIGINAL_BASE_PROOF_SHA='467ad42dfdcb755576962fd0acac2c622a075e7c7fef3dc7b6566c74351142c5'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        uid=int(next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')))
        return dict(pid=pid,startticks=s[19],uid=uid,pgid=int(s[2]))
    except FileNotFoundError:return None

def data():
    policy=json.loads((PORT/'execution_policy.json').read_bytes())
    mp=ROOT/policy['conditions']['pct01']['manifest'];assert sha(mp)==ORIGINAL_MANIFEST_SHA
    m=json.loads(mp.read_bytes());task=m['tasks'][0]
    assert task['task']==1 and task['domain']=='natural' and task['max_new_tokens']==128
    sp=SCREEN/'SPLIT.json';assert sha(sp)==ORIGINAL_SPLIT_SHA
    split=json.loads(sp.read_bytes());fit=set(split['fit_ids']['1'])
    assert len(fit)==327 and not fit.intersection(split['dev_ids']['1'])
    tp=ROOT/task['files']['train']['path'];assert sha(tp)==task['files']['train']['sha256']
    rows={}
    for line in tp.read_bytes().splitlines():
        row=json.loads(line)
        if row['id'] in fit:
            assert row['id'] not in rows;rows[row['id']]=row
    assert set(rows)==fit
    grouped={}
    for gid,g in split['groups'].items():
        ids=sorted(set(g['ids']).intersection(fit))
        if ids:grouped[gid]=ids
    assert len(grouped)==325 and set(i for v in grouped.values() for i in v)==fit
    order=sorted(grouped,key=lambda g:hashlib.sha256(('shared-fullspan-v1|'+g).encode()).digest())
    chosen={'diagnostic':order[:65],'recovery':order[65:97]}
    plan=[]
    for role,gids in chosen.items():
        for gid in gids:
            row=rows[grouped[gid][0]]
            assert all(sha(ROOT/n)==split['image_sha256'][n] for n in row['images'])
            binding={k:row.get(k) for k in ('id','question','context','answer','answers','images')}
            plan.append(dict(role=role,group=gid,id=row['id'],
                QA_binding_sha256=hashlib.sha256(json.dumps(binding,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                images=[dict(path=n,sha256=sha(ROOT/n)) for n in row['images']]))
    return task,rows,dict(manifest_path=str(mp.relative_to(ROOT)),manifest_sha256=sha(mp),
        split_sha256=sha(sp),S1_train_path=str(tp.relative_to(ROOT)),S1_train_sha256=sha(tp),rows=plan)

def loss_node():
    import copy
    tree=ast.parse((PORT/'runtime_utils.py').read_text());old=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='answer_loss')
    first=old.body[0];call=first.value.value
    assert {k.arg:ast.literal_eval(k.value) for k in call.keywords}==dict(add_special_tokens=False,truncation=True,max_length=512)
    new=copy.deepcopy(old);new.name='full_answer_loss'
    new.body[0].value.value.keywords=[ast.keyword(arg='add_special_tokens',value=ast.Constant(False)),ast.keyword(arg='truncation',value=ast.Constant(False))]
    back=copy.deepcopy(new);back.name=old.name;back.body[0]=copy.deepcopy(first)
    assert ast.dump(back,include_attributes=False)==ast.dump(old,include_attributes=False)
    return ast.fix_missing_locations(new),hashlib.sha256(ast.dump(new,include_attributes=False).encode()).hexdigest()

def ready():
    r=json.loads((REPORT/'READY.json').read_bytes())
    for key in ('sources','evidence','dependencies','environment','base_small_files'):
        for n,h in r[key].items():assert sha(ROOT/n if not Path(n).is_absolute() else Path(n))==h,n
    assert str(Path(sys.executable).resolve())==r['interpreter'] and loss_node()[1]==r['full_loss_AST_sha256']
    return r

def verify():
    assert not (REPORT/'VERIFY_STARTED.json').exists() and not (REPORT/'VERIFIED.json').exists()
    r=ready();done=json.loads((REPORT/'COMPLETE.json').read_bytes());exit_record=json.loads((REPORT/'ACTUAL_EXIT.json').read_bytes())
    started=json.loads((REPORT/'STARTED.json').read_bytes())
    who={k:started[k] for k in ('pid','startticks','uid','pgid')}
    assert who==exit_record['worker']==done['end_identity'] and who['uid']==r['execution_uid'] and who['pid']==who['pgid']
    assert started['boot_id']==r['boot_id'] and started['ready_sha256']==sha(REPORT/'READY.json')
    assert math.isfinite(started['started_monotonic']) and started['GPU'] in r['allowed_GPUs']
    assert exit_record['returncode']==0 and exit_record['observed_elapsed_seconds']<=BUDGET and exit_record['identity_after'] is None
    assert done['model_RUN_seconds']<=BUDGET and done['ready_sha256']==sha(REPORT/'READY.json')
    assert not (REPORT/'FAILURE.json').exists()
    write(REPORT/'VERIFY_STARTED.json',dict(complete_sha256=sha(REPORT/'COMPLETE.json'),once_only=True))
    task,rows,plan=data();assert plan==r['data']
    required={str((WORK/n).relative_to(ROOT)) for n in ('parent_diagnostic.jsonl','compact_diagnostic.jsonl','updates.jsonl','cache_index.json','resources.jsonl')}
    assert set(done['files'])==required
    for n,h in done['files'].items():assert sha(ROOT/n)==h,n
    for key in ('parent_package','compact_package'):
        package=done[key];assert (ROOT/package['path']).stat().st_size==package['bytes']
        assert isinstance(package['sha256'],str) and len(package['sha256'])==64
    resources=[json.loads(s) for s in (WORK/'resources.jsonl').read_text().splitlines()]
    assert resources and all(e['pid']==who['pid'] and e['visible_gpu']==str(started['GPU']) for e in resources)
    assert {'parent_before','cache_complete','group_packed','complete'} <= {e['phase'] for e in resources}
    assert len([e for e in resources if e['phase']=='group_packed'])==6
    journal=[json.loads(s) for s in (WORK/'updates.jsonl').read_text().splitlines()]
    assert len(journal)==7680
    for g in range(6):
        events=[e for e in journal if e['group']==g]
        assert [e['step'] for e in events]==list(range(1,1281))
        assert all(e['actual_optimizer_steps']==e['step'] and math.isfinite(e['loss']) and e['lr']>0 for e in events)
        assert all(len(e['optimizer_parameter_ids'])==13 and len(set(e['optimizer_parameter_ids']))==13 for e in events)
        assert all(e['trainable_coordinates']==68272128 and e['gradient_bytes']==273088512 and e['moment_bytes']==546177024 for e in events)
        assert next(e for e in events if e['step']==960)['mask_event']['fraction']==.75
        mask_events=[e for e in events if e['mask_event'] is not None]
        expected_steps=sorted(set(range(50,1281,50))|{960})
        assert [e['step'] for e in mask_events]==expected_steps
        for e in mask_events:
            fraction=.75*(1-(1-min(e['step']/960,1))**3)
            assert e['mask_event']['fraction']==fraction
            assert e['mask_event']['kept']==[math.ceil(1152*4864*(1-fraction))]*12
    assert done['cache_QA']==32 and done['cache_layer_pairs']==768 and done['cache_payload_bytes']<=CACHE_LIMIT
    assert done['parent_digest_before']==done['parent_digest_after']
    assert done['non_FFN_before']==done['non_FFN_after'] and done['packed_digest_before']==done['packed_digest_after']==done['reload_digest']
    assert done['packed_layout']==done['reload_layout']
    original=done['parent_layout'];compact=done['packed_layout']
    assert compact['factor_layers']==72 and compact['shared_bases']==6 and compact['dense_factors']==0
    assert compact['bitmap_bytes']==50429952 and compact['packed_values']==100859904
    assert compact['shared_basis_values']==6193152 and compact['weight_values']==686942240
    assert compact['trainable']==0
    tree=ast.parse((PORT/'runtime_utils.py').read_text());ns=dict(re=re,Counter=Counter)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('score','tokens')],type_ignores=[]),'<original-AnswerF1>','exec'),ns)
    score=ns['score'];quality={};public=[];diagnostic=[p for p in plan['rows'] if p['role']=='diagnostic']
    input_bindings=None
    for state in ('parent','compact'):
        predictions=[json.loads(s) for s in (WORK/(state+'_diagnostic.jsonl')).read_text().splitlines()]
        assert [p['id'] for p in predictions]==[p['id'] for p in diagnostic]
        assert all(p['target_tokens']>0 and math.isfinite(p['NLL']) and p['generation_cap']==128 and 0<p['output_tokens']<=128 for p in predictions)
        bindings=[p['inputs'] for p in predictions]
        if input_bindings is None:input_bindings=bindings
        else:assert bindings==input_bindings
        scores=[score(p['prediction'],rows[p['id']]['answers'])['token_f1'] for p in predictions]
        quality[state]=dict(S1_AnswerF1=sum(scores)/65,S1_NLL=sum(p['NLL'] for p in predictions)/65,QA=65)
        public.extend(dict(state=state,id=p['id'],group=q['group'],AnswerF1=s,NLL=p['NLL'],target_tokens=p['target_tokens'],output_tokens=p['output_tokens'],QA_binding_sha256=q['QA_binding_sha256']) for p,q,s in zip(predictions,diagnostic,scores))
    output=REPORT/'SCORES.jsonl'
    with output.open('x',encoding='utf8') as f:
        for row in public:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())
    storage=(compact['weight_values']<=.8*original['weight_values'] and done['compact_package']['bytes']<=.8*done['parent_package']['bytes'])
    passed=storage and quality['compact']['S1_AnswerF1']>=quality['parent']['S1_AnswerF1'] and quality['compact']['S1_NLL']<=quality['parent']['S1_NLL']
    proof=dict(quality=quality,storage_gate=storage,investment_gate_passed=passed,layout=compact,
        parent_package=done['parent_package'],compact_package=done['compact_package'],
        score_AST_sha256=hashlib.sha256(ast.dump(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('score','tokens')],type_ignores=[]),include_attributes=False).encode()).hexdigest(),
        score_requests=130,NLL_requests=162,generation_requests=130,recovery_group_updates=7680,
        complete_sha256=sha(REPORT/'COMPLETE.json'),ready_sha256=sha(REPORT/'READY.json'),scores_sha256=sha(output),
        next_action='necessity_review_only' if passed else 'close_fixed_carrier',not_framework_or_all_domain_preservation=True,
        no_automatic_continual_training=True,TEST_QA=0)
    write(REPORT/'VERIFIED.json',proof)
    return proof
