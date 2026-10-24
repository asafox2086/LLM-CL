"""Bounded four-state image-support diagnostic; imports do no model work."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).parent
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
SCREEN=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'
REPORT=ROOT/'summary/final_weight_reduction_20261024/paired_image_support_diagnostic_v1'
WORK=HERE/'private'
STATES=('qv_before','qv_after','osft_before','osft_after')
DOMAINS=('natural','medical','geography','finance','traffic')
BUDGET=1800
TOTAL_BUDGET=7200
PHYSICAL_GPU=0
BASE=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'

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
    if os.name!='nt':
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)

def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        uid=int(next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')))
        return dict(pid=pid,startticks=s[19],uid=uid,pgid=int(s[2]))
    except FileNotFoundError:return None

def inventory():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    cards={}
    for line in text.splitlines():
        if not line.strip():continue
        index,uuid,memory,util=map(str.strip,line.split(','))
        cards[int(index)]=dict(uuid=uuid,memory_MiB=int(memory),utilization=int(util),owners=[])
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    for line in text.splitlines():
        if not line.strip():continue
        pid,uuid=map(str.strip,line.split(','));who=identity(int(pid))
        if who is None:continue
        match=[c for c in cards.values() if c['uuid']==uuid];assert len(match)==1
        match[0]['owners'].append(who)
    return cards

def available():
    cards=inventory();ours={g for g,c in cards.items() if any(p['uid']==os.getuid() for p in c['owners'])}
    free={g for g,c in cards.items() if not c['owners'] and c['memory_MiB']<500 and c['utilization']<10}
    return PHYSICAL_GPU in free,cards

def pairing():
    p=json.loads((REPORT/'PAIRING.json').read_bytes())
    assert p['coverage_investment_gate_passed'] and p['paired_QA']==61 and not p['model_admitted']
    rows=[r for r in p['rows'] if r['donor_id'] is not None]
    assert len(rows)==61 and Counter(r['id'] for r in rows)==Counter(r['donor_id'] for r in rows)
    return p,rows

def metadata():
    p,rows=pairing();policy=json.loads((PORT/'execution_policy.json').read_bytes())
    path=ROOT/policy['conditions']['pct01']['manifest'];assert sha(path)==p['manifest_sha256']
    assert sha(SCREEN/'SPLIT.json')==p['split_sha256']
    return json.loads(path.read_bytes()),p,rows

def data():
    manifest,p,pairs=metadata();wanted={r['id'] for r in pairs};selected={}
    for t in manifest['tasks']:
        source=ROOT/t['files']['train']['path']
        if not any(r['task']==t['task'] for r in pairs):continue
        assert sha(source)==t['files']['train']['sha256']
        for line in source.read_bytes().splitlines():
            r=json.loads(line)
            if r['id'] in wanted:
                assert r['id'] not in selected;selected[r['id']]=r
    assert set(selected)==wanted
    for record in pairs:
        r=selected[record['id']]
        assert [sha(ROOT/n) for n in r['images']]==record['image_sha256']
    return manifest,p,pairs,selected

def loss_contract():
    run_tree=ast.parse((PORT/'run.py').read_text())
    assert any(isinstance(n,ast.ImportFrom) and n.module=='runtime_utils' and any(a.name=='answer_loss' and a.asname is None for a in n.names) for n in run_tree.body)
    tree=ast.parse((PORT/'runtime_utils.py').read_text())
    old=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='answer_loss')
    first=old.body[0];assert isinstance(first,ast.Assign) and isinstance(first.value,ast.Subscript)
    call=first.value.value;assert isinstance(call,ast.Call)
    assert {k.arg:ast.literal_eval(k.value) for k in call.keywords}==dict(add_special_tokens=False,truncation=True,max_length=512)
    import copy
    new=copy.deepcopy(old);new.name='full_answer_loss'
    new.body[0].value.value.keywords=[ast.keyword(arg='add_special_tokens',value=ast.Constant(False)),ast.keyword(arg='truncation',value=ast.Constant(False))]
    back=copy.deepcopy(new);back.name=old.name;back.body[0]=copy.deepcopy(first)
    assert ast.dump(back,include_attributes=False)==ast.dump(old,include_attributes=False)
    digest=hashlib.sha256(ast.dump(new,include_attributes=False).encode()).hexdigest()
    return ast.fix_missing_locations(new),digest

def freeze():
    assert 'torch' not in sys.modules and not (REPORT/'MODEL_READY.json').exists()
    review=json.loads((REPORT/'MODEL_SOURCE_REVIEW.json').read_bytes())
    assert review['decision']=='accept_four_state_diagnostic_source_only'
    assert review['protocol_sha256']==sha(REPORT/'MODEL_PROTOCOL.md')
    assert review['necessity_review_sha256']==sha(REPORT/'MODEL_NECESSITY_REVIEW.json')
    necessary=json.loads((REPORT/'MODEL_NECESSITY_REVIEW.json').read_bytes())
    assert necessary['model_protocol_sha256']==review['protocol_sha256']
    sources={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in ('common.py','worker.py','dispatch.py')}
    assert review['sources']==sources
    old=json.loads((SCREEN/'READY.json').read_bytes());done=json.loads((SCREEN/'SCREEN_COMPLETE.json').read_bytes())
    manifest,p,pairs=metadata();checkpoints={}
    for case in ('LoRA_qv','OSFT_all'):
        c=done['all_four_results'][case];path=ROOT/c['checkpoint_root']/c['checkpoint']['path']
        complete_path=SCREEN/(case+'_COMPLETE.json');proof_path=SCREEN/'sealed_case_reports_20261022'/(case+'_VERIFIED.json')
        proof=json.loads(proof_path.read_bytes());assert c==json.loads(complete_path.read_bytes())
        assert proof['case']==case and proof['complete_marker_sha256']==sha(complete_path) and proof['ready_sha256']==sha(SCREEN/'READY.json')
        assert proof['checkpoint']==c['checkpoint'] and proof['original_independent_verifier_passed']
        assert proof['FIT_QA']==1213 and proof['DEV_QA']==100 and proof['TEST_QA']==0 and proof['positive_finite_updates']==87
        assert json.loads(path.with_suffix('.json').read_bytes())==c['checkpoint']
        assert path.stat().st_size==c['checkpoint']['bytes'] and c['checkpoint']['stage']==19
        checkpoints[case]=dict(path=str(path.relative_to(ROOT)),metadata=c['checkpoint'],
            original_proof_path=str(proof_path.relative_to(ROOT)),original_proof_sha256=sha(proof_path))
    dependencies=[PORT/n for n in ('run.py','runtime_utils.py','retrieval.py','onevision_bridge.py','frozen_vision.py','methods.py','osft_adapter.py','resource_monitor.py','training_validity.py','checkpoints.py','execution_policy.json')]
    dependencies += [ROOT/'exp/osft_increment_screen_20261019/v1/study.py',SCREEN/'READY.json',SCREEN/'SCREEN_COMPLETE.json',SCREEN/'SPLIT.json']
    dependencies += [SCREEN/(c+'_COMPLETE.json') for c in ('LoRA_qv','OSFT_all')]
    dependencies += [SCREEN/'sealed_case_reports_20261022'/(c+'_VERIFIED.json') for c in ('LoRA_qv','OSFT_all')]
    for path in dependencies:
        name=str(path.relative_to(ROOT))
        if name in old['sources']:assert sha(path)==old['sources'][name],name
    for name in ('code/OSFT/source/src/mini_trainer/osft_utils.py','code/OSFT/files_sha256.json','code/OSFT/source_manifest.json'):
        assert sha(ROOT/name)==old['sources'][name];dependencies.append(ROOT/name)
    import importlib.metadata
    files=('torch/__init__.py','transformers/modeling_utils.py','transformers/models/qwen2/modeling_qwen2.py',
        'transformers/models/llava_onevision/modeling_llava_onevision.py','transformers/models/llava_onevision/processing_llava_onevision.py',
        'transformers/models/llava_onevision/image_processing_llava_onevision.py','transformers/models/siglip/modeling_siglip.py')
    environment={}
    for name in files:
        package,inside=name.split('/',1);spec=importlib.util.find_spec(package)
        assert spec and spec.submodule_search_locations and len(spec.submodule_search_locations)==1
        path=Path(next(iter(spec.submodule_search_locations)))/inside;assert path.is_file();environment[str(path.resolve())]=sha(path)
    small={str(x.relative_to(ROOT)):sha(x) for x in BASE.iterdir() if x.is_file() and x.stat().st_size<8*1024**2}
    evidence={str((REPORT/n).relative_to(ROOT)):sha(REPORT/n) for n in ('PAIRING.json','PROTOCOL.md','RESULT_REVIEW.json','MODEL_PROTOCOL.md','MODEL_NECESSITY_REVIEW.json','MODEL_SOURCE_REVIEW.json','HOST_SELECTION_RECORD.json','MODEL_TRANSPORT.json')}
    transport=json.loads((REPORT/'MODEL_TRANSPORT.json').read_bytes())
    assert transport['execution_root']==str(ROOT) and transport['destination_host']=='OLD' and transport['all_destination_sha256_matched']
    assert ROOT==Path('/data2/liyapeng_grp/program/LLMCL/.deployment/paired_image_support_20261024_view')
    ready=dict(kind='paired_image_support_four_states_v1',states=STATES,budget_seconds=BUDGET,total_budget_seconds=TOTAL_BUDGET,
        QA=61,NLL_requests=488,sources=sources,evidence=evidence,dependencies={str(x.relative_to(ROOT)):sha(x) for x in dependencies},
        environment=environment,base_small_files=small,base_weights=old['base_files'],checkpoints=checkpoints,
        interpreter=str(Path(sys.executable).resolve()),versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','cloudpickle')},
        full_loss_AST_sha256=loss_contract()[1],metadata_only=True,training_updates=0,generation_requests=0,TEST_QA=0,
        execution_host='OLD',execution_root=str(ROOT),execution_uid=os.getuid(),physical_GPU=PHYSICAL_GPU,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    assert 'torch' not in sys.modules;write(REPORT/'MODEL_READY.json',ready)

def check_ready():
    r=json.loads((REPORT/'MODEL_READY.json').read_bytes())
    assert r['kind']=='paired_image_support_four_states_v1' and r['states']==list(STATES)
    assert r['budget_seconds']==BUDGET and r['total_budget_seconds']==TOTAL_BUDGET and r['NLL_requests']==488
    assert r['execution_host']=='OLD' and r['execution_root']==str(ROOT) and r['physical_GPU']==PHYSICAL_GPU and r['execution_uid']==os.getuid()
    assert str(Path(sys.executable).resolve())==r['interpreter']
    for field in ('sources','evidence','dependencies','base_small_files'):
        for n,h in r[field].items():assert sha(ROOT/n)==h,n
    for n,h in r['environment'].items():assert sha(n)==h,n
    assert loss_contract()[1]==r['full_loss_AST_sha256']
    return r

def verify():
    import math,time
    from statistics import fmean
    begin=time.perf_counter();r=check_ready()
    assert not (REPORT/'MODEL_VERIFY_STARTED.json').exists() and not (REPORT/'MODEL_VERIFIED.json').exists()
    write(REPORT/'MODEL_VERIFY_STARTED.json',dict(pid=os.getpid(),ready_sha256=sha(REPORT/'MODEL_READY.json')))
    _,pairs=pairing();ids=[x['id'] for x in pairs];all_rows={};complete_files={};total=0
    for state in STATES:
        path=REPORT/(state+'_COMPLETE.json');c=json.loads(path.read_bytes())
        exit_proof=json.loads((REPORT/(state+'_ACTUAL_EXIT.json')).read_bytes())
        assert c['state']==state and c['ready_sha256']==sha(REPORT/'MODEL_READY.json')
        assert c['NLL_requests']==122 and c['QA']==61 and c['training_updates']==c['generation_requests']==c['TEST_QA']==0
        assert c['weight_digest_before']==c['weight_digest_after']
        assert 0<c['model_RUN_seconds']<=exit_proof['observed_elapsed_bound_seconds']<=BUDGET and exit_proof['code']==0
        assert not (REPORT/(state+'_FAILURE.json')).exists()
        started=json.loads((REPORT/(state+'_STARTED.json')).read_bytes())
        assert all(started[k]==v for k,v in exit_proof['identity'].items())
        assert started['ready_sha256']==c['ready_sha256'] and started['budget_seconds']==BUDGET
        for key,digest_key in (('measurement_path','measurement_sha256'),('initial_binding_path','initial_binding_sha256'),('resources_path','resources_sha256')):
            assert sha(ROOT/c[key])==c[digest_key]
        rr=[json.loads(x) for x in (ROOT/c['measurement_path']).read_text().splitlines()]
        assert [x['id'] for x in rr]==ids and len(rr)==61
        for x,pair in zip(rr,pairs):
            assert all(x[k]==pair[k] for k in ('id','task','domain','group','donor_id'))
            assert all(math.isfinite(x[k]) for k in ('original_NLL','wrong_NLL','C')) and x['target_tokens']>1
            assert x['original_NLL']>=0 and x['wrong_NLL']>=0 and x['C']==x['wrong_NLL']-x['original_NLL']
            assert x['original_image_sha256']==pair['image_sha256']
            assert x['wrong_image_sha256']==next(v['image_sha256'] for v in pairs if v['id']==pair['donor_id'])
            assert x['features'][0]['cache_key']!=x['features'][1]['cache_key']
            assert x['features'][0]['feature_shape']==x['features'][1]['feature_shape']
        resources=[json.loads(x) for x in (ROOT/c['resources_path']).read_text().splitlines()]
        assert resources and all(x['pid']==started['pid'] and x['visible_gpu']==str(PHYSICAL_GPU) for x in resources)
        all_rows[state]={x['id']:x for x in rr};complete_files[str(path.relative_to(ROOT))]=sha(path)
        total+=exit_proof['observed_elapsed_bound_seconds']
    assert total<=TOTAL_BUDGET
    results=[]
    for qid in ids:
        four={state:all_rows[state][qid] for state in STATES};anchor=four[STATES[0]]
        for x in four.values():
            for key in ('target_sha256','target_tokens','QA_binding_sha256','nonvisual_input_sha256','original_image_sha256','wrong_image_sha256'):
                assert x[key]==anchor[key],(qid,key)
            for i in range(2):
                for key in ('feature_sha256','feature_shape','image_paths'):
                    assert x['features'][i][key]==anchor['features'][i][key],(qid,i,key)
        item=dict(id=qid,domain=anchor['domain'],task=anchor['task'],group=anchor['group'],donor_id=anchor['donor_id'])
        for prefix in ('qv','osft'):
            a=four[prefix+'_before'];b=four[prefix+'_after']
            item[prefix+'_I_original']=a['original_NLL']-b['original_NLL']
            item[prefix+'_I_wrong']=a['wrong_NLL']-b['wrong_NLL']
            item[prefix+'_DeltaC']=b['C']-a['C']
        item['D']=item['osft_DeltaC']-item['qv_DeltaC']
        item['relative_I_original']=item['osft_I_original']-item['qv_I_original']
        item['state_NLL']={s:{k:x[k] for k in ('original_NLL','wrong_NLL','C')} for s,x in four.items()}
        results.append(item)
    keys=('qv_I_original','qv_I_wrong','qv_DeltaC','osft_I_original','osft_I_wrong','osft_DeltaC','D','relative_I_original')
    def summary(rows):return dict(QA=len(rows),known_groups=len({x['group'] for x in rows}),**{k:fmean(x[k] for x in rows) for k in keys})
    domains={d:summary([x for x in results if x['domain']==d]) for d in DOMAINS}
    tasks={str(t):summary([x for x in results if x['task']==t]) for t in sorted({x['task'] for x in results})}
    macro={k:fmean(domains[d][k] for d in DOMAINS) for k in keys}
    directions=dict(original_support_improves=macro['osft_I_original']>0,OSFT_pair_support_increases=macro['osft_DeltaC']>0,
        relative_pair_support_increase=macro['D']>0)
    write(REPORT/'MODEL_VERIFIED.json',dict(ready_sha256=sha(REPORT/'MODEL_READY.json'),complete_files_sha256=complete_files,
        total_NLL_requests=488,total_model_observed_exit_bound_seconds=total,QA=61,domains=domains,tasks=tasks,
        five_domain_equal_macro=macro,candidate_direction_checks=directions,all_candidate_directions=all(directions.values()),
        results=results,new_F1_scores=0,new_bootstrap=0,training_updates=0,TEST_QA=0,
        interpretation='Geometry-conditioned explored DEV; conditional gold-support trajectory, not knowledge, parameter causality, generation F1 or compression proof.',
        no_automatic_next_model=True,verify_seconds=time.perf_counter()-begin))

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('freeze','verify')
    (freeze if sys.argv[1]=='freeze' else verify)()
