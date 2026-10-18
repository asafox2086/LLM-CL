"""Native small-data controls: cold, full inheritance, and A-only inheritance.

Diagnostic experiment, not an admitted new algorithm or a formal main-table run.
All conditions use the same 26 target TRAIN QA and full507 final TEST.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import traceback
import torch

import onevision_learned_reuse_trial as parent
from onevision_knowledge_intervention import reset_seed,case_complete
from onevision_research_worker import atomic_torch,digest
from onevision_capacity_probe import finish_resources
from run import hashed,write

ROOT=parent.ROOT
REPORT=parent.BASE/'initialization_source_controls_20261018/v1'
FAMILY=ROOT/'exp/onevision_initialization_source_20261018/v1'
CASES=['cold','within_full','cross_full','within_A_only','cross_A_only']


def same(a,b):return a.keys()==b.keys() and all(torch.equal(a[k][v],b[k][v]) for k in a for v in ['A','B'])


def initialize(initial,source,name):
    assert name in CASES
    start=copy.deepcopy(initial if name=='cold' else source)
    if name.endswith('A_only'):
        for k in start:start[k]['B']=torch.zeros_like(start[k]['B'])
    return start


def plan():
    original,values=parent.plan()
    manifest,data,domains,tests,full,native,proof=values
    target=data[4,'train'];target_images={p for r in target for p in r['images']}
    within=[r for r in data[3,'train'] if r.get('metadata',{}).get('q_lang')=='en' and not(target_images&set(r['images']))]
    within=sorted(within,key=lambda r:hashed(r,'source-initialization-within'))[:6]
    cross=sorted(data[12,'train'],key=lambda r:hashed(r,'source-initialization-cross'))
    assert len(within)==len(cross)==6 and len(target)==26 and len(data[4,'test'])==507
    # Do not label missing author metadata as English. Verify this literal
    # English template for this source, preserving the missing original field.
    assert all(r['question']=='Is it a rural or an urban area' for r in cross)
    assert all(not(target_images&set(r['images'])) for r in within+cross)
    probe=sorted(data[4,'test'],key=lambda r:hashed(r,'source-initialization-probe'))[:32]
    sources=dict(original['source_sha256']);sources['analysis/'+Path(__file__).name]=digest(Path(__file__))
    spec=dict(source_sha256=sources,manifest_sha256=original['manifest_sha256'],
        native_parent_sha256=original['native_completion_sha256'],native_initial_source=str(native/'initial_adapter.pt'),
        seed=42,rank=8,alpha=32,dropout=.1,batch=16,target_task=4,target_TRAIN_ids=[r['id'] for r in target],
        target_TRAIN_QA=26,target_updates=2,target_epochs=1,target_full_TEST_QA=507,
        sources={'within':[r['id'] for r in within],'cross':[r['id'] for r in cross]},
        source_tasks={'within':3,'cross':12},source_unique_QA=6,source_rounds=2,
        source_QA_exposures=12,source_updates=2,source_optimizer='fresh original Adam per one-epoch6QA round',
        source_language='within author en; cross literal English template verified, author field remains missing',
        source_target_TRAIN_images_disjoint=True,cases=CASES,
        initial_probe_ids=[r['id'] for r in probe],initial_probe_QA=32,
        nativeAnyRes_BF16_not_formalV2=True,TEST_answers_for_selection=False,
        source_selection='fixed TRAIN hash and language/TRAIN-image controls, no TEST score selection',
        purpose='test source dependence and old function vs learned A basis; not assume cross-domain reuse helps',
        all5_final507_regardless_initial_probe=True)
    return spec,(manifest,data,native,proof,within,cross,probe)


def complete(phase,index,spec):
    name='sources' if phase=='sources' else CASES[index]
    p=REPORT/(name+'_complete.json')
    if not p.exists():return False
    obj=json.loads(p.read_text());assert obj['complete'] and obj['spec']==spec
    run=ROOT/obj['run'];assert run.resolve().is_relative_to(FAMILY.resolve())
    for rel,s in obj['artifact_sha256'].items():
        q=run/rel;assert q.resolve().is_relative_to(run.resolve()) and digest(q)==s
    if phase=='case':
        assert obj['target_full_TEST_QA']==507 and json.loads((run/'initial_function_gate.json').read_text())['passed']
    return True


def semantic(path):
    return [{k:r[k] for k in ['id','nll','prediction','references','scores']}
            for r in [json.loads(s) for s in path.read_text().splitlines()]]


def fit(x,run,label,rows):
    x.cache.clear();reset_seed();x.model.train();torch.cuda.reset_peak_memory_stats()
    before=x.monitor.snapshot('pure_TRAIN_'+label+'_before',model=x.model)
    result=x.train(label,rows)
    after=x.monitor.snapshot('pure_TRAIN_'+label+'_after',model=x.model)
    p=run/'pure_TRAIN_windows.json';records=json.loads(p.read_text())['windows'] if p.exists() else []
    records.append(dict(label=label,before=before,after=after,
        cuda_peak_allocated_bytes=after['gpu_peak_allocated_bytes'],
        cuda_peak_reserved_bytes=after['gpu_peak_reserved_bytes']))
    write(p,dict(windows=records,scope='TRAIN plus raw-gradient/checkpoint persistence; no TEST or model admission'))
    return result


def run_trial(phase,index):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0','1']
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    spec,values=plan();assert json.loads((REPORT/'CPU_PLAN_CHECK.json').read_text())['spec']==spec
    admission=json.loads((REPORT/'ADMISSION.json').read_text())
    assert admission['spec']==spec and admission['single_research_queue']
    lock=(REPORT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert phase=='sources' or 0<=index<len(CASES)
    name='sources' if phase=='sources' else CASES[index]
    assert not (REPORT/(name+'_attempt.json')).exists(), 'No blind retry of incomplete controls'
    if phase=='case':assert complete('sources',None,spec)
    manifest,data,native,native_proof,within,cross,probe=values
    import time
    run=FAMILY/name/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
    write(REPORT/(name+'_attempt.json'),dict(spec=spec,run=str(run.relative_to(ROOT)),pid=os.getpid()))
    write(run/'spec.json',spec)
    import onevision_research_device_entry as device
    cards=device.inventory();gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert device.select(cards,os.getuid())==gpu
    write(run/'physical_device_execution.json',dict(physical_GPU=gpu,actual_CUDA_VISIBLE_DEVICES=str(gpu),
        admission_inventory=cards,max_our_GPU_cards=3,single_research_stream=True))
    x=parent.MetricExperiment(manifest,run)
    expected=native/'initial_adapter.pt'
    assert digest(expected)==native_proof['artifact_sha256']['initial_adapter.pt']
    initial=torch.load(expected,map_location='cpu',weights_only=True)['adapter']
    assert same(x.original_initial,initial)
    x.monitor.snapshot('control_before',model=x.model)
    if phase=='sources':
        atomic_torch(run/'original_initial.pt',dict(adapter=initial))
        profiles=[]
        for source,rows in [('within',within),('cross',cross)]:
            current=copy.deepcopy(initial)
            for round_ in range(1,3):
                x.configure(current,finetune_shared=True)
                current=fit(x,run,f'{source}_round{round_}',rows)
            changed_A=sum(not torch.equal(current[k]['A'],initial[k]['A']) for k in initial)
            assert changed_A>0, 'A-only intervention must actually change a learned basis'
            atomic_torch(run/(source+'.pt'),dict(adapter=current))
            profiles.append(dict(source=source,changed_A_layers=changed_A,
                A_change_l2=math.sqrt(sum(float((current[k]['A']-initial[k]['A']).double().square().sum()) for k in initial)),
                B_norm_l2=math.sqrt(sum(float(current[k]['B'].double().square().sum()) for k in initial)),
                unique_QA=6,QA_exposures=12,updates=2))
        write(run/'source_profiles.json',dict(profiles=profiles,not_a_deployment_expert_pool=True))
    else:
        source_proof=json.loads((REPORT/'sources_complete.json').read_text());source_run=ROOT/source_proof['run']
        if name=='cold':start=initialize(initial,None,name)
        else:
            source='within' if name.startswith('within') else 'cross'
            p=source_run/(source+'.pt');assert digest(p)==source_proof['artifact_sha256'][source+'.pt']
            original_source=torch.load(p,map_location='cpu',weights_only=True)['adapter']
            start=initialize(initial,original_source,name)
        x.configure(start,finetune_shared=True);assert same(x.snapshot(),start)
        atomic_torch(run/'initial_adapter.pt',dict(adapter=start))
        selected={'medical':probe};x.evaluate('initial_probe',selected,selected)
        assert case_complete(run,'initial_probe',selected)
        base_equal=name=='cold' or name.endswith('A_only')
        if base_equal:
            assert all(torch.count_nonzero(start[k]['B'])==0 for k in start)
            if name!='cold':
                assert complete('case',0,spec)
                cold_run=ROOT/json.loads((REPORT/'cold_complete.json').read_text())['run']
                assert semantic(run/'evaluations/initial_probe/medical_image.jsonl')==semantic(cold_run/'evaluations/initial_probe/medical_image.jsonl')
        write(run/'initial_function_gate.json',dict(passed=True,initial_probe_QA=32,
            base_function_equal_required=base_equal,B_zero_all_layers=base_equal,
            A_only_exact_base_NLL_and_predictions_verified=name.endswith('A_only'),
            full_warm_may_change_initial_function_and_first_step_A_access=True))
        # Every condition gets the identical target TRAIN/order/RNG/budget.
        x.configure(start,finetune_shared=True)
        after=fit(x,run,'target_T04',data[4,'train'])
        assert not same(after,start)
        full={'medical':data[4,'test']};x.evaluate('target_full507',full,full)
        assert case_complete(run,'target_full507',full)
    x.monitor.snapshot('control_after',model=x.model)
    finish_resources(x,run,'control_before','control_after',
        'same-PID diagnostic; source pretraining charged separately; includes evaluation, not pure TRAIN peak')
    files={str(p.relative_to(run)):digest(p) for p in run.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.pt']}
    write(REPORT/(name+'_complete.json'),dict(complete=True,spec=spec,run=str(run.relative_to(ROOT)),
        phase=phase,condition=name,artifact_sha256=files,target_full_TEST_QA=507 if phase=='case' else None,
        source_pretraining_updates_charged=4 if phase=='sources' else 0 if name=='cold' else 2,
        independent_target_control_not_formal_CL_or_new_method_success=True))
    assert complete(phase,index,spec)


def main(check,phase,index):
    REPORT.mkdir(parents=True,exist_ok=True)
    if check:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        spec,values=plan();assert len(spec['sources']['within'])==len(spec['sources']['cross'])==6
        init={'layer':{'A':torch.ones(2,3),'B':torch.zeros(4,2)}}
        src={'layer':{'A':torch.full((2,3),2.),'B':torch.full((4,2),3.)}}
        before=copy.deepcopy(src)
        for name in CASES:
            state=initialize(init,src,name)
            assert same(src,before), 'Never mutate a saved source when resetting B'
            if name=='cold':assert same(state,init)
            elif name.endswith('A_only'):
                assert torch.equal(state['layer']['A'],src['layer']['A']) and not torch.count_nonzero(state['layer']['B'])
            else:assert same(state,src)
        write(REPORT/'CPU_PLAN_CHECK.json',dict(passed=True,CPU_only=True,spec=spec,
            actual5_initialization_branches_and_source_nonmutation_checked=True,GPU_started=False))
        return
    run_trial(phase,index)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--phase',choices=['sources','case']);p.add_argument('--case-index',type=int)
    a=p.parse_args()
    try:main(a.check,a.phase,a.case_index)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
