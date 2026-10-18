"""Direct lightweight continual-tuning trial, inspired by PiSSA/LoRA-XS.

Both fixed directions get the full stream and final test. No expert expansion,
route, merge, new task-wise initialization, or TEST-based method selection.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import fcntl
import gc
import json
import math
from pathlib import Path
import time

import torch

import onevision_directional_coeff_core as core
import onevision_learned_reuse_trial as native
from onevision_research_worker import atomic_torch, digest
from onevision_knowledge_intervention import reset_seed, case_complete
from onevision_capacity_probe import finish_resources, stratify_full_results
from run import hashed, write

ROOT=core.ROOT
REPORT=ROOT/'summary/onevision_diagnostics_20261016/directional_coeff_20261018/v1'
FAMILY=ROOT/'exp/onevision_directional_coeff_20261018/v1'
VIEWS=core.VIEWS


def read(path):
    return json.loads(path.read_text())


def plan():
    _,values=native.plan()
    manifest,data,domains,tests,full,original,proof=values
    parent=read(original/'protocol.json')
    paths=[Path(__file__),ROOT/'analysis/onevision_directional_coeff_core.py',
           ROOT/'analysis/onevision_directional_coeff_entry.py',ROOT/'code/LoRA_XS/source_manifest.json']
    sources=native.source_hashes();sources.update({str(p.relative_to(ROOT)):digest(p) for p in paths})
    spec=dict(source_sha256=sources,protocol_sha256=digest(REPORT/'PROTOCOL.md'),
        manifest_sha256=parent['manifest_sha256'],model_revision=parent['model_revision'],
        native_complete_sha256=digest(native.BASE/'deterministic_native/complete.json'),
        variants=VIEWS,seed=42,rank=16,target_projections=['q_proj','v_proj'],
        trainable_parameters=12288,original_LoRA_trainable_parameters=540672,
        frozen_basis_parameters=1081344,fixed_basis=True,coefficient_initialization='exact zeros',
        epochs=1,batch=16,LR=1e-4,dropout=.1,scale=1.,
        warmup='floor(.03*updates), capped below updates',activation='nativeAnyRes BF16',
        train_QA=1313,stages=19,actual_expected_updates=93,
        train_ids={str(t):[r['id'] for r in data[t,'train']] for t in range(1,20)},
        diagnostic_ids={d:[r['id'] for r in rr] for d,rr in tests.items()},
        generation_ids=parent['generation_ids'],full_QA=21454,
        common_initial='exact native base_initial500/155 and paired variant outputs',
        continuous='same coefficients carried across19tasks; no reset, merge or new module',
        direction_control='principal U*S,V vs deterministic random orthobases with same S',
        additional_TRAIN_probe_forwards=sum(2*min(2,len(data[t,'train'])) for t in range(1,20)),
        full_both_regardless_diagnostic=True,TEST_selection=False,
        full_FWT_BWT_missing_until_own_sparse=True,
        paper_adaptation='LoRA-XS formula/PiSSA principal directions; rank16qv/Rzero/commonLR/dropout, continuous19tasks; not original-recipe reproduction or novelty claim')
    return spec,values


def snapshot(x):
    return {n:l.coeff.detach().cpu().clone() for n,l in x.layers.items()}


def restore(x,state):
    assert state.keys()==x.layers.keys()
    with torch.no_grad():
        for n,l in x.layers.items():
            assert state[n].shape==l.coeff.shape
            l.coeff.copy_(state[n])


def basis_state(x):
    return {n:{k:getattr(l,k).detach().cpu().clone() for k in ['left','right','singular']} for n,l in x.layers.items()}


def install(x,view):
    for p in x.model.parameters():p.requires_grad=False
    layers={};start=time.monotonic()
    for name,old in list(x.layers.items()):
        assert not torch.count_nonzero(old.B.weight)
        layer=core.DirectionalLinear(old.weight,old.bias,name,view)
        parent,child=name.rsplit('.',1);setattr(x.model.get_submodule(parent),child,layer);layers[name]=layer
    x.layers=layers;x.initial=x.original_initial=None;x.pristine={}
    assert len(layers)==48 and sum(p.numel() for p in x.model.parameters() if p.requires_grad)==12288
    write(x.run/'basis_installation.json',dict(seconds=time.monotonic()-start,view=view,
        original_dense_weight_unchanged=True,trainable_parameters=12288,payloads={n:l.memory_payload() for n,l in layers.items()}))


def semantic_rows(run,label,domain):
    rr=[json.loads(s) for s in (run/f'evaluations/{label}/{domain}_image.jsonl').read_text().splitlines()]
    keys=['id','references','native_task','nll','prediction','scores']
    return [{k:r.get(k) for k in keys} for r in rr]


def compare(run,label,reference,reference_label,tests):
    assert case_complete(run,label,tests)
    for domain in tests:assert semantic_rows(run,label,domain)==semantic_rows(reference,reference_label,domain)


def probe(x,rows):
    mode=x.model.training
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            x.model.eval()
            return [float(x.loss(row)[0]) for row in rows]
    finally:x.model.train(mode)


def train_stage(x,stage,training):
    order=sorted(training,key=lambda r:hashed(r,'training'))
    selected=sorted(training,key=lambda r:hashed(r,'coeff-current-TRAIN-probe'))[:2]
    before_probe=probe(x,selected)
    reset_seed();x.model.train();x.cache.clear();gc.collect();torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before=x.monitor.snapshot('COEFF_TRAIN_before_'+str(stage),model=x.model)
    named=[(n,p) for n,p in x.model.named_parameters() if p.requires_grad]
    assert sum(p.numel() for _,p in named)==12288
    parameters=[p for _,p in named];stage_initial=snapshot(x)
    if stage>1:
        previous=torch.load(x.run/f'states/stage_{stage-1:02d}.pt',map_location='cpu',weights_only=True)['coefficients']
        assert all(torch.equal(stage_initial[n],previous[n]) for n in previous)
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0)
    updates=math.ceil(len(order)/16)
    from transformers import get_cosine_schedule_with_warmup
    scheduler=get_cosine_schedule_with_warmup(optimizer,min(int(.03*updates),max(0,updates-1)),updates)
    events=[]
    for step,offset in enumerate(range(0,len(order),16),1):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        rows=order[offset:offset+16];losses=[];tokens=0;old=snapshot(x)
        for row in rows:
            loss,details=x.loss(row);assert torch.isfinite(loss)
            (loss/len(rows)).backward();losses.append(float(loss.detach()));tokens+=details['target_tokens']
        raw=torch.cat([p.grad.detach().float().cpu().flatten() for _,p in named])
        assert torch.isfinite(raw).all() and raw.norm()>0
        atomic_torch(x.run/f'raw_gradients/T{stage:02d}/step{step}.pt',dict(raw_gradient=raw,
            before_coefficients=old,layout=[dict(name=n,shape=list(p.shape)) for n,p in named],
            TRAIN_ids=[r['id'] for r in rows],stage=stage,step=step))
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm)
        lr=optimizer.param_groups[0]['lr'];assert lr>0
        optimizer.step();new=snapshot(x)
        delta=math.sqrt(sum(float((new[n]-old[n]).double().square().sum()) for n in new))
        effective=math.sqrt(sum(l.displacement_squared(old[n]) for n,l in x.layers.items()))
        assert delta>0 and effective>0 and math.isfinite(delta) and math.isfinite(effective)
        scheduler.step()
        event=dict(event='coeff_train',stage=stage,step=step,QA=len(rows),loss=sum(losses)/len(rows),LR=lr,
            raw_gradient_l2=float(raw.norm()),gradient_l2_preclip=float(norm),
            actual_coefficient_displacement_l2=delta,effective_FP32_weight_displacement_l2=effective,
            target_tokens=tokens,seconds=time.monotonic()-tick)
        events.append(event)
        with (x.run/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    optimizer.zero_grad(set_to_none=True);del optimizer,scheduler,loss;gc.collect();torch.cuda.empty_cache()
    after=x.monitor.snapshot('COEFF_TRAIN_after_'+str(stage),model=x.model)
    write(x.run/f'TRAIN_windows/T{stage:02d}.json',dict(before=before,after=after,
        cuda_peak_allocated_bytes=after['gpu_peak_allocated_bytes'],cuda_peak_reserved_bytes=after['gpu_peak_reserved_bytes'],
        scope='actual model TRAIN including gradient persistence; excludes probes/install/evaluation/checkpoint'))
    change=math.sqrt(sum(l.displacement_squared(stage_initial[n]) for n,l in x.layers.items()))
    assert change>0
    write(x.run/f'TRAIN_weight_changes/T{stage:02d}.json',dict(whole_stage_zero_functional_update=False,
        effective_FP32_displacement_l2=change,no_BF16_merge_performed=True))
    after_probe=probe(x,selected)
    write(x.run/f'TRAIN_probes/T{stage:02d}.json',dict(ids=[r['id'] for r in selected],before_NLL=before_probe,
        after_NLL=after_probe,optimizer_updates=0,used_for_training_decisions=False))
    assert len(events)==updates
    return len(events)


def complete(phase,view,spec):
    path=REPORT/(phase+'_'+view+'_complete.json')
    if not path.exists():return False
    proof=read(path);assert proof['complete'] and proof['spec']==spec
    run=(ROOT/proof['run']).resolve();assert run.is_relative_to(FAMILY.resolve())
    for rel,h in proof['artifact_sha256'].items():assert digest(run/rel)==h
    assert read(run/'renderer_after_direction.json')['passed']
    if phase=='train':
        assert proof['trained_stages']==19 and proof['positive_finite_updates']==93
        assert read(run/'initial_exact.json')['passed']
    else:
        assert proof['full_QA']==21454 and read(run/'exact_final_reload.json')['passed']
        assert complete('train',view,spec)
    return True


def run_trial(phase,view):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0','1'] and view in VIEWS
    torch.set_num_threads(4)
    spec,values=plan();assert spec==read(REPORT/'CPU_PLAN.json')['spec']
    for rel,h in spec['source_sha256'].items():assert digest(ROOT/rel)==h
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    name=phase+'_'+view;assert not (REPORT/(name+'_attempt.json')).exists(),'Preserve failure; no blind retry'
    import onevision_research_device_entry as device
    cards=device.inventory();gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert device.select(cards,os.getuid())==gpu
    manifest,data,domains,tests,full,original,proof=values
    generation={d:[r for r in rr if r['id'] in set(spec['generation_ids'][d])] for d,rr in tests.items()}
    run=FAMILY/name/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
    write(REPORT/(name+'_attempt.json'),dict(run=str(run.relative_to(ROOT)),spec=spec,pid=os.getpid()))
    write(run/'spec.json',spec);write(run/'physical_device_execution.json',dict(physical_GPU=gpu,inventory=cards,single_research_stream=True))
    x=native.MetricExperiment(manifest,run);install(x,view);frozen=basis_state(x)
    trained=None
    if phase=='full':
        trained=ROOT/read(REPORT/('train_'+view+'_complete.json'))['run']
        reference_bases=torch.load(trained/'frozen_bases.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(frozen[n][k],reference_bases[n][k]) for n in frozen for k in frozen[n])
        restore(x,torch.load(trained/'states/stage_19.pt',map_location='cpu',weights_only=True)['coefficients'])
        write(run/'exact_basis_reload.json',dict(passed=True,all48_frozen_bases_exact=True))
    atomic_torch(run/'frozen_bases.pt',frozen);del frozen
    x.monitor.snapshot('COEFF_process_before',model=x.model);x.model.eval();x.forward_gate()
    write(run/'renderer_after_direction.json',read(run/'renderer_gpu_gate.json'))
    if phase=='train':
        atomic_torch(run/'states/stage_00.pt',dict(coefficients=snapshot(x),stage=0,source_sha256=spec['source_sha256']))
        x.evaluate('initial',tests,generation)
        compare(run,'initial',original,'base_initial',tests)
        if view==VIEWS[1]:
            reference=ROOT/read(REPORT/('train_'+VIEWS[0]+'_complete.json'))['run']
            compare(run,'initial',reference,'initial',tests)
        write(run/'initial_exact.json',dict(passed=True,NLL_QA=500,generated_QA=155,native_reference=str(original.relative_to(ROOT)),
            paired_variant_exact=view==VIEWS[1],zero_coefficient_function_exact=True))
        count=0
        for stage in range(1,20):
            count+=train_stage(x,stage,data[stage,'train'])
            atomic_torch(run/f'states/stage_{stage:02d}.pt',dict(coefficients=snapshot(x),stage=stage,
                source_sha256=spec['source_sha256'],positive_finite_updates=count))
        assert count==93
        x.evaluate('final',tests,generation);assert case_complete(run,'final',tests)
    else:
        x.evaluate('final_reload',tests,generation);compare(run,'final_reload',trained,'final',tests)
        write(run/'exact_final_reload.json',dict(passed=True,NLL_QA=500,generated_QA=155))
        x.evaluate('final_full_input_router',full,full);assert case_complete(run,'final_full_input_router',full)
        stratify_full_results(run,full,{r['id'] for rr in tests.values() for r in rr})
    x.monitor.snapshot('COEFF_process_after',model=x.model)
    finish_resources(x,run,'COEFF_process_before','COEFF_process_after',
        'whole process includes initialization/evaluation/probes/checkpoints; model TRAIN windows reset separately')
    artifacts={str(p.relative_to(run)):digest(p) for p in run.rglob('*') if p.is_file() and p.suffix in ['.pt','.json','.jsonl']}
    write(REPORT/(name+'_complete.json'),dict(complete=True,spec=spec,run=str(run.relative_to(ROOT)),artifact_sha256=artifacts,
        trained_stages=19 if phase=='train' else None,positive_finite_updates=93 if phase=='train' else 0,
        full_QA=21454 if phase=='full' else None,full_FWT=None,full_BWT=None))
    assert complete(phase,view,spec)
