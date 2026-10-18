"""Two fixed-budget continuous SVD controls, separate from all old trainers.

Rank8 q/v, frozen S, no experts or routing. Both complete1313TRAIN/19 stages
and full21454 regardless diagnostics. Protected/free share an exact saved S0.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import fcntl
import gc
import json
import math
from pathlib import Path
import time

import torch

import onevision_osft_budget_core_v3 as core
import onevision_learned_reuse_trial as native
from onevision_research_worker import atomic_torch, digest
from onevision_knowledge_intervention import reset_seed, case_complete
from onevision_capacity_probe import finish_resources, stratify_full_results
from run import freeze, hashed, write

ROOT = core.ROOT
REPORT = ROOT/'summary/onevision_diagnostics_20261016/osft_budget_20261018/v3'
FAMILY = ROOT/'exp/onevision_osft_budget_20261018/v3'
VIEWS = ['svd_qv_r8_free', 'svd_qv_r8_orthogonal']


def read(path):
    return json.loads(path.read_text())


def plan():
    _, values = native.plan()
    manifest, data, domains, tests, full, original, proof = values
    parent = read(original/'protocol.json')
    paths = [Path(__file__), ROOT/'analysis/onevision_osft_budget_core_v3.py',
             ROOT/'analysis/onevision_osft_budget_entry_v3.py']
    sources = native.source_hashes()
    sources.update({str(p.relative_to(ROOT)): digest(p) for p in paths})
    spec = dict(source_sha256=sources, protocol_sha256=digest(REPORT/'PROTOCOL.md'),
        manifest_sha256=parent['manifest_sha256'], model_revision=parent['model_revision'],
        native_complete_sha256=digest(native.BASE/'deterministic_native/complete.json'),
        author_kernel_sha256=core.author_kernels()[1], variants=VIEWS,
        seed=42, rank=8, target_projections=['q_proj', 'v_proj'], S_low_frozen=True,
        trainable_parameters=540672, epochs=1, batch=16, LR=1e-4,
        warmup='floor(.03*updates), capped below updates', dropout=0,
        original_LoRA_dropout=.1, activation='nativeAnyRes BF16',
        train_QA=1313, stages=19, actual_expected_updates=93,
        train_ids={str(t): [r['id'] for r in data[t, 'train']] for t in range(1, 20)},
        diagnostic_ids={d: [r['id'] for r in rows] for d, rows in tests.items()},
        generation_ids=parent['generation_ids'], full_QA=21454,
        common_initial='free stage00 exact saved factors; protected must reproduce all500/155',
        per_stage='refactor CURRENT reconstructed weights, never original base reset',
        old_vs_new_constraints='none vs pinned author gradient AND post-Adam parameter projection',
        full_both_regardless_diagnostic=True, TEST_selection=False,
        initial_native_bitwise_equivalence_claim=False, scalar_metrics_FWT_BWT_missing_until_own_sparse=True,
        source_scope='capacity-matched SVD control, not exact full OSFT or new-method novelty')
    return spec, values


def snapshot(x):
    return {name: {k: t.detach().cpu().clone() for k, t in layer.state_dict().items()}
            for name, layer in x.layers.items()}


def restore(x, state):
    assert state.keys() == x.layers.keys()
    for name, layer in x.layers.items():
        layer._compiled_eval_weight = None
        layer.load_state_dict(state[name], strict=True)


def install(x, protected):
    kernels, _ = core.author_kernels()
    layers, errors = {}, []
    for name, old in list(x.layers.items()):
        assert not torch.count_nonzero(old.B.weight), 'Initial old LoRA must be functionally untouched'
        layer = core.BudgetSVDLinear(old.weight, old.bias, rank=8, protected=protected, kernels=kernels)
        with torch.no_grad():
            rebuilt = layer.current_weight(torch.float32)
            error = float((rebuilt-old.weight.float()).norm()/old.weight.float().norm())
        assert error < 2e-5
        errors.append(dict(layer=name, relative_FP32_SVD_error=error))
        parent, child = name.rsplit('.', 1)
        setattr(x.model.get_submodule(parent), child, layer)
        layers[name] = layer
    x.layers = layers
    x.initial = x.original_initial = None
    assert len(layers) == 48
    assert sum(p.numel() for p in x.model.parameters() if p.requires_grad) == 540672
    write(x.run/'initial_SVD_weight_errors.json', dict(layers=errors, bound=2e-5,
        own_S0_required=True, bitwise_native_initial_claim=False))


def complete(phase, view, spec):
    pointer = REPORT/(phase+'_'+view+'_complete.json')
    if not pointer.exists():
        return False
    proof = read(pointer)
    assert proof['complete'] and proof['spec'] == spec
    run = (ROOT/proof['run']).resolve()
    assert run.is_relative_to(FAMILY.resolve())
    for rel, expected in proof['artifact_sha256'].items():
        p = (run/rel).resolve()
        assert p.is_relative_to(run) and digest(p) == expected
    assert read(run/'renderer_after_SVD.json')['passed']
    if phase == 'train':
        assert proof['trained_stages'] == 19 and proof['positive_finite_updates'] == 93
    else:
        assert proof['full_QA'] == 21454 and read(run/'exact_final_reload.json')['passed']
    return True


def rows(run, label, domain):
    result = [json.loads(s) for s in (run/f'evaluations/{label}/{domain}_image.jsonl').read_text().splitlines()]
    return [{k: v for k, v in row.items() if k != 'generation_seconds'} for row in result]


def compare(x, run, label, reference, reference_label, tests):
    assert case_complete(run, label, tests)
    for domain in tests:
        assert rows(run, label, domain) == rows(reference, reference_label, domain)


def train_stage(x, stage, training):
    reset_seed()
    x.model.train()
    x.cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = x.monitor.snapshot('SVD_TRAIN_before_'+str(stage), model=x.model)
    named = [(k, p) for k, p in x.model.named_parameters() if p.requires_grad]
    assert sum(p.numel() for _, p in named) == 540672
    parameters = [p for _, p in named]
    before_weights = {name: layer.current_weight().detach().cpu().clone() for name, layer in x.layers.items()}
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0)
    order = sorted(training, key=lambda r: hashed(r, 'training'))
    updates = math.ceil(len(order)/16)
    from transformers import get_cosine_schedule_with_warmup
    warmup = min(int(.03*updates), max(0, updates-1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, updates)
    records = []
    for step, start in enumerate(range(0, len(order), 16), 1):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        minibatch, losses, tokens = order[start:start+16], [], 0
        for row in minibatch:
            loss, details = x.loss(row)
            assert torch.isfinite(loss)
            (loss/len(minibatch)).backward()
            losses.append(float(loss.detach()))
            tokens += details['target_tokens']
        raw = torch.cat([p.grad.detach().float().cpu().flatten() for _, p in named])
        assert torch.isfinite(raw).all() and raw.norm() > 0
        for layer in x.layers.values():
            layer.project_gradients()
        projected = torch.cat([p.grad.detach().float().cpu().flatten() for _, p in named])
        assert torch.isfinite(projected).all() and projected.norm() > 0
        if not next(iter(x.layers.values())).protected:
            assert torch.equal(raw, projected)
        if stage == step == 1 and getattr(x, 'paired_free_run', None) is not None:
            original = torch.load(x.paired_free_run/'raw_gradients/T01/step1.pt', map_location='cpu', weights_only=True)
            assert torch.equal(raw, original['before_projection'])
            write(x.run/'paired_first_gradient_exact.json', dict(passed=True, TRAIN_QA=len(minibatch),
                before_projection=True, free_run=str(x.paired_free_run.relative_to(ROOT))))
        previous = [p.detach().clone() for p in parameters]
        old_factors = {k: p.detach().cpu().clone() for (k, _), p in zip(named, previous)}
        atomic_torch(x.run/f'raw_gradients/T{stage:02d}/step{step}.pt',
            dict(before_projection=raw, after_projection=projected,
                 layout=[dict(name=k, shape=list(p.shape), count=p.numel()) for k, p in named],
                 before_factor_parameters=old_factors,
                 TRAIN_ids=[r['id'] for r in minibatch], stage=stage, step=step))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert torch.isfinite(norm)
        used_lr = optimizer.param_groups[0]['lr']
        assert used_lr > 0
        optimizer.step()
        for layer in x.layers.values():
            layer.project_parameters()
        actual_delta = math.sqrt(sum(float((p.detach()-old).double().square().sum()) for p, old in zip(parameters, previous)))
        assert actual_delta > 0 and math.isfinite(actual_delta)
        effective_squared = sum(core.displacement_squared(old_factors[name+'.U_low'].to(layer.U_low.device),
            old_factors[name+'.V_low'].to(layer.V_low.device), layer.U_low.detach(), layer.V_low.detach(), layer.S_low)
            for name, layer in x.layers.items())
        scheduler.step()
        record = dict(event='svd_train', stage=stage, step=step, QA=len(minibatch),
            loss=sum(losses)/len(losses), LR=used_lr, raw_gradient_l2=float(raw.norm()),
            projected_gradient_l2=float(projected.norm()), gradient_l2_preclip=float(norm),
            actual_delta_after_Adam_and_projection_l2=actual_delta, target_tokens=tokens,
            effective_FP32_weight_displacement_l2=math.sqrt(effective_squared),
            seconds=time.monotonic()-tick)
        records.append(record)
        with (x.run/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        del previous, raw, projected
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, loss
    gc.collect()
    torch.cuda.empty_cache()
    after = x.monitor.snapshot('SVD_TRAIN_after_'+str(stage), model=x.model)
    write(x.run/f'TRAIN_windows/T{stage:02d}.json', dict(before=before, after=after,
        cuda_peak_allocated_bytes=after['gpu_peak_allocated_bytes'],
        cuda_peak_reserved_bytes=after['gpu_peak_reserved_bytes'],
        scope='one-stage model TRAIN including raw-gradient persistence; excludes checkpoint/refactor/evaluation'))
    changes = []
    for name, layer in x.layers.items():
        weight = layer.current_weight().detach().cpu()
        changes.append(dict(layer=name, changed_BF16_coordinates=int(torch.count_nonzero(weight != before_weights[name])),
            actual_BF16_weight_displacement_l2=float((weight.double()-before_weights[name].double()).norm())))
    assert sum(row['changed_BF16_coordinates'] for row in changes) > 0, 'No actual layer-weight learning this stage'
    write(x.run/f'TRAIN_weight_changes/T{stage:02d}.json', dict(layers=changes, whole_stage_zero_functional_update=False))
    assert len(records) == updates
    return len(records)


def run_trial(phase, view):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0', '1'] and view in VIEWS
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    reset_seed()
    spec, values = plan()
    assert spec == read(REPORT/'CPU_PLAN.json')['spec'] == read(REPORT/'ADMISSION.json')['spec']
    if phase == 'train' and view == VIEWS[1]:
        assert complete('train', VIEWS[0], spec)
    if phase == 'full':
        assert complete('train', view, spec)
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = (REPORT/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    name = phase+'_'+view
    assert not (REPORT/(name+'_attempt.json')).exists(), 'Preserve failure; no blind retry'
    import onevision_research_device_entry as device
    cards = device.inventory()
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert device.select(cards, os.getuid()) == gpu
    manifest, data, domains, tests, full, _, _ = values
    generation = {d: [r for r in rows if r['id'] in set(spec['generation_ids'][d])] for d, rows in tests.items()}
    assert sum(map(len, generation.values())) == 155
    run = FAMILY/name/time.strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    write(REPORT/(name+'_attempt.json'), dict(run=str(run.relative_to(ROOT)), spec=spec, pid=os.getpid()))
    write(run/'spec.json', spec)
    write(run/'physical_device_execution.json', dict(physical_GPU=gpu, inventory=cards, single_research_stream=True))
    x = native.MetricExperiment(manifest, run)
    install(x, view.endswith('_orthogonal'))
    trained = None
    if phase == 'full':
        trained = ROOT/read(REPORT/('train_'+view+'_complete.json'))['run']
        state = torch.load(trained/'states/stage_19.pt', map_location='cpu', weights_only=True)
        restore(x, state['layers'])
    elif view == VIEWS[1]:
        reference = ROOT/read(REPORT/('train_'+VIEWS[0]+'_complete.json'))['run']
        state = torch.load(reference/'states/stage_00.pt', map_location='cpu', weights_only=True)
        restore(x, state['layers'])
        assert all(torch.equal(v, state['layers'][k][n]) for k, layer in snapshot(x).items() for n, v in layer.items())
        x.paired_free_run = reference
    x.monitor.snapshot('SVD_process_before', model=x.model)
    x.model.eval()
    x.forward_gate()
    write(run/'renderer_after_SVD.json', read(run/'renderer_gpu_gate.json'))
    if phase == 'train':
        atomic_torch(run/'states/stage_00.pt', dict(layers=snapshot(x), stage=0, source_sha256=spec['source_sha256']))
        x.evaluate('initial', tests, generation)
        if view == VIEWS[1]:
            compare(x, run, 'initial', reference, 'initial', tests)
            write(run/'paired_initial_exact.json', dict(passed=True, NLL_QA=500, generated_QA=155,
                reference_run=str(reference.relative_to(ROOT)), stage00_all_tensors_exact=True))
        actual_updates = 0
        for stage in range(1, 20):
            if stage > 1:
                refactors = []
                for key, layer in x.layers.items():
                    with torch.no_grad():
                        current = layer.current_weight().detach()
                        layer.refactor(current)
                        rebuilt = layer.current_weight(torch.float32)
                        error = float((rebuilt-current.float()).norm()/current.float().norm())
                    assert error < 2e-5
                    refactors.append(dict(layer=key, relative_FP32_SVD_error=error,
                        changed_BF16_coordinates=int(torch.count_nonzero(rebuilt.to(current.dtype) != current))))
                write(run/f'refactor/T{stage:02d}.json', dict(layers=refactors, optimizer_updates=0,
                    current_weights_carried=True, predeclared_error_bound=2e-5))
            actual_updates += train_stage(x, stage, data[stage, 'train'])
            atomic_torch(run/f'states/stage_{stage:02d}.pt', dict(layers=snapshot(x), stage=stage,
                         source_sha256=spec['source_sha256'], positive_finite_updates=actual_updates))
        assert actual_updates == 93
        x.evaluate('final', tests, generation)
        assert case_complete(run, 'final', tests)
    else:
        x.evaluate('final_reload', tests, generation)
        compare(x, run, 'final_reload', trained, 'final', tests)
        write(run/'exact_final_reload.json', dict(passed=True, NLL_QA=500, generated_QA=155))
        x.evaluate('final_full_input_router', full, full)
        assert case_complete(run, 'final_full_input_router', full)
        stratify_full_results(run, full, {r['id'] for rows in tests.values() for r in rows})
    x.monitor.snapshot('SVD_process_after', model=x.model)
    finish_resources(x, run, 'SVD_process_before', 'SVD_process_after',
        'whole process includes SVD, checkpointing and evaluation; TRAIN windows separately reset')
    artifacts = {str(p.relative_to(run)): digest(p) for p in run.rglob('*')
                 if p.is_file() and p.suffix in ['.pt', '.json', '.jsonl']}
    write(REPORT/(name+'_complete.json'), dict(complete=True, spec=spec, run=str(run.relative_to(ROOT)),
        artifact_sha256=artifacts, trained_stages=19 if phase == 'train' else None,
        positive_finite_updates=93 if phase == 'train' else 0, full_QA=21454 if phase == 'full' else None,
        full_FWT=None, full_BWT=None))
    assert complete(phase, view, spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--phase', choices=['train', 'full'])
    parser.add_argument('--view', choices=VIEWS)
    args = parser.parse_args()
    if args.check:
        assert not torch.cuda.is_initialized()
        spec, _ = plan()
        write(REPORT/'CPU_PLAN.json', dict(CPU_only=True, spec=spec, GPU_started=False))
    else:
        run_trial(args.phase, args.view)


if __name__ == '__main__':
    main()
