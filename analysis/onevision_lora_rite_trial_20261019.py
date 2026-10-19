"""Continuous native LoRA with the pinned author's LoRA-RITE optimizer.

Published-optimizer adaptation, not a novel algorithm or a full recipe
reproduction. Same adapter is retained through every task; no expert growth.
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
import onevision_learned_reuse_trial as native
from onevision_lora_optimizer_counterfactual_20261019 import author_class, tensor_bytes
from onevision_directional_coeff_trial_v2 import compare, probe
from onevision_knowledge_intervention import reset_seed, case_complete
from onevision_research_worker import atomic_torch, digest
from onevision_capacity_probe import finish_resources, stratify_full_results
from run import hashed, write

ROOT = native.ROOT
REPORT = native.BASE / 'lora_rite_continuous_20261019/v1'
FAMILY = ROOT / 'exp/onevision_lora_rite_continuous_20261019/v1'


def read(path):
    return json.loads(path.read_text())


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[n][k], b[n][k]) for n in a for k in ('A', 'B'))


def displacement(a, b):
    """Exact real-arithmetic norm via low-rank Grams, no dense GPU product."""
    total = 0.
    for name in a:
        left = torch.cat((b[name]['B'].double(), a[name]['B'].double()), dim=1)
        right = torch.cat((b[name]['A'].double(), -a[name]['A'].double()), dim=0)
        total += max(0., float(((left.T @ left) * (right @ right.T).T).sum())) * 16.
    return math.sqrt(total)


def plan():
    old, values = native.plan()
    sources = dict(old['source_sha256'])
    paths = [Path(__file__), ROOT / 'analysis/onevision_lora_rite_entry_20261019.py',
             ROOT / 'analysis/onevision_lora_optimizer_counterfactual_20261019.py',
             ROOT / 'analysis/onevision_directional_coeff_trial_v2.py',
             ROOT / 'code/LoRA_RITE/source_manifest.json',
             ROOT / 'code/LoRA_RITE/source/lora_rite.py']
    sources.update({str(p.relative_to(ROOT)): digest(p) for p in paths})
    spec = dict(source_sha256=sources, protocol_sha256=digest(REPORT / 'PROTOCOL.md'),
        manifest_sha256=old['manifest_sha256'], model_revision=old['model_revision'],
        native_completion_sha256=old['native_completion_sha256'],
        method='continuous_LoRA_author_RITE', seed=42, rank=8, alpha=32,
        trainable_parameters=540672, expert_count=1, taskwise_new_experts=0,
        continuity='same A+B retained across19; only optimizer reset per task as SeqFT',
        train_ids=old['train_ids'], train_QA=1313, stages=19, expected_updates=93,
        diagnostic_ids=old['diagnostic_ids'], generation_ids=old['generation_ids'], full_QA=21454,
        optimizer='pinned author PyTorch LoRARite; parameters alternating A,B per q/v layer',
        optimizer_settings=dict(lr=1e-4, betas=[.9, .999], eps=1e-6,
            clip_unmagnified_grad=1., update_skipping=1., update_capping=0.,
            relative_epsilon=False, weight_decay=0., apply_escape=False, balance_param=False),
        external_factor_gradient_clipping=False,
        training='nativeAnyRes BF16; FP32 q/v LoRA; dropout.1;1epoch;batch16;micro1;floorwarmup/cosine',
        runtime='native deterministic True, threads6, seed42; same image/question/answer/score',
        full_required_regardless_diagnostic=True, TEST_selects_method_or_hypers=False,
        resource_claim='no new experts; actual optimizer state/scratch and CPU/GPU costs measured; no memory advantage assumed',
        full_FWT_BWT=None, paper_adaptation_not_novelty=True)
    return spec, values


def train_stage(x, stage, rows):
    order = sorted(rows, key=lambda r: hashed(r, 'training'))
    reset_seed(); x.model.train(); x.cache.clear(); gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = x.monitor.snapshot(f'RITE_TRAIN_T{stage:02d}_before', model=x.model)
    initial = x.snapshot()
    previous = torch.load(x.run / f'states/stage_{stage - 1:02d}.pt', map_location='cpu', weights_only=True)['adapter']
    assert same(initial, previous)
    parameters = [p for layer in x.layers.values() for p in (layer.A.weight, layer.B.weight)]
    assert all(p.requires_grad for p in parameters) and sum(p.numel() for p in parameters) == 540672
    named = [(name + '.' + k + '.weight', getattr(layer, k).weight)
             for name, layer in x.layers.items() for k in ('A', 'B')]
    optimizer = author_class()(parameters, lr=1e-4, betas=(.9, .999), clip_unmagnified_grad=1.)
    skipped = []
    original_skip = optimizer.helper.skip_update
    def skip(update, threshold):
        result = original_skip(update, threshold)
        skipped.append(bool(torch.count_nonzero(update) and not torch.count_nonzero(result)))
        return result
    optimizer.helper.skip_update = skip
    updates = math.ceil(len(order) / 16)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, min(int(.03 * updates), max(0, updates - 1)), updates)
    records = []
    for step, offset in enumerate(range(0, len(order), 16), 1):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True); skipped.clear()
        batch = order[offset:offset + 16]; losses = []; tokens = 0; old = x.snapshot()
        for row in batch:
            loss, detail = x.loss(row); assert torch.isfinite(loss)
            (loss / len(batch)).backward(); losses.append(float(loss.detach())); tokens += detail['target_tokens']
        raw = torch.cat([p.grad.detach().float().cpu().flatten() for p in parameters])
        assert torch.isfinite(raw).all() and float(raw.norm()) > 0
        atomic_torch(x.run / f'raw_gradients/T{stage:02d}/step{step}.pt', dict(raw_gradient=raw,
            before_adapter=old, layout=[dict(name=n, shape=list(p.shape)) for n, p in named],
            TRAIN_ids=[r['id'] for r in batch]))
        lr = optimizer.param_groups[0]['lr']; assert lr > 0
        # Author optimizer performs its own unmagnified-gradient clipping.
        with torch.no_grad():
            optimizer.step()
        after = x.snapshot()
        assert all(torch.isfinite(v).all() for item in after.values() for v in item.values())
        change = displacement(old, after)
        assert change > 0 and math.isfinite(change), 'Preserve evidence if update is ineffective'
        scheduler.step()
        event = dict(event='rite_train', stage=stage, step=step, QA=len(batch), loss=sum(losses) / len(losses),
            LR=lr, raw_factor_gradient_l2=float(raw.norm()), raw_factor_clip_not_applied=True,
            author_factor_updates_skipped=sum(skipped), author_skip_threshold=optimizer.update_skipping,
            optimizer_persistent_tensor_bytes=tensor_bytes(optimizer.state),
            actual_FP32_deltaW_l2=change, target_tokens=tokens, seconds=time.monotonic() - tick)
        records.append(event)
        with (x.run / 'events.jsonl').open('a') as out:
            out.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)
    after_memory = x.monitor.snapshot(f'RITE_TRAIN_T{stage:02d}_after', model=x.model)
    write(x.run / f'stages/T{stage:02d}.json', dict(stage=stage, updates=records,
        TRAIN_before=before, TRAIN_after=after_memory, optimizer_present_at_after=True,
        effective_stage_deltaW=displacement(initial, x.snapshot())))
    optimizer.zero_grad(set_to_none=True); del optimizer, scheduler, loss, raw
    gc.collect(); torch.cuda.empty_cache()
    return len(records)


def complete(phase, spec=None):
    path = REPORT / (phase + '_complete.json')
    if not path.exists():
        return False
    proof = read(path)
    assert proof['complete'] and (spec is None or proof['spec'] == spec)
    run = (ROOT / proof['run']).resolve(); assert run.is_relative_to(FAMILY.resolve())
    for rel, expected in proof['artifact_sha256'].items():
        p = (run / rel).resolve(); assert p.is_relative_to(run) and digest(p) == expected
    if phase == 'train':
        assert proof['positive_finite_updates'] == 93 and proof['trained_stages'] == 19
        assert read(run / 'initial_exact.json')['passed']
    else:
        assert proof['full_QA'] == 21454 and read(run / 'exact_final_reload.json')['passed']
        assert complete('train', proof['spec'])
    return True


def run_trial(phase):
    assert phase in ('train', 'full') and os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '1')
    torch.set_num_threads(6); torch.use_deterministic_algorithms(True); reset_seed()
    spec, values = plan(); assert spec == read(REPORT / 'CPU_PLAN.json')['spec']
    for rel, expected in spec['source_sha256'].items():
        assert digest(ROOT / rel) == expected
    lock = (REPORT / 'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (REPORT / (phase + '_attempt.json')).exists(), 'No blind retry; preserve failure'
    import onevision_research_device_entry as device
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES']); inventory = device.inventory()
    assert device.select(inventory, os.getuid()) == gpu
    manifest, data, domains, tests, full, original, proof = values
    generation = {d: [r for r in rr if r['id'] in set(spec['generation_ids'][d])] for d, rr in tests.items()}
    run = FAMILY / phase / time.strftime('%Y%m%d_%H%M%S'); run.mkdir(parents=True)
    write(REPORT / (phase + '_attempt.json'), dict(run=str(run.relative_to(ROOT)), spec=spec, pid=os.getpid()))
    write(run / 'spec.json', spec); write(run / 'physical_device_execution.json', dict(gpu=gpu, inventory=inventory))
    x = native.MetricExperiment(manifest, run)
    x.configure(x.original_initial, finetune_shared=True)
    assert sum(p.numel() for p in x.model.parameters() if p.requires_grad) == 540672
    x.monitor.snapshot('RITE_process_before', model=x.model); x.model.eval(); x.forward_gate()
    trained = None
    if phase == 'train':
        atomic_torch(run / 'states/stage_00.pt', dict(adapter=x.snapshot(), stage=0))
        x.evaluate('initial', tests, generation); compare(run, 'initial', original, 'base_initial', tests)
        write(run / 'initial_exact.json', dict(passed=True, NLL_QA=500, generated_QA=155))
        count = 0
        for stage in range(1, 20):
            count += train_stage(x, stage, data[stage, 'train'])
            atomic_torch(run / f'states/stage_{stage:02d}.pt', dict(adapter=x.snapshot(), stage=stage,
                source_sha256=spec['source_sha256'], positive_finite_updates=count))
        assert count == 93
        x.model.eval(); x.evaluate('final', tests, generation); assert case_complete(run, 'final', tests)
    else:
        assert complete('train', spec)
        trained = ROOT / read(REPORT / 'train_complete.json')['run']
        x.restore(torch.load(trained / 'states/stage_19.pt', map_location='cpu', weights_only=True)['adapter'])
        assert same(x.snapshot(), torch.load(trained / 'states/stage_19.pt', map_location='cpu', weights_only=True)['adapter'])
        x.evaluate('final_reload', tests, generation); compare(run, 'final_reload', trained, 'final', tests)
        write(run / 'exact_final_reload.json', dict(passed=True, NLL_QA=500, generated_QA=155))
        x.evaluate('final_full_input_router', full, full); assert case_complete(run, 'final_full_input_router', full)
        stratify_full_results(run, full, {r['id'] for rr in tests.values() for r in rr})
    x.monitor.snapshot('RITE_process_after', model=x.model)
    finish_resources(x, run, 'RITE_process_before', 'RITE_process_after',
        'whole process includes loading/evaluation/checkpoints; per-stage TRAIN windows reported separately')
    files = {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file() and p.suffix in ('.pt', '.json', '.jsonl')}
    write(REPORT / (phase + '_complete.json'), dict(complete=True, spec=spec, run=str(run.relative_to(ROOT)),
        artifact_sha256=files, trained_stages=19 if phase == 'train' else 0,
        positive_finite_updates=93 if phase == 'train' else 0, full_QA=21454 if phase == 'full' else None,
        full_FWT=None, full_BWT=None))
    assert complete(phase, spec)
