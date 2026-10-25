"""Read already-learned working adapters at each QA's actual task boundary.

No training, scoring, old-prediction reading or model work occurs on import.
The two S19 exact-reconstruction controls precede every new generation.
"""
from pathlib import Path
import gc
import importlib.util
import json
import os
import sys
import time
import traceback

PHASE = 'acquisition_readout'
METHODS = ('original', 'naive')
DOMAINS = ('medical', 'geography')
PREFIX = {'original': 'original', 'naive': 'qv'}
COORDINATES = 540672
PARAMETERS = 96
ADAPTER_BYTES = 2162688
RUN = Path(os.environ['LLMCL_RUN'])


def read(path):
    return json.loads(Path(path).read_bytes())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def same_receipt(actual, expected):
    """Compare every semantic field, excluding only physical device identity."""
    for key, value in actual.items():
        if key != 'device':
            assert expected[key] == value, key


def parents(ctx):
    result = {}
    assert set(ctx.ready['acquisition_methods']) == set(METHODS)
    for name in METHODS:
        source = ctx.ready['acquisition_methods'][name]
        directory = Path(source['run_dir'])
        train_path, verified_path = Path(source['train_complete']), Path(source['verified'])
        assert directory.is_absolute() and train_path.is_absolute() and verified_path.is_absolute()
        for path in (train_path, verified_path):
            assert str(path) in ctx.ready['prior_files']
            assert ctx.sha(path) == ctx.ready['prior_files'][str(path)]
        train, verified = read(train_path), read(verified_path)
        train_phase = 'train_' + PREFIX[name]
        assert verified['complete'] and train['ready_sha256'] == verified['ready_sha256']
        assert verified['source_COMPLETE_sha256'][train_phase] == ctx.sha(train_path)
        assert train['FIT_backwards'] == 1213 and train['optimizer_updates'] == 87
        assert train['frozen_unchanged'] and train['coordinates'] == COORDINATES
        initial = directory / (PREFIX[name] + '_initial.pt')
        original_initial = train['initial_checkpoint']
        assert Path(original_initial['path']).name == initial.name
        assert train['files'][initial.name] == original_initial['sha256']
        assert initial.stat().st_size == original_initial['bytes']
        result[name] = dict(directory=directory, train=train, train_path=train_path,
                            verified_path=verified_path, initial=initial)
    return result


def reconstruct(ctx, name, stage, parent, deployment, adaptation):
    import torch
    assert 1 <= stage <= 19
    prefix, directory, train = PREFIX[name], parent['directory'], parent['train']
    snapshot = directory / f'{prefix}_adapter_S{stage:02}.pt'
    stage_path = directory / f'{prefix}_STAGE_{stage:02}_COMPLETE.json'
    snapshot_sha, stage_sha = ctx.sha(snapshot), ctx.sha(stage_path)
    assert snapshot_sha == train['files'][snapshot.name]
    assert stage_sha == train['files'][stage_path.name]
    sealed = read(stage_path)
    assert sealed['stage'] == stage and sealed['bytes'] == ADAPTER_BYTES and sealed['frozen_unchanged']
    model, initial_reload = deployment.load_complete(parent['initial'], ctx.device, ctx.storage)
    same_receipt(deployment.receipt(model, ctx.storage), train['original_base'])
    ctx.reset()
    method = adaptation.make_plain(model, ctx.asset, ctx.storage)
    expected = {n + suffix for n in method.rank_plan for suffix in ('.A.weight', '.B.weight')}
    working = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert len(method.rank_plan) == 48 and set(working) == expected and len(working) == PARAMETERS
    assert all(rank == 8 for rank in method.rank_plan.values())
    assert sum(p.numel() for p in working.values()) == COORDINATES
    assert all(p.is_cuda and p.dtype == torch.float32 for p in working.values())
    frozen = lambda: ctx.storage.digest({n: p for n, p in model.named_parameters() if n not in working})
    frozen_before = frozen()
    slow_ids = {id(t) for t in adaptation.ema_state(method).values()}
    ordinary = lambda: {n: t for n, t in model.core.named_buffers() if id(t) not in slow_ids}
    buffers_before = ctx.storage.digest(ordinary())
    state = torch.load(snapshot, map_location='cpu', weights_only=True)
    assert isinstance(state, dict) and set(state) == expected
    assert all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and t.dtype == torch.float32
               and tuple(t.shape) == tuple(working[n].shape) for n, t in state.items())
    assert sum(t.numel() for t in state.values()) == COORDINATES
    assert sum(t.numel() * t.element_size() for t in state.values()) == ADAPTER_BYTES
    assert ctx.storage.digest(state) == sealed['adapter_digest']
    shapes = {n: list(t.shape) for n, t in state.items()}
    parameter_ids = {n: id(p) for n, p in working.items()}
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        for n, value in state.items():
            working[n].copy_(value.to(ctx.device))
        assert all(torch.isfinite(p).all() for p in working.values())
    assert parameter_ids == {n: id(p) for n, p in model.named_parameters() if p.requires_grad}
    copied_digest = ctx.storage.digest(working)
    assert copied_digest == sealed['adapter_digest']
    assert frozen() == frozen_before and ctx.storage.digest(ordinary()) == buffers_before
    del state, working
    fold = adaptation.fold(model, method)
    assert len(fold) == 48
    assert not any(n.endswith(('._ilora_slow_A', '._ilora_slow_B')) for n, _ in model.core.named_buffers())
    receipt = deployment.receipt(model, ctx.storage)
    assert receipt['layout']['weights'] == dict(coefficients=893675552, bytes=1787351104)
    assert receipt['layout']['ordinary_buffers']['bytes'] == 9032
    assert receipt['layout']['trainable'] == 0
    ctx.bind(model)
    proof = dict(method=name, stage=stage, initial_checkpoint_path=str(parent['initial']),
                 initial_checkpoint_declared_sha256=train['initial_checkpoint']['sha256'],
                 initial_container_rehashed=False, initial_complete_state_checked=True,
                 initial_reload=initial_reload, source_train_complete_sha256=ctx.sha(parent['train_path']),
                 source_verified_sha256=ctx.sha(parent['verified_path']),
                 snapshot_path=str(snapshot), snapshot_sha256=snapshot_sha,
                 stage_proof_path=str(stage_path), stage_proof_sha256=stage_sha,
                 stage_adapter_digest=sealed['adapter_digest'], copied_adapter_digest=copied_digest,
                 working_parameters=PARAMETERS, coordinates=COORDINATES, adapter_bytes=ADAPTER_BYTES,
                 keys=sorted(expected), shapes=shapes,
                 copy_not_add=True, same_parameter_objects=True, frozen_unchanged=True,
                 ordinary_buffers_unchanged=True, no_EMA_state_restored=True,
                 temporary_slow_buffers_removed=True, fold=fold, reloaded=receipt,
                 fresh_initial_each_stage=True, cumulative_fold=False)
    return model, proof


def release(ctx):
    import torch
    ctx.bridge.LAST_NATIVE = None
    ctx.bridge.ACTIVE = None
    # Remaining allocator/workspace bytes are descriptive, never an empty-GPU gate.
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ready = read(RUN / 'READY.json')
    source = Path(ready['arc_source_dir'])
    assert source.is_absolute()
    common = load_module('acquisition_frozen_arc_common', source / 'common.py')
    common.LIMITS[PHASE] = 1800
    ctx = common.Context(PHASE)
    ctx.sha, ctx.asset = common.sha, common.ASSET
    assert ctx.gpu == 1
    for filename in ('common.py', 'deployment.py', 'adaptation.py', 'readouts.py'):
        path = source / filename
        assert str(path) in ctx.ready['sources'] and common.sha(path) == ctx.ready['sources'][str(path)]
    deployment = load_module('acquisition_frozen_deployment', source / 'deployment.py')
    adaptation = load_module('acquisition_frozen_adaptation', source / 'adaptation.py')
    readouts = load_module('acquisition_frozen_readouts', source / 'readouts.py')
    adaptation.configure_gpu_math()
    parent = parents(ctx)
    chosen = [(task, row) for task, row in ctx.rows if task['domain'] in DOMAINS]
    assert len(chosen) == len({row['id'] for _, row in chosen}) == 40
    assert all(sum(task['domain'] == d for task, _ in chosen) == 20 for d in DOMAINS)
    assert all(2 <= task['task'] <= 11 if task['domain'] == 'medical'
               else 12 <= task['task'] <= 15 for task, _ in chosen)
    selected = [dict(id=row['id'], task=task['task'], domain=task['domain'],
                     generation_cap=task['max_new_tokens']) for task, row in chosen]
    assert selected == ctx.ready['selected_cases']
    files, controls = [], {}
    # This entire loop completes before the first call to _generate.
    for name in METHODS:
        model, proof = reconstruct(ctx, name, 19, parent[name], deployment, adaptation)
        same_receipt(proof['reloaded'], parent[name]['train']['checkpoint'])
        invariant = deployment.finish(model, proof['reloaded'], ctx.storage)
        controls[name] = dict(proof, invariant=invariant, full_final_receipt_exact=True,
                              new_generations=0, FIT_backwards=0, optimizer_updates=0)
        del model
        release(ctx)
    common.write('S19_CONTROLS.json', controls)
    files.append(RUN / 'S19_CONTROLS.json')
    records, stage_events = [], []
    for name in METHODS:
        for stage in sorted({task['task'] for task, _ in chosen}):
            rows = [(task, row) for task, row in chosen if task['task'] == stage]
            model, proof = reconstruct(ctx, name, stage, parent[name], deployment, adaptation)
            batcher = common.Batcher(ctx, model, 'ACQUIRED_DEV')
            model.eval()
            for task, row in rows:
                batch, binding = readouts._batch(model, task, row, ctx.storage, batcher.readout)
                tokens = readouts._generate(model, batch, task['max_new_tokens'], ctx.stop)
                records.append(dict(binding, method=name, id=row['id'], task=task['task'],
                                    domain=task['domain'], acquired_stage=stage, phase='ACQUIRED_DEV',
                                    prediction=ctx.processor.tokenizer.decode(tokens, skip_special_tokens=True),
                                    references=row['answers'], generation_cap=task['max_new_tokens'],
                                    output_tokens=tokens))
                del batch, tokens
            invariant = deployment.finish(model, proof['reloaded'], ctx.storage)
            filename = f'{name}_S{stage:02}_RECONSTRUCTION.json'
            common.write(filename, dict(proof, invariant=invariant,
                                       new_generations=len(rows), new_scores=0,
                                       FIT_backwards=0, optimizer_updates=0,
                                       readout_IDs=[row['id'] for _, row in rows],
                                       input_records=batcher.records))
            files.append(RUN / filename)
            stage_events.append(dict(method=name, stage=stage, new_generations=len(rows),
                                     proof=filename, sha256=common.sha(RUN / filename)))
            del batcher, model
            release(ctx)
    assert len(records) == len({(x['method'], x['id']) for x in records}) == 80
    common.write('acquired_predictions.json', records)
    files.append(RUN / 'acquired_predictions.json')
    ctx.end(dict(new_generations=80, new_scores=0, NLL_requests=0, FIT_backwards=0,
                 optimizer_updates=0, TEST_QA=0, S19_new_generations=0,
                 initial_new_generations=0, final_new_generations=0,
                 methods=list(METHODS), domains=list(DOMAINS), stage_events=stage_events,
                 both_S19_controls_before_any_new_generation=True,
                 frozen_parent_runs_and_sources_unchanged=True,
                 endpoint='working adapters reconstructed from each own initial full graph; GPU FP32 fold to FP16'), files)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        path = RUN / (PHASE + '_FAILURE.json')
        if not path.exists():
            path.write_text(json.dumps(dict(unix=time.time(), traceback=traceback.format_exc(),
                                           no_quality_claim=True), indent=2) + '\n', encoding='utf-8')
        raise
