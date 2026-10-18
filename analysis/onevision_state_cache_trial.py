"""Actual continuous LoRA training + bounded functional states on GPU1.

Independent version, no edits to any old active/queued trainer. All19 fresh
trained states must exactly equal the original native SeqFT checkpoints.
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
import traceback
import torch
import torch.nn.functional as F

import onevision_state_cache_core as core
import onevision_learned_reuse_trial as old
import onevision_feature_identity_repair as feature_repair
from onevision_knowledge_intervention import reset_seed, case_complete
from onevision_research_worker import atomic_torch, digest
from onevision_capacity_probe import finish_resources, stratify_full_results
from run import hashed, write

ROOT, BASE = old.ROOT, old.BASE
REPORT = BASE/'functional_state_cache_20261018/v1'
FAMILY = ROOT/'exp/onevision_state_cache_20261018/v1'


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k][v], b[k][v]) for k in a for v in ('A', 'B'))


def plan():
    previous, values = old.plan()
    manifest, data, domains, tests, full, native, native_proof = values
    screen_path = BASE/'functional_state_cache_20261018/TRAIN_SCREEN.json'
    screen = json.loads(screen_path.read_text())
    arrival = screen['arrival_stage']  # Only IDs/arrival, never use its loss matrix for decisions.
    train = {r['id']: (t, r) for t in range(1, 20) for r in data[t, 'train']}
    assert len(train) == 1313 and len(arrival) == 20
    assert all(i in train and train[i][0] == t for i, t in arrival.items())
    prototype_ids = {str(t): [r['id'] for r in sorted(data[t, 'train'],
                        key=lambda r: hashed(r, 'functional-state-prototype'))[:4]] for t in range(1, 20)}
    sources = dict(previous['source_sha256'])
    for n in ['onevision_state_cache_core.py', 'onevision_state_cache_trial.py',
              'onevision_feature_identity_repair.py', 'onevision_state_cache_report.py', 'answer_f1.py']:
        sources['analysis/'+n] = digest(ROOT/'analysis'/n)
    spec = {'source_sha256': sources, 'protocol_sha256': digest(REPORT/'PROTOCOL.md'),
            'manifest_sha256': previous['manifest_sha256'], 'model_revision': previous['model_revision'],
            'native_completion_sha256': previous['native_completion_sha256'],
            'TRAIN_ids': previous['train_ids'], 'TRAIN_QA': 1313, 'stages': 19, 'seed': 42,
            'rank': 8, 'trainable_LoRA_parameters': 540672, 'main_updates': 93,
            'one_continuous_main_TRAIN_stream': True, 'each_stage_exact_native_AB_required': True,
            'TRAIN_memory_arrivals': arrival, 'prototype_TRAIN_ids': prototype_ids,
            'Kmax': core.CAP, 'penalty_nats': core.PENALTY, 'kernel_temperature': core.TEMPERATURE,
            'router_trainable_parameters': 0, 'views': list(core.VIEWS),
            'full_candidates': ['cache_geometry', 'cache_quality'],
            'all_candidates_full21454_regardless_diagnostic': True,
            'TEST_selection_or_hyperparameter_choice': False,
            'nativeAnyRes_BF16_not_formalV2': True,
            'generation_ids': previous['generation_ids'], 'source_screen_sha256': digest(screen_path),
            'control_identity': 'seq_geometry == seq_quality == original native SeqFT, after original route fields removed',
            'extra_TRAIN_NLL_and_features_and_all_memory_charged': True}
    return spec, values


class CacheExperiment(old.MetricExperiment):
    def __init__(self, manifest, run):
        self.calling = None
        super().__init__(manifest, run)
        self.feature_gpu_gate()

    def feature(self, row):
        return fixed_feature(self, row)

    def feature_gpu_gate(self):
        # Reproduce the exact known auxiliary-ID failure and its single-ID fix.
        original = old.MetricExperiment.feature
        _, data, *_ = old.freeze()
        compatible, failing = data[1, 'train'][0], data[16, 'test'][400]
        assert failing['id'] == 'FinMME:5082'
        rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        versions = [(id(p), p._version) for p in self.model.parameters()]
        self.feature_cache.clear()
        a = original(self, compatible).clone()
        self.feature_cache.clear()
        b = self.feature(compatible)
        assert torch.equal(a, b)
        self.feature_cache.clear()
        try:
            original(self, failing)
        except KeyError as error:
            assert error.args == ('id',)
        else:
            raise AssertionError('Known feature failure did not reproduce')
        value = self.feature(failing)
        assert value.numel() == 1792 and torch.isfinite(value).all()
        assert [(id(p), p._version) for p in self.model.parameters()] == versions
        assert torch.equal(torch.get_rng_state(), rng) and torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        self.feature_cache.clear()
        write(self.run/'feature_GPU_gate.json', {'passed': True, 'known_failure_reproduced': failing['id'],
            'compatible_feature_exact': True, 'parameter_versions_and_RNG_exact': True,
            'additional_feature_calls': 4, 'optimizer_updates': 0, 'TEST_answers_used': False,
            'source_sha256': digest(Path(__file__)), 'repair_source_sha256': digest(Path(feature_repair.__file__))})

    def set_calling(self, state=None, view=None):
        self.calling = (state, view) if state is not None else None
        self.active_key = None

    def activate_route(self, row):
        if self.calling is None:
            self.last_route = {'route_mode': 'fixed_training_or_TRAIN_probe'}
            return
        state, view = self.calling
        keys = [19] if view.startswith('seq_') else sorted(state['pool'])
        if len(keys) == 1:
            chosen, details = keys[0], {'single_state_identity': True}
        else:
            chosen, details = core.route(self.feature(row), keys,
                torch.stack([state['prototypes'][k] for k in keys]), state['memory_features'],
                torch.tensor([[state['losses'][k][i] for i in state['memory_ids']] for k in keys]),
                view.endswith('_quality'))
        if self.active_key != chosen:
            self.restore(state['pool'][chosen])
            self.active_key = chosen
        self.last_route = {'route_mode': view, 'route_state': chosen, 'route_details': details}


fixed_feature, _ = feature_repair.compiled_feature(old.MetricExperiment.feature.__globals__)


def semantic(rows, strip_routes=True):
    return [{k: v for k, v in r.items() if k != 'generation_seconds' and
             (not strip_routes or not k.startswith('route_'))} for r in rows]


def rows_at(run, label, domain):
    return [json.loads(line) for line in (run/f'evaluations/{label}/{domain}_image.jsonl').read_text().splitlines()]


def complete(phase, view, spec):
    name = 'build' if phase == 'build' else 'full_'+view
    p = REPORT/(name+'_complete.json')
    if not p.exists():
        return False
    proof = json.loads(p.read_text())
    assert proof['complete'] and proof['spec'] == spec
    run = ROOT/proof['run']
    assert run.resolve().is_relative_to(FAMILY.resolve())
    for n, expected in proof['artifact_sha256'].items():
        p = run/n
        assert p.resolve().is_relative_to(run.resolve()) and digest(p) == expected
    assert json.loads((run/'original_exact_gate.json').read_text())['passed']
    if phase == 'full':
        assert proof['full_QA'] == 21454 and json.loads((run/'candidate_reload_exact.json').read_text())['passed']
    return True


def run_trial(phase, view=None):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    reset_seed()
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = (REPORT/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec, values = plan()
    assert json.loads((REPORT/'CPU_PLAN_CHECK.json').read_text())['spec'] == spec
    assert phase == 'build' or view in spec['full_candidates']
    if complete(phase, view, spec):
        return
    name = 'build' if phase == 'build' else 'full_'+view
    assert not (REPORT/(name+'_needs_attention.json')).exists(), 'No blind retry'
    pointer = REPORT/(name+'_attempt.json')
    assert not pointer.exists(), 'Do not resume or splice an incomplete new run'
    manifest, data, domains, tests, full, native, native_proof = values
    train = {r['id']: r for t in range(1, 20) for r in data[t, 'train']}
    run = FAMILY/name/time.strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    write(pointer, {'spec': spec, 'run': str(run.relative_to(ROOT)), 'pid': os.getpid()})
    write(run/'protocol.json', spec)
    x = CacheExperiment(manifest, run)
    initial = torch.load(native/'initial_adapter.pt', map_location='cpu', weights_only=True)['adapter']
    assert same(x.original_initial, initial)
    generation = {d: [r for r in rs if r['id'] in set(spec['generation_ids'][d])] for d, rs in tests.items()}
    assert sum(map(len, generation.values())) == 155

    def evaluate(label, selected, generated, state, which):
        x.set_calling(state, which)
        assert not case_complete(run, label, selected), 'Each view fresh within a new process'
        x.evaluate(label, selected, generated)
        assert case_complete(run, label, selected)

    if phase == 'build':
        current, pool, prototypes, losses, features = initial, {}, {}, {}, {}
        arrival = spec['TRAIN_memory_arrivals']
        nll_calls, feature_calls, events = [], [], []
        train_windows = []
        x.monitor.snapshot('continuous_method_before', model=x.model)
        for stage in range(1, 20):
            x.set_calling()
            x.configure(current, finetune_shared=True)
            x.cache.clear()
            reset_seed()  # Original deterministic native fit resets at every task.
            torch.cuda.reset_peak_memory_stats()
            before = x.monitor.snapshot(f'pure_TRAIN_T{stage:02d}_before', model=x.model)
            current = x.train(f'continuous_T{stage:02d}', data[stage, 'train'])
            after = x.monitor.snapshot(f'pure_TRAIN_T{stage:02d}_after', model=x.model)
            train_windows.append({'stage': stage, 'before': before, 'after': after,
                'cuda_peak_allocated_bytes': after['gpu_peak_allocated_bytes'],
                'cuda_peak_reserved_bytes': after['gpu_peak_reserved_bytes']})
            expected_path = native/f'adapters/sequential_T{stage:02d}.pt'
            assert digest(expected_path) == native_proof['artifact_sha256'][str(expected_path.relative_to(native))]
            expected = torch.load(expected_path, map_location='cpu', weights_only=True)['adapter']
            assert same(current, expected), f'Actual TRAIN trajectory differs at stage{stage}'
            del expected
            write(run/f'exact_gates/T{stage:02d}.json', {'passed': True, 'all48_AB_bitwise_equal': True,
                'native_state_sha256': digest(expected_path), 'stage': stage})
            pool[stage] = current
            x.model.eval()
            proto_rows = [train[i] for i in spec['prototype_TRAIN_ids'][str(stage)]]
            vectors = []
            for row in proto_rows:
                vectors.append(x.feature(row))
                feature_calls.append({'stage': stage, 'id': row['id'], 'purpose': 'current_prototype'})
            prototypes[stage] = F.normalize(torch.stack(vectors).mean(0), dim=0)
            ids = [i for i, t in arrival.items() if t <= stage]
            for ident in ids:
                if ident not in features:
                    features[ident] = x.feature(train[ident]).clone()
                    feature_calls.append({'stage': stage, 'id': ident, 'purpose': 'arrived_memory'})
            for state_id in sorted(pool):
                losses.setdefault(state_id, {})
                x.restore(pool[state_id])
                for ident in ids:
                    if ident not in losses[state_id]:
                        with torch.no_grad():
                            loss, _ = x.loss(train[ident])
                        value = float(loss)
                        assert math.isfinite(value) and arrival[ident] <= stage and state_id <= stage
                        losses[state_id][ident] = value
                        nll_calls.append({'decision_stage': stage, 'state': state_id, 'TRAIN_id': ident, 'NLL': value})
            selected, objective = core.retain(stage, {k: [losses[k][i] for i in ids] for k in pool})
            event = {'stage': stage, 'selected_states': selected, 'candidate_states': sorted(pool),
                     'arrived_TRAIN_ids': ids, 'temporary_candidate_states': len(pool), **objective}
            events.append(event)
            pool = {k: pool[k] for k in selected}
            prototypes = {k: prototypes[k] for k in selected}
            losses = {k: losses[k] for k in selected}
            state = {'stage': stage, 'pool': pool, 'prototypes': prototypes, 'losses': losses,
                     'memory_ids': ids, 'memory_features': torch.stack([features[i] for i in ids])}
            atomic_torch(run/f'pool/stage_{stage:02d}.pt', state)
            write(run/'decisions.json', {'events': events, 'actual_TRAIN_NLL_calls': nll_calls,
                'logical_feature_calls': feature_calls, 'actual_metric_feature_misses': x.metric_feature_misses,
                'router_parameters': 0, 'pool_parameter_count': len(pool)*540672,
                'pool_tensor_bytes': sum(v.numel()*v.element_size() for a in pool.values() for layer in a.values() for v in layer.values()),
                'source_parent_training_repeated_and_charged': True})
            print(json.dumps({'event': 'continuous_cache_stage', **event}), flush=True)
        atomic_torch(run/'final_state.pt', state)
        write(run/'pure_TRAIN_windows.json', {'windows': train_windows,
            'scope': 'original main optimizer updates and raw-gradient persistence; excludes selection NLL/prototypes/TEST'})
        x.configure(current, finetune_shared=False)
        x.monitor.snapshot('continuous_method_after', model=x.model)
        for which in core.VIEWS:
            evaluate(which, tests, generation, state, which)
        for domain in tests:
            control = semantic(rows_at(run, 'seq_geometry', domain))
            assert control == semantic(rows_at(native, 'sequential_T19', domain))
            assert control == semantic(rows_at(run, 'seq_quality', domain))
        write(run/'original_exact_gate.json', {'passed': True, 'QA': 500, 'generated': 155,
            'same_native_final_and_both_single_state_views': True, 'no_tolerance': True,
            'route_metadata_only_removed': True, 'all19_actual_training_AB_exact': True})
        finish_resources(x, run, 'continuous_method_before', 'continuous_method_after',
                         'same-PID main TRAIN+functional selection; see per-stage pure TRAIN windows; global includes TEST')
    else:
        assert complete('build', None, spec)
        parent = json.loads((REPORT/'build_complete.json').read_text())
        build = ROOT/parent['run']
        state = torch.load(build/'final_state.pt', map_location='cpu', weights_only=True)
        assert 19 in state['pool'] and len(state['pool']) <= core.CAP and len(state['memory_ids']) == 20
        x.configure(state['pool'][19], finetune_shared=False)
        x.monitor.snapshot('cache_inference_before', model=x.model)
        evaluate('candidate_reload', tests, generation, state, view)
        for domain in tests:
            assert semantic(rows_at(run, 'candidate_reload', domain), False) == semantic(rows_at(build, view, domain), False)
        write(run/'candidate_reload_exact.json', {'passed': True, 'QA': 500, 'generated': 155, 'no_tolerance': True})
        write(run/'original_exact_gate.json', {'passed': True, 'build_proof_sha256': digest(REPORT/'build_complete.json')})
        evaluate('final_full_input_router', full, full, state, view)
        stratify_full_results(run, full, {r['id'] for rs in tests.values() for r in rs})
        x.monitor.snapshot('cache_inference_after', model=x.model)
        finish_resources(x, run, 'cache_inference_before', 'cache_inference_after',
                         'fresh-process inference only; parent93 main updates and TRAIN query cost charged separately')
    files = [p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.pt')]
    write(REPORT/(name+'_complete.json'), {'complete': True, 'spec': spec, 'run': str(run.relative_to(ROOT)),
        'phase': phase, 'view': view, 'main_updates': 93 if phase == 'build' else 0,
        'parent_main_updates_charged': 93, 'full_QA': 21454 if phase == 'full' else None,
        'artifact_sha256': {str(p.relative_to(run)): digest(p) for p in files}})
    assert complete(phase, view, spec)


def check():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    count = core.checks()
    spec, values = plan()
    assert sum(math.ceil(len(v)/16) for (t, split), v in values[1].items() if split == 'train') == 93
    screen = json.loads((BASE/'functional_state_cache_20261018/TRAIN_SCREEN.json').read_text())
    # Integration check only: do not rerun the already completed science audit.
    for cap in (1, 3):
        pool = []
        for stage in range(1, 20):
            ids = [i for i, t in spec['TRAIN_memory_arrivals'].items() if t <= stage]
            candidates = sorted(set(pool)|{stage})
            losses = {s: [screen['actual_saved_TRAIN_NLL'][str(s)][i] for i in ids] for s in candidates}
            pool, _ = core.retain(stage, losses, cap)
            assert pool == screen['traces'][str(cap)][stage-1]['selected_states']
            count += 1
    write(REPORT/'CPU_PLAN_CHECK.json', {'passed': True, 'CPU_only': True, 'checks': count,
        'spec': spec, 'GPU_started': False, 'performance_evidence': False})
    print(json.dumps({'passed': True, 'CPU_only': True, 'checks': count, 'GPU_started': False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--phase', choices=['build', 'full'])
    p.add_argument('--view', choices=core.VIEWS)
    args = p.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        if args.check:
            check()
        else:
            assert args.phase is not None
            run_trial(args.phase, args.view)
    except Exception:
        label = 'CPU' if args.check else 'build' if args.phase == 'build' else 'full_'+str(args.view)
        write(REPORT/(label+'_needs_attention.json'), {'traceback': traceback.format_exc(), 'no_blind_retry': True})
        raise
