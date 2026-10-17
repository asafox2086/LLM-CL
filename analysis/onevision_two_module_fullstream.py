"""Full19-stage training for the two-module research ablations, new source.

Shared frozen1313 TRAIN,155 diagnostic generations;21454 confirmation is a
separate explicit inference phase. Neither cohort is substituted for the other.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import fcntl
import gc
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
sys.path.insert(0, str(ROOT/'exp/onevision_diagnostics_20261016'))
from run import freeze, hashed, write
from onevision_knowledge_intervention import reset_seed, case_complete
from onevision_research_worker import atomic_torch, digest
from onevision_full_input_repair import RepairedExperiment
from onevision_two_module_pilot import Pilot, VARIANTS
from onevision_two_module_core import TrainLoss, choose_reuse
from onevision_two_module_engine import epoch, isolated
from answer_f1 import answer_f1

REPORT = ROOT/'summary/onevision_diagnostics_20261016/two_module_fullstream'


def plan():
    manifest, data, domains, tests, _, _, protocol = freeze()
    native = json.loads((ROOT/'summary/onevision_diagnostics_20261016/deterministic_native/complete.json').read_text())
    assert native['complete']
    for path, sha in native['source_sha256'].items():
        assert digest(ROOT/path) == sha
    directory = ROOT/native['run'];native_protocol = json.loads((directory/'protocol.json').read_text())
    assert digest(directory/'protocol.json') == native['artifact_sha256']['protocol.json']
    generation = {domain: [next(r for r in tests[domain] if r['id'] == identity) for identity in ids]
                  for domain, ids in native_protocol['generation_ids'].items()}
    current = {t: sorted(data[t, 'train'], key=lambda r: hashed(r, 'two-module-current'))[:8] for t in range(1, 20)}
    memories = {t: sorted(data[t, 'train'], key=lambda r: hashed(r, 'two-module-old'))[:2] for t in range(1, 20)}
    engine_gate = json.loads((REPORT/'CPU_ENGINE_VALIDATION.json').read_text())
    assert engine_gate['passed'] and engine_gate['CPU_only']
    for path, sha in engine_gate['source_sha256'].items():
        assert digest(ROOT/path) == sha
    metric = json.loads((ROOT/'summary/primary_metric_20261017/metric_lock.json').read_text())
    assert digest(ROOT/'analysis/answer_f1.py') == metric['metric_sha256'] and metric['option_mapping'] is False
    files = [Path(__file__), ROOT/'analysis/onevision_two_module_engine.py',
             ROOT/'analysis/onevision_two_module_core.py', ROOT/'analysis/onevision_two_module_pilot.py',
             ROOT/'analysis/onevision_full_input_repair.py', ROOT/'analysis/onevision_capacity_probe.py',
             ROOT/'analysis/onevision_knowledge_intervention.py', ROOT/'analysis/onevision_research_worker.py',
             ROOT/'exp/onevision_diagnostics_20261016/run.py',
             ROOT/'exp/five_domain_native_cl_20261016_sparse/runtime_utils.py',
             ROOT/'exp/five_domain_native_cl_20261016_sparse/resource_monitor.py', ROOT/'analysis/answer_f1.py']
    initial = directory/'initial_adapter.pt'
    assert digest(initial) == native['artifact_sha256']['initial_adapter.pt']
    spec = {'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in files},
            'manifest_sha256': protocol['source_manifest_sha256'], 'native_reference_run': native['run'],
            'native_reference_complete_sha256': digest(ROOT/'summary/onevision_diagnostics_20261016/deterministic_native/complete.json'),
            'initial_adapter_sha256': digest(initial), 'variants': VARIANTS,
            'metric_version': metric['version'], 'metric_sha256': metric['metric_sha256'],
            'base_revision': '74dd0bf867a4cda7950c17663794267c60cf4b40',
            'seed': 42, 'rank': 8, 'alpha': 32, 'trainable_parameters': 540672, 'epochs': 1, 'batch': 16,
            'target_TRAIN_QA': sum(len(data[t, 'train']) for t in range(1, 20)), 'full_TEST_QA': 21454,
            'TRAIN_ids': {str(t): [r['id'] for r in data[t, 'train']] for t in range(1, 20)},
            'training_order_ids': {str(t): [r['id'] for r in sorted(data[t, 'train'], key=lambda r: hashed(r, 'training'))] for t in range(1, 20)},
            'current_probe_ids': {str(t): [r['id'] for r in current[t]] for t in current},
            'old_memory_ids': {str(t): [r['id'] for r in memories[t]] for t in memories},
            'diagnostic_test_ids': {d: [r['id'] for r in rows] for d, rows in tests.items()},
            'diagnostic_generation_ids': {d: [r['id'] for r in rows] for d, rows in generation.items()},
            'old_loss_budget_nats': .02, 'reference': 'own-task acquisition TRAIN memory NLL, not cumulative step-relative budget',
            'first_task': 'A/B both bypassed; shared actual original SeqFT stage1 exact gate',
            'validity': 'At least one positive actual parameter update each native stage; otherwise whole variant invalid for CL ranking.',
            'mode': 'Research nativeAnyRes; NOT shared formal V2 resized input protocol',
            'full_confirmation': 'separate frozen-state current+next/final inference; no full FWT/BWT from155 diagnostic scores'}
    assert spec['target_TRAIN_QA'] == 1313 and sum(map(len, generation.values())) == 155
    train_ids = {r['id'] for t in range(1, 20) for r in data[t, 'train']}
    test_ids = {r['id'] for t in range(1, 20) for r in data[t, 'test']}
    assert len(train_ids) == 1313 and len(test_ids) == 21454 and not train_ids & test_ids
    return manifest, data, domains, tests, generation, current, memories, json.loads(json.dumps(spec))


class Stream(Pilot):
    def train_evidence(self, state, current, memory):
        mode = self.model.training
        try:
            self.restore(state);self.model.eval()
            with torch.no_grad():
                values = [float(self.loss(row)[0]) for row in current]
                old = {task: [float(self.loss(row)[0]) for row in rows] for task, rows in memory.items()}
            self.probe_QA_forwards += len(current)+sum(map(len, memory.values()))
            return TrainLoss(sum(values)/len(values), tuple(r['id'] for r in current)), {
                task: TrainLoss(sum(values)/len(values), tuple(r['id'] for r in memory[task]))
                for task, values in old.items()}
        finally:
            self.model.train(mode)


def checkpoint(path):
    meta = json.loads(path.with_suffix('.json').read_text())
    assert digest(path) == meta['sha256']
    # Local source-checked snapshots contain NumPy/Python RNG, not external pickle.
    return torch.load(path, map_location='cpu', weights_only=False)


def recoverable_checkpoint(path):
    """A committed file without its atomic SHA sidecar is unproven, not usable."""
    if not path.exists():
        return False
    if not path.with_suffix('.json').exists():
        return False
    meta = json.loads(path.with_suffix('.json').read_text())
    assert digest(path) == meta['sha256'], 'Checkpoint corruption; preserve and stop'
    return True


def verify_complete(variant, spec):
    path = REPORT/(variant+'_complete.json')
    if not path.exists():
        return False
    proof = json.loads(path.read_text());assert proof['complete'] and proof['spec'] == spec
    directory = ROOT/proof['run']
    assert directory.resolve().is_relative_to((ROOT/'exp/onevision_two_module_20261017/fullstream').resolve())
    for name, sha in proof['artifact_sha256'].items():
        artifact = directory/name
        assert artifact.resolve().is_relative_to(directory.resolve()) and digest(artifact) == sha
    assert proof['trained_stages'] == 19 and proof['TRAIN_QA'] == 1313
    return True


def memory_window(directory, before, after):
    keys = ['VmRSS_bytes', 'Pss_bytes', 'cpu_uss_bytes', 'gpu_allocated_bytes',
            'gpu_reserved_bytes', 'gpu_process_nvml_bytes']
    rows = [json.loads(line) for line in (directory/'resources.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['pid'] == before['pid'] and before['unix'] <= r['unix'] <= after['unix']]
    return {'unit': 'bytes', 'before': before, 'after': after,
            'growth_bytes': {k: after[k]-before[k] if before.get(k) is not None and after.get(k) is not None else None for k in keys},
            'sampled_training_peak_bytes': {k: max((r[k] for r in rows if r.get(k) is not None), default=None) for k in keys},
            'cuda_training_peak_allocated_bytes': after['gpu_peak_allocated_bytes'],
            'cuda_training_peak_reserved_bytes': after['gpu_peak_reserved_bytes'],
            'scope': 'Same PID native-stage training plus TRAIN probes; excludes pre/post TEST inference. CPU/NVML peaks sampled every5s.'}


def check_evaluation(directory, label, selected, generated):
    assert case_complete(directory, label, selected)
    for domain, expected in selected.items():
        path = directory/'evaluations'/label/f'{domain}_image.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        gids = {r['id'] for r in generated[domain]}
        for row, source in zip(rows, expected):
            assert row['id'] == source['id'] and row['references'] == source['answers']
            assert math.isfinite(row['nll'])
            assert ('prediction' in row) == (row['id'] in gids)
            if 'prediction' in row:
                assert answer_f1(row['prediction'], row['references']) == row['scores']['token_f1']


def finish_stream(x):
    x.monitor.snapshot('training_after', model=x.model);x.monitor.finish(x.model)
    windows = {p.parent.name: json.loads(p.read_text()) for p in (x.run/'stages').glob('T*/training_memory.json')}
    rows = [json.loads(line) for line in (x.run/'resources.jsonl').read_text().splitlines()]
    keys = ['VmRSS_bytes', 'Pss_bytes', 'cpu_uss_bytes', 'gpu_allocated_bytes',
            'gpu_reserved_bytes', 'gpu_process_nvml_bytes']
    # Never subtract a pre-reboot PID's allocation from a resumed process.
    per_pid = {str(pid): {key: max((r[key] for r in rows if r['pid'] == pid and r.get(key) is not None), default=None)
                         for key in keys} for pid in {r['pid'] for r in rows}}
    write(x.run/'memory_summary.json', {'unit': 'bytes', 'per_stage_training_windows': windows,
          'per_PID_sampled_overall_peaks_including_TEST': per_pid,
          'cross_PID_growth_not_defined': True,
          'note': 'Use each same-PID stage before/after/growth for training. Overall peaks include inference; sampled CPU/NVML may miss short spikes. Temporary clones and logged CPU optimizer states are billed; model counts separate from process RAM.'})


def evaluate(x, label, selected, generated, state_sha, spec):
    folder = x.run/'evaluations'/label;folder.mkdir(parents=True, exist_ok=True)
    metadata = {'state_sha256': state_sha, 'source_sha256': spec['source_sha256'],
                'ids': {d: [r['id'] for r in rows] for d, rows in selected.items()},
                'generation_ids': {d: [r['id'] for r in rows] for d, rows in generated.items()}}
    marker = folder/'inference_protocol.json'
    if marker.exists():
        assert json.loads(marker.read_text()) == metadata
    else:
        write(marker, metadata)
    if case_complete(x.run, label, selected):
        check_evaluation(x.run, label, selected, generated)
        return
    for domain in selected:
        path = folder/f'{domain}_image.jsonl'
        if not path.exists():
            continue
        lines = path.read_text().splitlines();valid = []
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                assert index == len(lines)-1
                archive = folder/'interrupted_tail';archive.mkdir(exist_ok=True)
                shutil.copy2(path, archive/(str(time.time_ns())+'_'+path.name))
                path.write_text('\n'.join(valid)+ ('\n' if valid else ''))
                break
            expected = selected[domain][index]
            assert row['id'] == expected['id'] and row['references'] == expected['answers']
            valid.append(line)
    x.model.eval();x.evaluate(label, selected, generated)
    check_evaluation(x.run, label, selected, generated)


def worker(variant):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    manifest, data, domains, tests, generation, current, memories, spec = plan()
    assert spec == json.loads((REPORT/'CPU_FULLSTREAM_PLAN.json').read_text())['spec']
    if verify_complete(variant, spec):
        return
    assert shutil.disk_usage(ROOT).free > 12*1024**3
    pointer = REPORT/(variant+'_attempt.json')
    if pointer.exists():
        info = json.loads(pointer.read_text());assert info['spec'] == spec
        directory = ROOT/info['run']
    else:
        directory = ROOT/'exp/onevision_two_module_20261017/fullstream'/variant/time.strftime('%Y%m%d_%H%M%S')
        directory.mkdir(parents=True)
        write(pointer, {'run': str(directory.relative_to(ROOT)), 'spec': spec})
        write(directory/'protocol.json', spec)
    assert not (directory/'invalid_training.json').exists(), 'Invalid variant needs a new intervention/version, not blind retry'
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    x = Stream(manifest, directory);x.probe_QA_forwards = 0
    write(directory/'processes'/f'{os.getpid()}_{time.time_ns()}.json', {'pid': os.getpid(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
          'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()})
    native_proof = json.loads((ROOT/'summary/onevision_diagnostics_20261016/deterministic_native/complete.json').read_text())
    reference = ROOT/spec['native_reference_run']
    initial = torch.load(reference/'initial_adapter.pt', map_location='cpu', weights_only=True)['adapter']
    x.configure(initial, finetune_shared=True);x.model.eval()
    RepairedExperiment.forward_gate(x)
    initial_path = directory/'initial_adapter.pt'
    if recoverable_checkpoint(initial_path):
        old_initial = checkpoint(initial_path)
        assert old_initial['spec'] == spec
        assert all(torch.equal(initial[n][f], old_initial['adapter'][n][f]) for n in initial for f in ('A', 'B'))
    else:
        atomic_torch(initial_path, {'adapter': initial, 'spec': spec})
    initial_sha = digest(initial_path)
    write(directory/'model_counts.json', {'total_parameters': sum(p.numel() for p in x.model.parameters()),
          'trainable_parameters': 540672, 'retained_LoRA_modules': 1, 'rank': 8,
          'adaptive_module_count_tested': False, 'temporary_candidates_billed': True})
    evaluate(x, 'base_initial', tests, generation, initial_sha, spec)
    state = initial;old_references = {}
    for task in manifest['tasks']:
        stage, domain = task['task'], task['domain']
        folder = directory/'stages'/f'T{stage:02d}';saved = folder/'trained.pt'
        selected_test = {domain: tests[domain]}
        generated = {domain: [r for r in generation[domain] if r['_task'] == stage]}
        assert generated[domain]
        x.configure(state, finetune_shared=True)
        previous_sha = initial_sha if stage == 1 else digest(directory/'stages'/f'T{stage-1:02d}'/'trained.pt')
        evaluate(x, f'pre_T{stage:02d}', selected_test, generated, previous_sha, spec)
        if recoverable_checkpoint(saved):
            completed = checkpoint(saved)
            assert completed['spec'] == spec and completed['stage'] == stage
            state, old_references = completed['adapter'], completed['old_references']
            x.restore(state)
        else:
            if folder.exists():
                quarantine = directory/'interrupted';quarantine.mkdir(exist_ok=True)
                folder.rename(quarantine/(folder.name+'_'+str(time.time_ns())))
            folder.mkdir(parents=True)
            write(REPORT/'status.json', {'phase': 'training', 'variant': variant, 'stage': stage,
                                        'pid': os.getpid(), 'run': str(directory.relative_to(ROOT))})
            memory = {str(t): memories[t] for t in range(1, stage)}
            memory_ids = {t: tuple(r['id'] for r in rows) for t, rows in memory.items()}
            current_ids = tuple(r['id'] for r in current[stage])
            evidence = lambda adapter: x.train_evidence(adapter, current[stage], memory)
            use_A, use_B = VARIANTS[variant]
            x.monitor.stage = stage;torch.cuda.reset_peak_memory_stats()
            before = x.monitor.snapshot(f'T{stage:02d}_TRAIN_before', stage, model=x.model)
            probe_before = x.probe_QA_forwards
            if stage > 1 and use_A:
                selected, choice = isolated(choose_reuse, state, evidence, current_ids,
                                            memory_ids, old_references, use_B)
            else:
                selected, choice = state, {'chosen_strength': 1., 'disabled': True}
            x.configure(selected, finetune_shared=True);x.cache.clear();reset_seed();x.model.train()
            named = [(n, p) for n, p in x.model.named_parameters() if p.requires_grad]
            assert sum(p.numel() for _, p in named) == 540672
            order = sorted(data[stage, 'train'], key=lambda r: hashed(r, 'training'))

            def save_update(event, raw, payload):
                atomic_torch(folder/f"update_{event['update']:03d}.pt", {
                    'raw_gradient': raw, 'layout': [(n, list(p.shape)) for n, p in named],
                    **payload, 'TRAIN_ids': event['train_ids']})
                with (folder/'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(event)+'\n');stream.flush();os.fsync(stream.fileno())

            trained = epoch(named, order, x.loss, x.snapshot, x.restore, evidence,
                            current_ids, memory_ids, old_references, use_B and stage > 1, save_update)
            state = x.snapshot()
            if variant == 'control' or stage == 1:
                reference_path = reference/'adapters'/f'sequential_T{stage:02d}.pt'
                assert digest(reference_path) == native_proof['artifact_sha256'][str(reference_path.relative_to(reference))]
                expected = torch.load(reference_path, map_location='cpu', weights_only=True)['adapter']
                assert all(torch.equal(state[n][f], expected[n][f]) for n in state for f in ('A', 'B')), 'Original19-stage exact gate FAILED'
                write(folder/'original_state_exact_gate.json', {'passed': True, 'stage': stage,
                                                               'all_A_B_exact': True, 'no_tolerance': True})
            # References remain own acquisition losses, never reset per update.
            _, own = isolated(x.train_evidence, state, memories[stage], {str(stage): memories[stage]})
            old_references = dict(old_references);old_references[str(stage)] = own[str(stage)].value
            gc.collect();torch.cuda.empty_cache()
            after = x.monitor.snapshot(f'T{stage:02d}_TRAIN_after', stage, model=x.model)
            write(folder/'training_memory.json', memory_window(directory, before, after))
            metadata = {k: v for k, v in trained.items() if k not in ('optimizer_state', 'RNG_after')}
            metadata.update(stage=stage, choice=choice, probe_QA_forwards=x.probe_QA_forwards-probe_before)
            write(folder/'training.json', metadata)
            atomic_torch(saved, {'stage': stage, 'adapter': state, 'old_references': old_references,
                                'optimizer_state': trained['optimizer_state'], 'RNG_after': trained['RNG_after'], 'spec': spec})
            if trained['effective_updates'] == 0:
                finish_stream(x)
                write(directory/'invalid_training.json', {'stage': stage, 'variant': variant,
                      'reason': 'No positive accepted parameter update; cannot rank as successful CL',
                      'source_sha256': spec['source_sha256']})
                write(REPORT/(variant+'_invalid.json'), {'run': str(directory.relative_to(ROOT)), 'stage': stage,
                                                       'formal_score_available': False, 'spec': spec,
                                                       'training_sha256': digest(folder/'training.json'),
                                                       'memory_sha256': digest(folder/'training_memory.json')})
                return
            del trained
        write(directory/'latest_stage.json', {'stage': stage, 'checkpoint': str(saved.relative_to(directory)),
                                             'sha256': digest(saved), 'spec_sha256': digest(directory/'protocol.json')})
        evaluate(x, f'acquired_T{stage:02d}', tests if stage == 19 else selected_test,
                 generation if stage == 19 else generated, digest(saved), spec)
    finish_stream(x)
    artifacts = [*directory.rglob('*.pt'), *directory.rglob('*.jsonl'), *directory.rglob('*.json')]
    write(REPORT/(variant+'_complete.json'), {'complete': True, 'run': str(directory.relative_to(ROOT)),
          'spec': spec, 'trained_stages': 19, 'TRAIN_QA': 1313, 'diagnostic_generated_QA': 155,
          'full21454_confirmed': False, 'artifact_sha256': {str(p.relative_to(directory)): digest(p) for p in artifacts}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--check', action='store_true')
    parser.add_argument('--variant', choices=list(VARIANTS));args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        if args.check:
            *_, spec = plan();assert not torch.cuda.is_initialized()
            write(REPORT/'CPU_FULLSTREAM_PLAN.json', {'passed': True, 'GPU_used': False, 'spec': spec})
            print('Full19-stage/1313TRAIN plan frozen; actual GPU gates and21454 confirmation pending')
        else:
            assert args.variant is not None
            lock = (REPORT/(args.variant+'.lock')).open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(args.variant)
    except Exception:
        write(REPORT/(str(args.variant)+'_needs_attention.json'), {'pid': os.getpid(), 'traceback': traceback.format_exc()})
        raise
