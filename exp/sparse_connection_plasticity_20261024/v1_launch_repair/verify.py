"""Unique CPU verification; no pretrained-model loading, generation or recovery."""
import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from common import (ASSETS, BASE_VALUES, CASES, FFN_SLOTS, FINAL_VALUES, PATTERN_BYTES, PORT,
    QV_COORDINATES, REPORT, WORK, check_ready, data, sha, write)

DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')


def once(name):
    path = REPORT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(dict(pid=os.getpid(), monotonic=time.monotonic()), stream)
        stream.flush(); os.fsync(stream.fileno())


def exited(phase):
    status = json.loads((REPORT / 'phases' / phase / 'EXIT.json').read_text())
    assert status['returncode'] == 0 and status['observed_wait_returned']
    complete = REPORT / 'phases' / phase / 'COMPLETE.json'
    assert status['complete_sha256'] == sha(complete)
    started_path = REPORT / 'phases' / phase / 'STARTED.json'
    assert status['started_sha256'] == sha(started_path)
    started = json.loads(started_path.read_text())
    result = json.loads(complete.read_text())
    assert result['worker'] == status['worker']
    assert all(started[key] == status['worker'][key] for key in ('pid', 'uid', 'startticks', 'pgid', 'boot_id'))
    assert started['ready_sha256'] == result['ready_sha256'] == sha(REPORT / 'READY.json')
    assert started['physical_GPU'] == json.loads((REPORT / 'DEVICE_SELECTION.json').read_text())['physical_GPU']
    assert started['budget_seconds'] == (3600 if phase == 'init' else 7200)
    return result


def resources(root, complete):
    rows = [json.loads(line) for line in (root / 'resources.jsonl').read_text().splitlines()]
    physical = str(json.loads((REPORT / 'DEVICE_SELECTION.json').read_text())['physical_GPU'])
    assert rows and all(row['pid'] == complete['worker']['pid'] and row['visible_gpu'] == physical for row in rows)
    assert rows[-1]['phase'] == 'phase_finished'
    summary = json.loads((root / 'memory_summary.json').read_text())
    assert summary == complete['memory'] and summary['pid'] == complete['worker']['pid']
    assert summary['samples'] == len(rows) and summary['before'] == rows[0] and summary['after'] == rows[-1]
    for key, value in summary['observed_peak_bytes'].items():
        expected = max(row[key] for row in rows if row.get(key) is not None) if any(row.get(key) is not None for row in rows) else None
        assert value == expected
    before, after = complete['cache_path_before'], complete['cache_path_after']
    for key in ('visual_cache_hits', 'visual_cache_misses'):
        assert 0 <= before[key] <= after[key]
    assert complete['cache_counts_are_not_total_visual_tower_calls']
    expected_native = 2 if complete['phase'] == 'init' else 19
    assert complete['native_core_bypass_NLL_paths'] == complete['native_core_bypass_short_generations'] == expected_native


def verify_initial():
    once('VERIFY_INITIAL_STARTED.json')
    ready = check_ready(); complete = exited('init')
    assert complete['complete'] and complete['phase'] == 'init'
    assert 0 < complete['model_RUN_seconds'] <= 3600
    assert complete['ready_sha256'] == sha(REPORT / 'READY.json')
    root = WORK / 'init'
    required = {'resources.jsonl', 'memory_summary.json', 'config.json', 'training_input_IDs.jsonl',
        'initialization/INITIAL_FUNCTION.json', *[f'initialization/block_{i:02d}.json' for i in range(1, 25)]}
    if complete['resource_gate_passed']:
        required |= {'CP0_BINDING.json', 'checkpoints/latest.json', 'checkpoints/stage_00.json'}
    assert set(complete['file_sha256']) == required
    for relative, digest in complete['file_sha256'].items():
        assert sha(root / relative) == digest, relative
    assert (root / 'training_input_IDs.jsonl').stat().st_size == 0
    resources(root, complete)
    path = WORK / 'init/initialization/INITIAL_FUNCTION.json'
    assert sha(path) == complete['initial_function_sha256']
    initial = json.loads(path.read_text())
    assert initial['complete'] and initial['calibration_requests'] == 384 and initial['diagnostic_requests'] == 32
    assert initial['compatibility_NLL_paths'] == initial['compatibility_short_generations'] == 4
    assert initial['total_extra_teacher_forced_requests'] == 420
    for gate in (initial['dense_native_gate'], initial['compact_native_gate']):
        assert gate['official_4_token_generation_exact'] and gate['official_NLL_error'] < .015
    for key in ('dense_diagnostic', 'compact_diagnostic'):
        d = initial[key]
        assert d['QA'] == len(d['records']) == 16
        assert [r['id_sha256'] for r in d['records']] == ready['diagnostic_ID_SHA256']
        values = [r['mean_answer_token_NLL'] for r in d['records']]
        assert all(math.isfinite(v) for v in values)
        assert sum(values) / 16 == d['QA_equal_mean_NLL']
    for a, b in zip(initial['dense_diagnostic']['records'], initial['compact_diagnostic']['records']):
        for key in ('id_sha256', 'prepared_tensors_sha256', 'target_sha256', 'target_tokens'):
            assert a[key] == b[key]
    gap = initial['compact_diagnostic']['QA_equal_mean_NLL'] - initial['dense_diagnostic']['QA_equal_mean_NLL']
    assert gap == initial['NLL_gap'] and initial['resource_gate_passed'] == (gap <= math.log(2))
    assert complete['resource_gate_passed'] == initial['resource_gate_passed']
    assert len(initial['physical_layers']) == 72
    assert len({r['name'] for r in initial['physical_layers']}) == 72
    assert all(r['physical_pack_materialization_exact'] and r['calibration_QA'] == 16 for r in initial['physical_layers'])
    assert sum(r['layer_layout']['stored_weight_slots'] for r in initial['physical_layers']) == FFN_SLOTS
    assert sum(r['layer_layout']['pattern_bytes'] for r in initial['physical_layers']) == PATTERN_BYTES
    if initial['resource_gate_passed']:
        path = WORK / 'init/CP0_BINDING.json'
        assert sha(path) == complete['CP0_binding_sha256']
        binding = json.loads(path.read_text()); meta = binding['checkpoint']
        assert binding['untrained'] and meta['stage'] == 0
        assert sha(WORK / 'init' / meta['path']) == meta['sha256']
    proof = dict(complete=True, resource_gate_passed=initial['resource_gate_passed'],
        NLL_gap=gap, cutoff_nats=math.log(2), observed_model_seconds=complete['model_RUN_seconds'],
        initial_complete_sha256=sha(REPORT / 'phases/init/COMPLETE.json'),
        initial_function_sha256=complete['initial_function_sha256'],
        ready_sha256=sha(REPORT / 'READY.json'), no_F1_or_model_quality_guarantee=True)
    write(REPORT / 'INITIAL_VERIFIED.json', proof)
    return proof


def original_scorer():
    tree = ast.parse((PORT / 'runtime_utils.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('score', 'tokens')]
    assert len(nodes) == 2
    namespace = dict(re=re, Counter=Counter)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-AnswerF1-score>', 'exec'), namespace)
    return namespace['score']


def verify_case(case, manifest, fit, dev, split, plan):
    import torch
    assert not torch.cuda.is_initialized()
    root = WORK / case; complete = exited(case)
    assert complete['complete'] and complete['phase'] == case
    assert (complete['FIT_QA'], complete['DEV_QA'], complete['updates'], complete['TEST_QA']) == (1213, 100, 87, 0)
    assert 0 < complete['model_RUN_seconds'] <= 7200
    assert complete['ready_sha256'] == sha(REPORT / 'READY.json')
    required = {'NATIVE_GATE.json', 'FFN_ADAPTATION.json', 'config.json', 'INITIAL_BINDING.json',
        'logs/events.jsonl', 'training_input_IDs.jsonl', 'resources.jsonl', 'memory_summary.json', 'checkpoints/latest.json'}
    for stage_plan in plan:
        stage = stage_plan['stage']
        required |= {f'layouts/S{stage:02d}.json', f'validity/stage_{stage:02d}.json',
            f'gradients/S{stage:02d}_first_raw.pt', f'checkpoints/stage_{stage:02d}.json'}
        for step in range(1, stage_plan['updates'] + 1):
            required |= {f'optimizer/S{stage:02d}_{step:03d}.json', f'gradients/S{stage:02d}_{step:03d}.json'}
    assert set(complete['file_sha256']) == required
    for relative, digest in complete['file_sha256'].items():
        path = (root / relative).resolve(); assert path.is_relative_to(root.resolve())
        assert sha(path) == digest, relative
    resources(root, complete)
    config = json.loads((root / 'config.json').read_text())
    assert config['phase'] == case and config['method'] == 'SparseConnection' + case
    assert (config['FIT_QA'], config['DEV_QA'], config['TEST_QA'], config['updates']) == (1213, 100, 0, 87)
    assert config['qv_learning_rate'] == 1e-4 and config['FFN_learning_rate'] == (1e-5 if case == 'Plastic' else None)
    assert config['ready_sha256'] == sha(REPORT / 'READY.json')
    binding = json.loads((root / 'INITIAL_BINDING.json').read_text())
    assert binding['CP0_binding_sha256'] == sha(WORK / 'init/CP0_BINDING.json')
    cp0 = json.loads((WORK / 'init/CP0_BINDING.json').read_text())
    assert binding['initial_parameter_digest'] == cp0['model_parameter_digest']
    assert binding['mask_digest'] == complete['mask_digest'] == cp0['mask_digest']
    expected = QV_COORDINATES + (FFN_SLOTS if case == 'Plastic' else 0)
    assert binding['trainable_coordinates'] == complete['training_coordinates'] == expected
    layout = complete['final_layout']
    assert layout['weight_coefficients'] == FINAL_VALUES and layout['packed_modules'] == 72
    assert layout['packed_slots'] == FFN_SLOTS and layout['pattern_bytes'] == PATTERN_BYTES
    assert layout['trainable_coordinates'] == 0 and layout['lookup_bytes'] == 72 * 6 * 2 * 8
    assert all(r['values_dtype'] == 'torch.float16' for r in layout['packed_layers'].values())
    assert complete['full_reload'] and complete['stable_before'] == complete['stable_after']
    adaptation = json.loads((root / 'FFN_ADAPTATION.json').read_text())
    initial_ffn, final_ffn = adaptation['initial_FP16'], adaptation['final_FP16']
    assert set(initial_ffn['layer_sha256']) == set(final_ffn['layer_sha256']) and len(initial_ffn['layer_sha256']) == 72
    changed_layers = sum(initial_ffn['layer_sha256'][name] != digest for name, digest in final_ffn['layer_sha256'].items())
    assert changed_layers == adaptation['FP16_changed_layers']
    if case == 'Frozen':
        assert changed_layers == 0
        assert adaptation['initial_FP32_master_sha256'] == adaptation['before_final_cast_FP32_master_sha256']
    inputs = [json.loads(line) for line in (root / 'training_input_IDs.jsonl').read_text().splitlines()]
    wanted = []
    for task in manifest['tasks']:
        tid = task['task']
        order = sorted(fit[tid], key=lambda r: hashlib.sha256(f"42:{tid}:{r['id']}".encode()).digest())
        wanted.extend((tid, hashlib.sha256(row['id'].encode()).hexdigest()) for row in order)
    assert [(r['stage'], r['id_sha256']) for r in inputs] == wanted and len(inputs) == 1213
    events = [json.loads(line) for line in (root / 'logs/events.jsonl').read_text().splitlines()]
    events = [e for e in events if e['event'] == 'train_update']; assert len(events) == 87
    assert len(list((root / 'optimizer').glob('*.json'))) == 87
    assert len(list((root / 'gradients').glob('*.json'))) == 87
    assert len(list((root / 'gradients').glob('*.pt'))) == 19
    for stage_plan in plan:
        stage = stage_plan['stage']; records = [e for e in events if e['stage'] == stage]
        assert len(records) == stage_plan['updates']
        validity = json.loads((root / 'validity' / f'stage_{stage:02d}.json').read_text())
        assert validity['passed'] and validity['actual_parameters_changed']
        assert validity['updates'] == validity['positive_lr_updates'] == stage_plan['updates']
        assert validity['parameters_before_sha256'] != validity['parameters_after_sha256']
        for index, event in enumerate(records, 1):
            assert event['step'] == index and event['total_qa'] == stage_plan['QA']
            assert event['qa_seen'] == min(index * 16, stage_plan['QA'])
            assert math.isfinite(event['loss']) and all(math.isfinite(v) and v > 0 for v in event['learning_rate_used'])
            witness = json.loads((root / 'optimizer' / f'S{stage:02d}_{index:03d}.json').read_text())
            assert witness['parameter_IDs_match'] and witness['coordinates'] == expected
            assert witness['gradient_bytes'] == 4 * expected and witness['two_moment_bytes'] == 8 * expected
            assert witness['all_moments_CPU'] and witness['actual_learning_rate'] == event['learning_rate_used']
            params = witness['actual_parameters']
            assert len({p['id'] for p in params}) == len({p['name'] for p in params}) == len(params)
            raw = json.loads((root / 'gradients' / f'S{stage:02d}_{index:03d}.json').read_text())['raw']
            assert set(raw) == {p['name'] for p in params}
            assert all(r['finite'] and math.isfinite(r['norm']) for r in raw.values())
            if index == 1:
                snapshot = torch.load(root / 'gradients' / f'S{stage:02d}_first_raw.pt', map_location='cpu', weights_only=True)
                assert set(snapshot) == set(raw)
                for parameter in params:
                    value = snapshot[parameter['name']]
                    assert list(value.shape) == parameter['shape'] and value.dtype == torch.float32
                    assert bool(torch.isfinite(value).all())
                del snapshot
    gates = json.loads((root / 'NATIVE_GATE.json').read_text())
    assert gates['NLL_paths'] == gates['short_generations'] == 38 and gates['current_stage_only']
    assert [g['stage'] for g in gates['gates']] == list(range(1, 20))
    assert all(g['official_4_token_generation_exact'] and g['official_NLL_error'] < .015 for g in gates['gates'])
    meta = json.loads((root / 'checkpoints/latest.json').read_text()); assert meta['stage'] == 19
    assert sha(root / meta['path']) == meta['sha256']
    package_bytes = meta['bytes'] + sum(p.stat().st_size for p in (root / 'checkpoints/processor').rglob('*') if p.is_file())
    package_below_dense_weight_only_lower_bound = package_bytes < 2 * BASE_VALUES
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / 'checkpoints/processor', local_files_only=True)
    expected_predictions = {f"evaluations/stage_19/T{task['task']:02d}_dev_predictions.jsonl"
        for task in manifest['tasks'] if dev[task['task']]}
    assert set(complete['predictions']['file_sha256']) == expected_predictions
    assert complete['predictions']['count'] == 100 and complete['predictions']['scores_not_computed']
    score = original_scorer(); values = {}; domains = {d: [] for d in DOMAINS}
    for task in manifest['tasks']:
        tid = task['task']
        if not dev[tid]:
            continue
        relative = f'evaluations/stage_19/T{tid:02d}_dev_predictions.jsonl'
        assert sha(root / relative) == complete['predictions']['file_sha256'][relative]
        predictions = [json.loads(line) for line in (root / relative).read_text().splitlines()]
        assert [r['id'] for r in predictions] == [r['id'] for r in dev[tid]]
        for prediction, row in zip(predictions, dev[tid]):
            assert prediction['references'] == row['answers']
            assert prediction['generation_cap'] == task['max_new_tokens']
            assert prediction['image_sha256'] == {n: split['image_sha256'][n] for n in row['images']}
            ids = prediction['output_token_IDs']; assert len(ids) == prediction['output_tokens'] > 0
            assert prediction['eos_token_id'] == tokenizer.eos_token_id
            assert tokenizer.decode(ids, skip_special_tokens=True) == prediction['prediction']
            assert len(ids) <= task['max_new_tokens'] and (ids[-1] == prediction['eos_token_id'] or len(ids) == task['max_new_tokens'])
            f1 = score(prediction['prediction'], row['answers'])['token_f1']
            assert math.isfinite(f1) and 0 <= f1 <= 100
            values[row['id']] = f1; domains[task['domain']].append(f1)
    assert len(values) == 100 and all(len(v) == 20 for v in domains.values())
    quality = {domain: sum(v) / len(v) for domain, v in domains.items()}
    return dict(quality=quality, domain_macro=sum(quality.values()) / 5,
        final_layout=layout, actual_final_checkpoint_and_processor_bytes=package_bytes,
        smaller_than_dense_FP16_weight_only_lower_bound=package_below_dense_weight_only_lower_bound,
        model_RUN_seconds=complete['model_RUN_seconds'], final_checkpoint_sha256=meta['sha256'],
        initial_binding_sha256=sha(root / 'INITIAL_BINDING.json'), actual_train_updates=87,
        CP0_binding_sha256=binding['CP0_binding_sha256'],
        validated_first_gradient_snapshots=19, actual_optimizer_witnesses=87,
        FFN_adaptation=adaptation), values


def paired(candidate, reference, split):
    groups = {d: [] for d in DOMAINS}
    for gid, group in split['groups'].items():
        if group['dev']:
            assert all(ident in candidate and ident in reference for ident in group['ids'])
            groups[group['domain']].append((gid, group['ids']))
    for domain in groups:
        groups[domain].sort()
    assert sum(map(len, groups.values())) == 89
    rng = random.Random(20261024); differences = []
    for _ in range(10000):
        domain_means = []
        for domain in DOMAINS:
            drawn = [rng.randrange(len(groups[domain])) for _ in groups[domain]]
            ids = [ident for j in drawn for ident in groups[domain][j][1]]
            domain_means.append(sum(candidate[i] - reference[i] for i in ids) / len(ids))
        differences.append(sum(domain_means) / 5)
    differences.sort()
    def quantile(fraction):
        pos = fraction * (len(differences) - 1); lower = math.floor(pos); upper = math.ceil(pos)
        return differences[lower] + (pos - lower) * (differences[upper] - differences[lower])
    rows = [dict(id_sha256=hashlib.sha256(ident.encode()).hexdigest(), domain=d,
                 Plastic_F1=candidate[ident], Frozen_F1=reference[ident], difference_pp=candidate[ident] - reference[ident])
            for d in DOMAINS for _, ids in groups[d] for ident in ids]
    return dict(primary='Plastic_minus_Frozen', all_100_scalar_pairs=rows,
        exploratory_group_interval_pp=[quantile(.025), quantile(.975)], replicates=10000, seed=20261024,
        explored_DEV_single_seed_not_new_holdout_or_equivalence=True)


def verify():
    once('VERIFY_STARTED.json'); ready = check_ready()
    initial_path = REPORT / 'INITIAL_VERIFIED.json'
    initial = json.loads(initial_path.read_text()); assert initial['complete'] and initial['resource_gate_passed']
    _, manifest, split, fit, dev, _, plan, _, _, _ = data()
    cases = {}; scores = {}
    for case in CASES:
        cases[case], scores[case] = verify_case(case, manifest, fit, dev, split, plan)
    # The binding files differ in trainability; the CP0 reference within them is identical.
    assert cases['Frozen']['CP0_binding_sha256'] == cases['Plastic']['CP0_binding_sha256']
    aggregate = initial['observed_model_seconds'] + sum(c['model_RUN_seconds'] for c in cases.values())
    assert aggregate <= 18000
    old = json.loads((ASSETS / 'reference_answer_guided_READY.json').read_text())
    assert old['reference_best_macro'] == 43.79198740853657
    eligible = (all(c['smaller_than_dense_FP16_weight_only_lower_bound'] for c in cases.values())
        and cases['Plastic']['domain_macro'] > max(cases['Frozen']['domain_macro'], old['reference_best_macro']))
    result = dict(complete=True, cases=cases, new_pair=paired(scores['Plastic'], scores['Frozen'], split),
        aggregate_model_seconds=aggregate, inherited_legal_FIT_only_anchor=old['reference_best_macro'],
        old_anchor_source_sha256=sha(ASSETS / 'reference_answer_guided_READY.json'),
        possible_followup_necessity_review_only=eligible, automatic_full_confirmation=False,
        old_scores_and_pairs_not_repeated=True, ready_sha256=sha(REPORT / 'READY.json'),
        initial_verified_sha256=sha(initial_path), TEST_QA=0, goal_not_achieved=True)
    return result


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ap = argparse.ArgumentParser(); ap.add_argument('--initial-only', action='store_true'); args = ap.parse_args()
    begin = time.monotonic()
    if args.initial_only:
        result = verify_initial()
    else:
        result = verify(); result['CPU_verification_seconds'] = time.monotonic() - begin
        write(REPORT / 'VERIFIED.json', result)
    print(json.dumps(dict(verified=True, initial_only=args.initial_only)))
