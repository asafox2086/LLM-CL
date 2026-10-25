"""Post-exit integrity checks, then exactly400 original Answer-F1 calls once.

No Torch import/checkpoint deserialization/model execution. Historical quality
is copied; this verifier establishes neither PAS nor folded-OSFT superiority.
"""
from pathlib import Path
import ast
import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import time

from paired_common import RUN, ASSET, ROOT, LIMITS, read, sha, write

ARMS = ('dynamic', 'static')
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
BASE = {'coefficients': 893675552, 'bytes': 1787351104}
WEIGHTS = {'coefficients': 893675576, 'bytes': 1787351200}
GRADIENT_BYTES = 17596512
OPTIMIZER_BYTES = 35193504


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest_string(value):
    assert isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def keyed(rows):
    result = {row['id']: row for row in rows}
    assert len(result) == len(rows)
    return result


def beta_list(values):
    assert isinstance(values, list) and len(values) == 24 and all(finite(x) for x in values)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def expected_plan():
    names = [f'model.layers.{i}.self_attn.v_proj' for i in range(24)]
    chosen = set(sorted(names, key=lambda n: hashlib.sha256(('42:coordinate-remainder:' + n).encode()).hexdigest())[:6])
    return {f'model.layers.{i}.self_attn.{kind}_proj': 65 + int(f'model.layers.{i}.self_attn.{kind}_proj' in chosen)
            for i in range(24) for kind in ('q', 'v')}


def response_metadata(value, arm, final):
    assert value['arm'] == arm and value['final'] is final
    expected = 'dynamic-beta-v1' if arm == 'dynamic' else ('static-effective-gamma-v1' if final else 'static-beta-calibration-v1')
    assert value['format'] == expected and value['both_arms_explicit_branch'] and not value['down_matrix_copied']
    assert value['layers'] == value['beta_parameters'] == 24 and value['hidden_size'] == 896 and value['intermediate_size'] == 4864
    assert value['beta_dtype'] == 'torch.float32' and value['gate_score_arithmetic'] == 'GPU FP32'
    assert value['dynamic_epsilon'] == 1e-8 and value['original_gate_up_down_state_paths_preserved']
    assert value['beta_semantics'] == ('effective_gamma' if arm == 'static' and final else 'beta')
    assert value['sign_dtype'] == 'torch.float16' and value['derived_sign_buffer_bytes'] == 233472
    assert value['calibration_buffer_bytes'] == (96 if arm == 'static' and not final else 0)
    assert value['calibration_buffer_name'] == ('calibration_scale' if arm == 'static' and not final else None)
    assert value['final_coefficients'] == WEIGHTS['coefficients'] and value['final_weight_bytes'] == WEIGHTS['bytes']


def receipt(value, arm, final, gpu):
    assert value['layout']['weights'] == WEIGHTS and value['layout']['trainable'] == 0
    assert value['device'] == f'cuda:{gpu}'
    response_metadata(value['response'], arm, final)
    for key in ('digest', 'all_buffers_digest', 'functional_config_digest', 'generation_config_digest', 'language_generation_config_digest'):
        digest_string(value[key])
    manifest = value['buffer_manifest']
    signs = {n: b for n, b in manifest.items() if n.endswith('.mlp.response_sign')}
    calibrations = {n: b for n, b in manifest.items() if n.endswith('.mlp.calibration_scale')}
    assert set(signs) == {f'language_model.model.layers.{i}.mlp.response_sign' for i in range(24)}
    assert all(b == {'shape': [4864], 'dtype': 'torch.float16', 'bytes': 9728} for b in signs.values())
    expected_calibrations = {f'language_model.model.layers.{i}.mlp.calibration_scale' for i in range(24)} if arm == 'static' and not final else set()
    assert set(calibrations) == expected_calibrations
    assert all(b == {'shape': [], 'dtype': 'torch.float32', 'bytes': 4} for b in calibrations.values())
    assert value['layout']['ordinary_buffers']['bytes'] == 9032 + 233472 + (96 if expected_calibrations else 0)


def same_receipt(left, right, *, device=True):
    # Reload adds explicit proof flags; export adds path/size/hash fields.
    keys = ('layout', 'digest', 'all_buffers_digest', 'buffer_manifest', 'parameter_alias_groups',
            'buffer_alias_groups', 'functional_config_digest', 'generation_config_digest',
            'language_generation_config_digest', 'language_attention_classes', 'vision_attention_classes', 'response')
    for key in keys:
        assert left[key] == right[key], key
    if device:
        assert left['device'] == right['device']


def main():
    tick = time.monotonic()
    assert not (RUN / 'VERIFY_STARTED.json').exists() and not (RUN / 'VERIFIED.json').exists()
    assert not list(RUN.glob('*_FAILURE.json'))
    ready = read(RUN / 'PAIR_READY.json'); ready_hash = sha(RUN / 'PAIR_READY.json')
    for field in ('sources', 'prior_files'):
        for name, expected in ready[field].items():
            assert sha(RUN / name) == expected, name
    for name, expected in ready['environment'].items():
        assert sha(name) == expected, name
    for name, expected in ready['runtime_files'].items():
        if not name.endswith('.safetensors'):
            assert sha(ASSET / name) == expected, name
    assert sha(ROOT / 'exp/ffn_static_delete_20261025/v1/core.py') == ready['storage_helper_sha256']
    required_priors = {'calibration_COMPLETE.json', 'calibration_VERIFIED.json', 'prior_VERIFIED.json',
                       'prior_COMPLETE.json', 'prior_READY.json', 'prior_parent_predictions.json'}
    assert required_priors <= set(ready['prior_files'])
    calibration = read(RUN / 'calibration_COMPLETE.json'); calibrated = read(RUN / 'calibration_VERIFIED.json')
    assert calibrated['complete'] and calibrated['actual_successful_exit'] and calibrated['nondegenerate_only']
    assert calibrated['COMPLETE_sha256'] == sha(RUN / 'calibration_COMPLETE.json')
    assert calibrated['ready_sha256'] == calibration['ready_sha256'] and calibration['state_and_buffers_unchanged']
    assert calibration['layout']['weights'] == BASE and calibration['language_NLL_requests'] == 32
    assert calibration['extra_sampled_Down_requests'] == 768 and calibration['new_generations'] == calibration['new_scores'] == calibration['backwards'] == calibration['optimizer_updates'] == 0
    assert len(calibration['layers']) == 24 and [x['layer'] for x in calibration['layers']] == list(range(24))
    scales = [x['RMS'] for x in calibration['layers']]
    assert all(finite(x) and x > 0 for x in scales)

    all_exit = read(RUN / 'ALL_EXIT.json')
    assert all_exit['complete'] and all_exit['ready_sha256'] == ready_hash
    for key, expected in dict(new_generations=410, new_scores=0, FIT_backwards=2426, optimizer_updates=174, TEST_QA=0).items():
        assert all_exit[key] == expected
    assert set(LIMITS) == {'train_dynamic', 'train_static', 'eval_dynamic', 'eval_static'} and sum(LIMITS.values()) == 8400
    assert set(all_exit['source_COMPLETE_sha256']) == set(LIMITS)
    phases, starts, exits = {}, {}, {}
    for phase, limit in LIMITS.items():
        start = read(RUN / (phase + '_STARTED.json')); end = read(RUN / (phase + '_COMPLETE.json')); ex = read(RUN / (phase + '_EXIT.json'))
        identity = start['identity']
        assert identity == end['identity'] and identity['pid'] == identity['pgid'] == ex['pid'] and identity['uid'] == os.getuid()
        proc = Path('/proc') / str(identity['pid']) / 'stat'
        if proc.exists():
            current_start = proc.read_text().rsplit(') ', 1)[1].split()[19]
            assert current_start != str(identity['startticks']), 'Original worker still present'
        assert ex['returncode'] == 0 and start['ready_sha256'] == end['ready_sha256'] == ready_hash
        assert start['limit'] == limit and finite(end['seconds']) and 0 < end['seconds'] <= limit
        assert finite(start['unix']) and finite(ex['unix']) and ex['unix'] >= start['unix'] + end['seconds']
        assert start['GPU'] == ready['phase_GPUs'][phase]
        assert sha(RUN / (phase + '_COMPLETE.json')) == all_exit['source_COMPLETE_sha256'][phase]
        for name, expected in end['files'].items():
            assert sha(RUN / name) == expected, name
        phases[phase], starts[phase], exits[phase] = end, start, ex
    assert len({s['boot'] for s in starts.values()}) == 1
    assert starts['train_dynamic']['GPU'] != starts['train_static']['GPU']
    assert max(starts[p]['unix'] for p in ('train_dynamic', 'train_static')) < min(exits[p]['unix'] for p in ('train_dynamic', 'train_static'))
    assert starts['eval_dynamic']['unix'] >= max(exits[p]['unix'] for p in ('train_dynamic', 'train_static'))
    assert starts['eval_static']['unix'] >= exits['eval_dynamic']['unix']
    assert starts['eval_dynamic']['GPU'] == starts['eval_static']['GPU']
    assert sum(p['seconds'] for p in phases.values()) <= 8400

    port = ASSET / 'exp/two_backbone_native_cl_20261017_v2'
    policy = read(port / 'execution_policy.json'); manifest_path = ASSET / policy['conditions']['pct01']['manifest']; manifest = read(manifest_path)
    split_path = ASSET / 'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'; split = read(split_path)
    assert sha(manifest_path) == ready['manifest_sha256'] and sha(split_path) == ready['split_sha256']
    gold, fit_rows, order, plan = {}, {}, [], []
    for task in manifest['tasks']:
        stage = task['task']; path = ASSET / task['files']['train']['path']
        assert sha(path) == task['files']['train']['sha256']
        values = keyed([json.loads(line) for line in path.read_text().splitlines()])
        fit_ids, dev_ids = split['fit_ids'][str(stage)], split['dev_ids'][str(stage)]
        assert set(fit_ids).isdisjoint(dev_ids) and set(fit_ids) | set(dev_ids) == set(values)
        assert not (set(values) & (set(gold) | set(fit_rows)))
        fit_rows.update({rid: (task, values[rid]) for rid in fit_ids}); gold.update({rid: (task, values[rid]) for rid in dev_ids})
        ordered = sorted(fit_ids, key=lambda rid: hashlib.sha256(f'42:{stage}:{rid}'.encode()).digest())
        order.extend(ordered)
        plan.append(dict(stage=stage, QA=len(ordered), updates=(len(ordered) + 15) // 16,
                         order_sha256=hashlib.sha256(''.join(rid + '\n' for rid in ordered).encode()).hexdigest()))
    assert [p['stage'] for p in plan] == list(range(1, 20)) and plan == ready['training_plan']
    assert len(fit_rows) == 1213 and len(gold) == 100 and sum(p['updates'] for p in plan) == 87
    assert set(fit_rows).isdisjoint(gold) and all(sum(t['domain'] == d for t, _ in gold.values()) == 20 for d in DOMAINS)
    original_main = next(n for n in ast.parse((port / 'run.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    original_loop = next(n for n in original_main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'stage')
    cutoff = next(i for i, n in enumerate(original_loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'wanted' for t in n.targets))
    prefix = ast.Module(body=original_loop.body[:cutoff - 1], type_ignores=[])
    assert hashlib.sha256(ast.dump(prefix, include_attributes=False).encode()).hexdigest() == ready['training_ast_sha256']
    assert len(calibration['records']) == 32 and len({x['group_id'] for x in calibration['records']}) == 32
    assert all(x['id'] in fit_rows and fit_rows[x['id']][0]['task'] == 1 for x in calibration['records'])

    spec = importlib.util.spec_from_file_location('pair_raw_retrieval', port / 'retrieval.py')
    retrieval = importlib.util.module_from_spec(spec); spec.loader.exec_module(retrieval)
    from PIL import Image
    image_cache, raw_cache = {}, {}
    def raw(task, row):
        if row['id'] in raw_cache:
            return raw_cache[row['id']]
        context = row.get('context', ''); indexes = []
        if len(context) > 6000:
            chunks = [context[i:i + 1000] for i in range(0, len(context), 800)]
            indexes = sorted(retrieval.retrieve(row['question'], chunks, 5)); context = '\n'.join(chunks[i] for i in indexes)
        shapes, images = [], []
        for name in row['images']:
            if name not in image_cache:
                assert sha(ASSET / name) == split['image_sha256'][name]
                with Image.open(ASSET / name) as image:
                    width, height = image.size
                before = [width, height]
                if width * height > 401408:
                    ratio = math.sqrt(401408 / (width * height)); width, height = max(1, int(width * ratio)), max(1, int(height * ratio))
                image_cache[name] = dict(original=before, common_input=[width, height])
            shapes.append(image_cache[name]); images.append(dict(path=name, sha256=split['image_sha256'][name]))
        contract = dict(images=images, instruction=task['instruction'], question=row['question'], original_context=row.get('context', ''),
                        resolved_context=context, retrieved_chunk_indexes=indexes, image_shapes=shapes, common_image_area_cap=401408)
        result = hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        raw_cache[row['id']] = (result, contract)
        return result, contract

    reference = read(RUN / 'prior_VERIFIED.json'); old = read(RUN / 'prior_COMPLETE.json')
    assert reference['complete'] and reference['source_COMPLETE_sha256'] == sha(RUN / 'prior_COMPLETE.json')
    assert reference['ready_sha256'] == sha(RUN / 'prior_READY.json')
    assert ready['historical_parent_quality'] == reference['quality']['parent']
    assert sha(RUN / 'prior_parent_predictions.json') == old['files']['parent_predictions.json']
    old_predictions = keyed(read(RUN / 'prior_parent_predictions.json'))
    assert set(old_predictions) == set(gold)
    assert all(old_predictions[rid]['references'] == row['answers'] for rid, (_, row) in gold.items())
    historical = reference['quality']['parent']
    assert set(historical['domains']) == set(DOMAINS) and set(keyed(historical['per_id'])) == set(gold)

    predictions, input_records, initializations, compiles, memory = {}, {}, {}, {}, {}
    rank_plan = expected_plan()
    for arm in ARMS:
        tr, ev = phases['train_' + arm], phases['eval_' + arm]
        train_gpu, eval_gpu = starts['train_' + arm]['GPU'], starts['eval_' + arm]['GPU']
        assert tr['arm'] == ev['arm'] == arm
        assert tr['new_generations'] == 100 and tr['FIT_backwards'] == 1213 and tr['optimizer_updates'] == 87 and tr['coordinates'] == 4399128
        assert tr['training_plan'] == plan and tr['training_ast_sha256'] == ready['training_ast_sha256']
        assert tr['frozen_unchanged'] and tr['buffers_unchanged'] and tr['adapter_before'] != tr['adapter_after']
        required = {arm + suffix for suffix in ('_initial_predictions.json', '_ADAPTER_INITIAL.json', '_COMPILE.json', '_FIT_INPUTS.json', '_TRAIN_MEMORY.json', '_initial.pt', '_complete.pt')}
        required.add(arm + '_training/logs/events.jsonl')
        for stage in plan:
            sid = stage['stage']
            required.update({f'{arm}_STAGE_{sid:02}_COMPLETE.json', f'{arm}_adapter_S{sid:02}.pt', f'{arm}_training/validity/stage_{sid:02}.json'})
            required.update(f'{arm}_grad_S{sid:02}_{i:03}.json' for i in range(1, stage['updates'] + 1))
        assert required <= set(tr['files'])
        assert set(ev['files']) == {arm + '_memory.json', arm + '_final_predictions.json'}
        assert len(list(RUN.glob(arm + '_grad_S*.json'))) == 87
        assert len(list(RUN.glob(arm + '_STAGE_*_COMPLETE.json'))) == 19
        assert len(list(RUN.glob(arm + '_adapter_S*.pt'))) == 19
        assert tr['original_base']['layout']['weights'] == BASE and tr['original_base']['response'] is None
        assert tr['original_base']['digest'] == calibration['initial_state_digest'] and tr['original_base']['all_buffers_digest'] == calibration['all_buffers_digest']
        for name, final in (('initial_checkpoint', False), ('initial_reload', False), ('initial_invariant', False), ('checkpoint', True)):
            receipt(tr[name], arm, final, train_gpu)
        same_receipt(tr['initial_checkpoint'], tr['initial_reload']); same_receipt(tr['initial_reload'], tr['initial_invariant'])
        assert tr['initial_invariant']['complete_state_and_buffers_unchanged']
        for name in ('self_contained_meta_construction', 'all_buffers_explicitly_restored', 'strict_complete_reload', 'mixed_parameter_dtypes_preserved'):
            assert tr['initial_reload'][name] and ev['reloaded'][name]
        for state, field in (('initial', 'initial_checkpoint'), ('complete', 'checkpoint')):
            name = arm + '_' + state + '.pt'; cp = tr[field]
            assert (RUN / name).stat().st_size == cp['bytes'] and cp['sha256'] == tr['files'][name]
            assert Path(cp['path']).resolve() == (RUN / name).resolve()
            assert cp['complete_graph'] and not cp['external_base_weights_required'] and cp['no_adapter_optimizer_or_teacher'] and cp['alias_preserving_cpu_snapshot']
        assert ev['new_generations'] == 105 and ev['FIT_backwards'] == ev['optimizer_updates'] == 0
        receipt(ev['reloaded'], arm, True, eval_gpu); receipt(ev['invariant'], arm, True, eval_gpu)
        same_receipt(tr['checkpoint'], ev['reloaded'], device=False); same_receipt(ev['reloaded'], ev['invariant'])
        assert ev['invariant']['complete_state_and_buffers_unchanged']
        fold = tr['fold']; assert len(fold) == 48 and {x['module']: x['rank'] for x in fold} == rank_plan
        assert all(x['scale'] == 4 and x['dtype'] == 'torch.float16' and x['merge_dtype'] == 'torch.float32' and x['device'] == f'cuda:{train_gpu}'
                   and not x['autocast_enabled'] and not x['matmul_allow_tf32'] for x in fold)
        initial = read(RUN / (arm + '_ADAPTER_INITIAL.json')); initializations[arm] = initial
        assert initial['coordinates'] == 4399128 and initial['parameters'] == 120 and initial['rank_plan'] == rank_plan
        beta_list(initial['beta']); assert not any(initial['beta']); response_metadata(initial['response'], arm, False)
        assert len(initial['initialization']) == 48 and {x['module']: x['rank'] for x in initial['initialization']} == rank_plan
        assert all(x['B_zero'] and x['scale'] == 4 and x['dropout'] == .1 for x in initial['initialization'])
        for x in initial['initialization']:
            digest_string(x['A_digest']); digest_string(x['B_digest'])
        for stage in plan:
            sid, updates = stage['stage'], stage['updates']; st = read(RUN / f'{arm}_STAGE_{sid:02}_COMPLETE.json')
            valid = read(RUN / f'{arm}_training/validity/stage_{sid:02}.json')
            assert st['stage'] == sid and st['updates'] == updates and st['bytes'] == GRADIENT_BYTES
            assert st['optimizer_state_bytes'] == OPTIMIZER_BYTES and st['all_optimizer_state_GPU'] and st['frozen_unchanged']
            beta_list(st['beta']); digest_string(st['adapter_digest'])
            assert f'{arm}_adapter_S{sid:02}.pt' in tr['files']
            assert valid['passed'] and valid['actual_parameters_changed'] and valid['updates'] == valid['positive_lr_updates'] == updates
            assert valid['parameters_before_sha256'] != valid['parameters_after_sha256']
            for i in range(1, updates + 1):
                assert read(RUN / f'{arm}_grad_S{sid:02}_{i:03}.json') == dict(parameters=120, finite=True, gradient_bytes=GRADIENT_BYTES)
        events = [json.loads(line) for line in (RUN / (arm + '_training/logs/events.jsonl')).read_bytes().splitlines()]
        assert len(events) == 87
        for stage in plan:
            selected = [e for e in events if e['stage'] == stage['stage']]; assert len(selected) == stage['updates']
            warmup = min(math.floor(.03 * stage['updates']), stage['updates'] - 1)
            for k, e in enumerate(selected):
                assert e['event'] == 'train_update' and e['step'] == k + 1 and e['qa_seen'] == min((k + 1) * 16, stage['QA']) and e['total_qa'] == stage['QA']
                assert e['warmup_updates'] == warmup and finite(e['loss'])
                expected_lr = 1e-4 * (.5 * (1 + math.cos(math.pi * (k - warmup) / max(1, stage['updates'] - warmup))) if k >= warmup else k / max(1, warmup))
                assert len(e['learning_rate_used']) == 1 and e['learning_rate_used'][0] > 0 and math.isclose(e['learning_rate_used'][0], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
                assert finite(e['gradient_l2_preclip']) and e['gradient_l2_preclip'] >= 0 and finite(e['gradient_l2_postclip']) and 0 <= e['gradient_l2_postclip'] <= 1.00001
                assert e['visual_cache_cpu_bytes'] == e['visual_cache_cpu_limit'] == 0
        compiled = read(RUN / (arm + '_COMPILE.json')); compiles[arm] = compiled
        beta_list(compiled['learned_beta']); beta_list(compiled['final_beta'])
        assert compiled['learned_beta'] == read(RUN / f'{arm}_STAGE_19_COMPLETE.json')['beta']
        assert compiled['effective_beta_digest'] == compiled['expected_effective_beta_digest']
        digest_string(compiled['effective_beta_digest']); assert compiled['metadata'] == tr['compile']
        response_metadata(compiled['metadata'], arm, True)
        assert compiled['metadata']['no_dense_D_fold'] and compiled['metadata']['final_saved_named_buffers_include_sign']
        assert compiled['metadata']['compiled_on_device'] == f'cuda:{train_gpu}' and compiled['metadata']['compile_arithmetic'] == 'GPU FP32 scalar only'
        assert compiled['metadata']['static_calibration_absorbed'] is (arm == 'static')
        expected_beta = [f32(b * f32(c)) for b, c in zip(compiled['learned_beta'], scales)] if arm == 'static' else compiled['learned_beta']
        assert compiled['final_beta'] == expected_beta
        inputs = read(RUN / (arm + '_FIT_INPUTS.json')); input_records[arm] = inputs
        assert len(inputs) == 1213 and [x['id'] for x in inputs] == order
        for x in inputs:
            task, row = fit_rows[x['id']]; rhash, contract = raw(task, row)
            assert x['phase'] == 'TRAIN' and x['task'] == task['task'] and x['domain'] == task['domain']
            assert x['raw_input_sha256'] == rhash and x['image_SHA256'] == contract['images'] and x['image_shapes'] == contract['image_shapes']
            digest_string(x['transport_sha256'])
        for state in ('initial', 'final'):
            rows = read(RUN / (arm + '_' + state + '_predictions.json')); values = keyed(rows); assert set(values) == set(gold)
            for rid, x in values.items():
                task, row = gold[rid]; rhash, contract = raw(task, row)
                assert x['task'] == task['task'] and x['domain'] == task['domain'] and x['references'] == row['answers']
                assert x['raw_input_sha256'] == rhash and x['image_SHA256'] == contract['images'] and x['image_shapes'] == contract['image_shapes']
                assert x['generation_cap'] == task['max_new_tokens'] and isinstance(x['prediction'], str)
                assert isinstance(x['output_tokens'], list) and len(x['output_tokens']) <= x['generation_cap'] and all(type(t) is int and t >= 0 for t in x['output_tokens'])
                assert x['input_shape'] == [1, x['prompt_tokens']] == x['input_tensor_shapes']['input_ids']
                assert x['input_tensor_shapes']['attention_mask'] == x['input_shape']
                digest_string(x['prepared_sha256']); digest_string(x['transport_sha256'])
            predictions[arm + '_' + state] = rows
        memory[arm] = read(RUN / (arm + '_memory.json')); assert len(memory[arm]) == 5
        final_predictions = keyed(predictions[arm + '_final'])
        for domain, x in zip(DOMAINS, memory[arm]):
            rid = min((rid for rid, (t, _) in gold.items() if t['domain'] == domain), key=lambda rid: hashlib.sha256(rid.encode()).digest())
            task, row = gold[rid]; rhash, _ = raw(task, row)
            assert x['id'] == rid and x['domain'] == domain and x['new_tokens'] == 32 and x['raw_input_sha256'] == rhash
            assert all(type(x[k]) is int and x[k] > 0 for k in ('baseline_allocated', 'peak_allocated', 'peak_reserved'))
            assert x['baseline_allocated'] <= x['peak_allocated'] <= x['peak_reserved']
            for key in ('prepared_sha256', 'transport_sha256', 'input_shape', 'input_tensor_shapes'):
                assert x[key] == final_predictions[rid][key]
    assert initializations['dynamic']['initialization'] == initializations['static']['initialization']
    assert all(a['raw_input_sha256'] == b['raw_input_sha256'] and a['transport_sha256'] == b['transport_sha256'] for a, b in zip(input_records['dynamic'], input_records['static']))
    keyed_predictions = {name: keyed(rows) for name, rows in predictions.items()}
    for rid in gold:
        assert len({rows[rid]['transport_sha256'] for rows in keyed_predictions.values()}) == 1
        first, second = (keyed_predictions[arm + '_initial'][rid] for arm in ARMS)
        assert first['output_tokens'] == second['output_tokens'] and first['prediction'] == second['prediction']
    for a, b in zip(memory['dynamic'], memory['static']):
        assert all(a[key] == b[key] for key in ('id', 'domain', 'raw_input_sha256', 'transport_sha256', 'input_shape', 'input_tensor_shapes', 'new_tokens'))

    nodes = [n for n in ast.parse((port / 'runtime_utils.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ('tokens', 'score')]
    assert {n.name for n in nodes} == {'tokens', 'score'}
    tree = ast.Module(body=nodes, type_ignores=[]); scorer_hash = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    assert scorer_hash == reference['scorer_ast_sha256']
    scope = dict(re=re, Counter=collections.Counter); exec(compile(tree, '<original Answer F1>', 'exec'), scope)
    write('VERIFY_STARTED.json', dict(unix=time.time(), ready_sha256=ready_hash, new_scores=400, scorer_ast_sha256=scorer_hash, all_workers_exited=True))
    quality = {'historical_wide': historical}; actual_scores = 0
    for name, rows in predictions.items():
        per_id = []
        by_id = keyed(rows)
        # Same canonical order as the sealed anchor avoids floating-sum/order
        # artifacts at an exact per-domain point floor. Old scores stay copied.
        for anchor_row in historical['per_id']:
            row = by_id[anchor_row['id']]
            score = scope['score'](row['prediction'], row['references'])['token_f1']; actual_scores += 1
            assert finite(score) and 0 <= score <= 100
            per_id.append(dict(id=row['id'], domain=row['domain'], F1=score))
        domains = {d: dict(F1=sum(x['F1'] for x in per_id if x['domain'] == d) / 20, QA=20) for d in DOMAINS}
        quality[name] = dict(domains=domains, macro=sum(x['F1'] for x in domains.values()) / 5, per_id=per_id)
    assert actual_scores == 400 and quality['dynamic_initial'] == quality['static_initial']
    floor = {arm: all(quality[arm + '_final']['domains'][d]['F1'] >= historical['domains'][d]['F1'] for d in DOMAINS) for arm in ARMS}
    eligible = [arm for arm in ARMS if floor[arm]]
    candidate = max(eligible, key=lambda arm: (quality[arm + '_final']['macro'], arm == 'static')) if eligible else None
    a, b = (keyed(quality[arm + '_final']['per_id']) for arm in ARMS)
    comparison = dict(macro_dynamic_minus_static=quality['dynamic_final']['macro'] - quality['static_final']['macro'],
                      domains={d: quality['dynamic_final']['domains'][d]['F1'] - quality['static_final']['domains'][d]['F1'] for d in DOMAINS},
                      per_id=[dict(id=rid, domain=a[rid]['domain'], dynamic=a[rid]['F1'], static=b[rid]['F1'], difference=a[rid]['F1'] - b[rid]['F1']) for rid in sorted(gold)])
    result = dict(complete=True, ready_sha256=ready_hash, all_exit_sha256=sha(RUN / 'ALL_EXIT.json'),
                  source_COMPLETE_sha256=all_exit['source_COMPLETE_sha256'], scorer_ast_sha256=scorer_hash,
                  quality=quality, paired_comparison=comparison, historical_wide_all_domain_floor=floor, retained_arm=candidate,
                  decision=('retain_' + candidate + '_for_fair_strong_reference_review') if candidate else 'close_fixed_response_selection_pair',
                  checkpoints={arm: phases['train_' + arm]['checkpoint'] for arm in ARMS}, memory=memory, scalar_compile=compiles,
                  phase_seconds={p: x['seconds'] for p, x in phases.items()}, model_phase_seconds=sum(x['seconds'] for x in phases.values()),
                  controller_seconds=all_exit['controller_seconds'], calibration_model_seconds_separate=calibration['seconds'],
                  calibration_COMPLETE_sha256=sha(RUN / 'calibration_COMPLETE.json'), calibration_VERIFIED_sha256=sha(RUN / 'calibration_VERIFIED.json'),
                  new_generations=410, new_scores=400, FIT_backwards=2426, optimizer_updates=174, TEST_QA=0, new_NLL_requests=0,
                  historical_quality_reused_without_rescoring=True, initial_output_tokens_equal=True, matched_QV_only_control=False,
                  PAS_or_folded_OSFT_deployment_superiority_proven=False, method_quality_or_formal_success_proven=False,
                  interpretation='Fixed dynamic/static explicit-branch comparison on explored DEV; historical wide floor is screening only. No matched QV-only efficacy claim or strong-baseline cost victory.',
                  verification_seconds=time.monotonic() - tick)
    write('VERIFIED.json', result)
    print(json.dumps(dict(VERIFIED_sha256=sha(RUN / 'VERIFIED.json'), macro={name: q['macro'] for name, q in quality.items()},
                          floor=floor, candidate=candidate, new_scores=actual_scores, model_phase_seconds=result['model_phase_seconds'])))


if __name__ == '__main__':
    main()
