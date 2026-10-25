"""Verify the plain r8-QV GPU baseline, then score exactly100 new final answers.

No Torch import, checkpoint deserialization, model execution or historical
rescoring. Overall macro F1 is primary; no per-domain nondecrease gate.
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
import time

from common import ROOT, RUN, ASSET, LIMITS, read, sha, write

DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
WEIGHTS = {'coefficients': 893675552, 'bytes': 1787351104}
GRADIENT_BYTES = 2162688
OPTIMIZER_BYTES = 4325760


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest_string(value):
    assert isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def keyed(rows):
    result = {row['id']: row for row in rows}
    assert len(result) == len(rows)
    return result


def expected_wide_plan():
    names = [f'model.layers.{i}.self_attn.v_proj' for i in range(24)]
    chosen = set(sorted(names, key=lambda n: hashlib.sha256(('42:coordinate-remainder:' + n).encode()).hexdigest())[:6])
    return {f'model.layers.{i}.self_attn.{kind}_proj': 65 + int(f'model.layers.{i}.self_attn.{kind}_proj' in chosen)
            for i in range(24) for kind in ('q', 'v')}


def expected_plan():
    return {name: 8 for name in expected_wide_plan()}


def receipt(value, gpu):
    assert value['layout']['weights'] == WEIGHTS and value['layout']['trainable'] == 0
    assert value['layout']['ordinary_buffers']['bytes'] == 9032
    assert value['device'] == f'cuda:{gpu}' and value['response'] is None
    for key in ('digest', 'all_buffers_digest', 'functional_config_digest', 'generation_config_digest', 'language_generation_config_digest'):
        digest_string(value[key])
    assert not any(name.endswith(('.mlp.beta', '.mlp.response_sign', '.mlp.calibration_scale'))
                   for name in value['buffer_manifest'])
    assert all(x['dtype'] == 'torch.float32' for n, x in value['buffer_manifest'].items() if n.endswith('inv_freq'))


def same_receipt(left, right, *, device=True):
    keys = ('layout', 'digest', 'all_buffers_digest', 'buffer_manifest', 'parameter_alias_groups',
            'buffer_alias_groups', 'functional_config_digest', 'generation_config_digest',
            'language_generation_config_digest', 'language_attention_classes', 'vision_attention_classes', 'response')
    for key in keys:
        assert left[key] == right[key], key
    if device:
        assert left['device'] == right['device']


def same_function(left_path, right_path, name, *, r8_rank_guard=False):
    def find(path):
        source = path.read_text()
        if r8_rank_guard:
            source = source.replace('rank in (65,66)', 'rank ==8')
        return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(find(left_path), include_attributes=False) == ast.dump(find(right_path), include_attributes=False), name


def main():
    tick = time.monotonic()
    assert not (RUN / 'VERIFY_STARTED.json').exists() and not (RUN / 'VERIFIED.json').exists()
    assert not list(RUN.glob('*_FAILURE.json'))
    ready = read(RUN / 'READY.json'); ready_hash = sha(RUN / 'READY.json')
    for field in ('sources', 'prior_files'):
        for name, expected in ready[field].items():
            assert sha(RUN / name) == expected, name
    for name, expected in ready['environment'].items():
        assert sha(name) == expected, name
    for name, expected in ready['runtime_files'].items():
        if not name.endswith('.safetensors'):
            assert sha(ASSET / name) == expected, name
    assert sha(ROOT / 'exp/ffn_static_delete_20261025/v1/core.py') == ready['storage_helper_sha256']
    priors = {'calibration_COMPLETE.json', 'prior_pair_READY.json', 'prior_pair_VERIFIED.json',
              'prior_train_dynamic_COMPLETE.json', 'prior_dynamic_ADAPTER_INITIAL.json',
              'prior_static_ADAPTER_INITIAL.json', 'prior_dynamic_FIT_INPUTS.json',
              'prior_dynamic_final_predictions.json', 'prior_static_final_predictions.json',
              'prior_eval_dynamic_COMPLETE.json', 'prior_eval_static_COMPLETE.json',
              'prior_qv_READY.json', 'prior_qv_VERIFIED.json', 'prior_qv_train_COMPLETE.json',
              'prior_qv_eval_COMPLETE.json', 'prior_qv_ADAPTER_INITIAL.json', 'prior_qv_final_predictions.json'}
    assert priors <= set(ready['prior_files'])
    pair_ready = read(RUN / 'prior_pair_READY.json'); pair = read(RUN / 'prior_pair_VERIFIED.json')
    assert pair['complete'] and pair['ready_sha256'] == sha(RUN / 'prior_pair_READY.json')
    assert pair['new_scores'] == 400 and pair['new_generations'] == 410
    calibration = read(RUN / 'calibration_COMPLETE.json')
    assert sha(RUN / 'calibration_COMPLETE.json') == pair['calibration_COMPLETE_sha256'] == pair_ready['prior_files']['calibration_COMPLETE.json']
    assert calibration['state_and_buffers_unchanged'] and calibration['layout']['weights'] == WEIGHTS
    old_train = read(RUN / 'prior_train_dynamic_COMPLETE.json')
    assert sha(RUN / 'prior_train_dynamic_COMPLETE.json') == pair['source_COMPLETE_sha256']['train_dynamic']
    assert old_train['ready_sha256'] == pair['ready_sha256']
    assert old_train['files']['dynamic_ADAPTER_INITIAL.json'] == sha(RUN / 'prior_dynamic_ADAPTER_INITIAL.json')
    assert old_train['files']['dynamic_FIT_INPUTS.json'] == sha(RUN / 'prior_dynamic_FIT_INPUTS.json')
    # Static initialization was also compared with dynamic in the sealed pair
    # verifier. Its copied file hash is bound by this new READY.
    old_initials = {arm: read(RUN / ('prior_' + arm + '_ADAPTER_INITIAL.json')) for arm in ('dynamic', 'static')}
    assert old_initials['dynamic']['initialization'] == old_initials['static']['initialization']
    assert old_initials['dynamic']['rank_plan'] == old_initials['static']['rank_plan'] == expected_wide_plan()
    for arm in ('dynamic', 'static'):
        old_eval = read(RUN / ('prior_eval_' + arm + '_COMPLETE.json'))
        assert sha(RUN / ('prior_eval_' + arm + '_COMPLETE.json')) == pair['source_COMPLETE_sha256']['eval_' + arm]
        assert old_eval['ready_sha256'] == pair['ready_sha256']
        assert old_eval['files'][arm + '_final_predictions.json'] == sha(RUN / ('prior_' + arm + '_final_predictions.json'))
    pair_source = Path(ready['prior_pair_run'])
    for name in ('paired_adaptation.py', 'paired_readouts.py'):
        assert sha(pair_source / name) == pair_ready['sources'][name]
    assert sha(RUN / 'readouts.py') == pair_ready['sources']['paired_readouts.py']
    for name in ('configure_gpu_math', 'fold'):
        same_function(RUN / 'control_adaptation.py', pair_source / 'paired_adaptation.py', name, r8_rank_guard=name == 'fold')
    wide_ready = read(RUN / 'prior_qv_READY.json'); wide = read(RUN / 'prior_qv_VERIFIED.json')
    assert wide['complete'] and wide['ready_sha256'] == sha(RUN / 'prior_qv_READY.json')
    assert wide['new_scores'] == 100 and wide['new_generations'] == 105
    assert wide['prior_pair_VERIFIED_sha256'] == sha(RUN / 'prior_pair_VERIFIED.json')
    assert wide['prior_pair_READY_sha256'] == sha(RUN / 'prior_pair_READY.json')
    assert wide_ready['prior_files']['calibration_COMPLETE.json'] == sha(RUN / 'calibration_COMPLETE.json')
    for phase, copied in (('train_qv', 'prior_qv_train_COMPLETE.json'), ('eval_qv', 'prior_qv_eval_COMPLETE.json')):
        old_complete = read(RUN / copied)
        assert old_complete['ready_sha256'] == wide['ready_sha256']
        assert sha(RUN / copied) == wide['source_COMPLETE_sha256'][phase]
    wide_train = read(RUN / 'prior_qv_train_COMPLETE.json'); wide_eval = read(RUN / 'prior_qv_eval_COMPLETE.json')
    assert wide_train['files']['qv_ADAPTER_INITIAL.json'] == sha(RUN / 'prior_qv_ADAPTER_INITIAL.json')
    assert wide_eval['files']['qv_final_predictions.json'] == sha(RUN / 'prior_qv_final_predictions.json')
    wide_initial = read(RUN / 'prior_qv_ADAPTER_INITIAL.json')
    assert wide_initial['rank_plan'] == expected_wide_plan() and wide_initial['coordinates'] == 4399104
    wide_source = Path(ready['prior_qv_run'])
    assert sha(wide_source / 'control_adaptation.py') == wide_ready['sources']['control_adaptation.py']
    assert sha(wide_source / 'readouts.py') == wide_ready['sources']['readouts.py'] == sha(RUN / 'readouts.py')

    all_exit = read(RUN / 'ALL_EXIT.json')
    assert all_exit['complete'] and all_exit['ready_sha256'] == ready_hash
    for key, expected in dict(new_generations=105, new_scores=0, FIT_backwards=1213, optimizer_updates=87, TEST_QA=0).items():
        assert all_exit[key] == expected
    assert LIMITS == {'train_qv': 3600, 'eval_qv': 600}
    assert set(all_exit['source_COMPLETE_sha256']) == set(LIMITS)
    phases, starts, exits = {}, {}, {}
    for phase, limit in LIMITS.items():
        start = read(RUN / (phase + '_STARTED.json')); end = read(RUN / (phase + '_COMPLETE.json')); ex = read(RUN / (phase + '_EXIT.json'))
        identity = start['identity']
        assert identity == end['identity'] and identity['pid'] == identity['pgid'] == ex['pid'] and identity['uid'] == os.getuid()
        proc = Path('/proc') / str(identity['pid']) / 'stat'
        if proc.exists():
            assert proc.read_text().rsplit(') ', 1)[1].split()[19] != str(identity['startticks']), 'Original worker still present'
        assert ex['returncode'] == 0 and start['ready_sha256'] == end['ready_sha256'] == ready_hash
        assert start['limit'] == limit and finite(end['seconds']) and 0 < end['seconds'] <= limit
        assert finite(start['unix']) and finite(ex['unix']) and ex['unix'] >= start['unix'] + end['seconds']
        assert start['GPU'] == ready['phase_GPUs'][phase]
        assert sha(RUN / (phase + '_COMPLETE.json')) == all_exit['source_COMPLETE_sha256'][phase]
        for name, expected in end['files'].items():
            assert sha(RUN / name) == expected, name
        phases[phase], starts[phase], exits[phase] = end, start, ex
    assert starts['train_qv']['boot'] == starts['eval_qv']['boot']
    assert starts['eval_qv']['unix'] >= exits['train_qv']['unix']
    assert starts['train_qv']['GPU'] == pair_ready['phase_GPUs']['train_dynamic'] == 0
    assert starts['eval_qv']['GPU'] == pair_ready['phase_GPUs']['eval_dynamic'] == pair_ready['phase_GPUs']['eval_static'] == 1
    assert sum(x['seconds'] for x in phases.values()) <= 4200

    port = ASSET / 'exp/two_backbone_native_cl_20261017_v2'
    policy = read(port / 'execution_policy.json'); manifest_path = ASSET / policy['conditions']['pct01']['manifest']; manifest = read(manifest_path)
    split_path = ASSET / 'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'; split = read(split_path)
    assert sha(manifest_path) == ready['manifest_sha256'] == pair_ready['manifest_sha256'] == wide_ready['manifest_sha256']
    assert sha(split_path) == ready['split_sha256'] == pair_ready['split_sha256'] == wide_ready['split_sha256']
    gold, fit_rows, order, plan = {}, {}, [], []
    for task in manifest['tasks']:
        stage = task['task']; path = ASSET / task['files']['train']['path']
        assert sha(path) == task['files']['train']['sha256']
        values = keyed([json.loads(line) for line in path.read_text().splitlines()])
        fit_ids, dev_ids = split['fit_ids'][str(stage)], split['dev_ids'][str(stage)]
        assert set(fit_ids).isdisjoint(dev_ids) and set(fit_ids) | set(dev_ids) == set(values)
        assert not (set(values) & (set(gold) | set(fit_rows)))
        fit_rows.update({rid: (task, values[rid]) for rid in fit_ids}); gold.update({rid: (task, values[rid]) for rid in dev_ids})
        ordered = sorted(fit_ids, key=lambda rid: hashlib.sha256(f'42:{stage}:{rid}'.encode()).digest()); order.extend(ordered)
        plan.append(dict(stage=stage, QA=len(ordered), updates=(len(ordered) + 15) // 16,
                         order_sha256=hashlib.sha256(''.join(rid + '\n' for rid in ordered).encode()).hexdigest()))
    assert [p['stage'] for p in plan] == list(range(1, 20)) and plan == ready['training_plan'] == pair_ready['training_plan']
    assert len(fit_rows) == 1213 and len(gold) == 100 and sum(p['updates'] for p in plan) == 87
    assert set(fit_rows).isdisjoint(gold) and all(sum(t['domain'] == d for t, _ in gold.values()) == 20 for d in DOMAINS)
    original_main = next(n for n in ast.parse((port / 'run.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    original_loop = next(n for n in original_main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'stage')
    cutoff = next(i for i, n in enumerate(original_loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'wanted' for t in n.targets))
    prefix = ast.Module(body=original_loop.body[:cutoff - 1], type_ignores=[])
    assert hashlib.sha256(ast.dump(prefix, include_attributes=False).encode()).hexdigest() == ready['training_ast_sha256'] == pair_ready['training_ast_sha256']

    spec = importlib.util.spec_from_file_location('qv_raw_retrieval', port / 'retrieval.py')
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

    tr, ev = phases['train_qv'], phases['eval_qv']
    train_gpu, eval_gpu = starts['train_qv']['GPU'], starts['eval_qv']['GPU']
    assert tr['arm'] == ev['arm'] == 'qv'
    assert tr['new_generations'] == 0 and tr['FIT_backwards'] == 1213 and tr['optimizer_updates'] == 87 and tr['coordinates'] == 540672
    assert tr['training_plan'] == plan and tr['training_ast_sha256'] == ready['training_ast_sha256']
    assert tr['frozen_unchanged'] and tr['buffers_unchanged'] and tr['adapter_before'] != tr['adapter_after']
    assert tr['original_r8_initialization_inherited'] and tr['paired_training_inputs_exact'] and tr['no_response_module']
    required = {'qv_ADAPTER_INITIAL.json', 'qv_FIT_INPUTS.json', 'qv_TRAIN_MEMORY.json', 'qv_initial.pt', 'qv_complete.pt', 'qv_training/logs/events.jsonl'}
    for stage in plan:
        sid = stage['stage']
        required.update({f'qv_STAGE_{sid:02}_COMPLETE.json', f'qv_adapter_S{sid:02}.pt', f'qv_training/validity/stage_{sid:02}.json'})
        required.update(f'qv_grad_S{sid:02}_{i:03}.json' for i in range(1, stage['updates'] + 1))
    assert required <= set(tr['files']) and set(ev['files']) == {'qv_memory.json', 'qv_final_predictions.json'}
    assert len(list(RUN.glob('qv_grad_S*.json'))) == 87 and len(list(RUN.glob('qv_STAGE_*_COMPLETE.json'))) == 19
    assert len(list(RUN.glob('qv_adapter_S*.pt'))) == 19
    for field in ('original_base', 'initial_checkpoint', 'initial_reload', 'initial_invariant', 'checkpoint'):
        receipt(tr[field], train_gpu)
    same_receipt(tr['original_base'], old_train['original_base'], device=False)
    assert tr['original_base']['digest'] == calibration['initial_state_digest']
    assert tr['original_base']['all_buffers_digest'] == calibration['all_buffers_digest']
    assert tr['original_base']['layout'] == calibration['layout']
    same_receipt(tr['original_base'], tr['initial_checkpoint']); same_receipt(tr['initial_checkpoint'], tr['initial_reload'])
    same_receipt(tr['initial_reload'], tr['initial_invariant']); assert tr['initial_invariant']['complete_state_and_buffers_unchanged']
    for field in ('self_contained_meta_construction', 'all_buffers_explicitly_restored', 'strict_complete_reload', 'mixed_parameter_dtypes_preserved'):
        assert tr['initial_reload'][field] and ev['reloaded'][field]
    for state, field in (('initial', 'initial_checkpoint'), ('complete', 'checkpoint')):
        name = 'qv_' + state + '.pt'; cp = tr[field]
        assert (RUN / name).stat().st_size == cp['bytes'] and cp['sha256'] == tr['files'][name]
        assert Path(cp['path']).resolve() == (RUN / name).resolve()
        assert cp['complete_graph'] and not cp['external_base_weights_required'] and cp['no_adapter_optimizer_or_teacher'] and cp['alias_preserving_cpu_snapshot']
    assert ev['new_generations'] == 105 and ev['FIT_backwards'] == ev['optimizer_updates'] == 0
    receipt(ev['reloaded'], eval_gpu); receipt(ev['invariant'], eval_gpu)
    same_receipt(tr['checkpoint'], ev['reloaded'], device=False); same_receipt(ev['reloaded'], ev['invariant'])
    assert ev['invariant']['complete_state_and_buffers_unchanged']
    fold = tr['fold']; rank_plan = expected_plan()
    assert len(fold) == 48 and {x['module']: x['rank'] for x in fold} == rank_plan
    assert all(x['scale'] == 4 and x['dtype'] == 'torch.float16' and x['merge_dtype'] == 'torch.float32' and x['device'] == f'cuda:{train_gpu}'
               and not x['autocast_enabled'] and not x['matmul_allow_tf32'] for x in fold)
    initial = read(RUN / 'qv_ADAPTER_INITIAL.json')
    assert initial['coordinates'] == 540672 and initial['parameters'] == 96 and initial['rank_plan'] == rank_plan
    assert initial['original_r8_initialization_inherited'] and initial['no_response_module']
    assert len(initial['initialization']) == 48 and {x['module']: x['rank'] for x in initial['initialization']} == rank_plan
    assert all(x['B_zero'] and x['scale'] == 4 and x['dropout'] == .1 and x['prefix_r8_exact'] and x['alpha'] == 32 for x in initial['initialization'])
    for x in initial['initialization']:
        digest_string(x['A_digest']); digest_string(x['B_digest'])
    for stage in plan:
        sid, updates = stage['stage'], stage['updates']; st = read(RUN / f'qv_STAGE_{sid:02}_COMPLETE.json')
        valid = read(RUN / f'qv_training/validity/stage_{sid:02}.json')
        assert st['stage'] == sid and st['updates'] == updates and st['bytes'] == GRADIENT_BYTES
        assert st['optimizer_state_bytes'] == OPTIMIZER_BYTES and st['all_optimizer_state_GPU'] and st['frozen_unchanged']
        digest_string(st['adapter_digest'])
        assert valid['passed'] and valid['actual_parameters_changed'] and valid['updates'] == valid['positive_lr_updates'] == updates
        assert valid['parameters_before_sha256'] != valid['parameters_after_sha256']
        for i in range(1, updates + 1):
            assert read(RUN / f'qv_grad_S{sid:02}_{i:03}.json') == dict(parameters=96, finite=True, gradient_bytes=GRADIENT_BYTES)
    events = [json.loads(line) for line in (RUN / 'qv_training/logs/events.jsonl').read_bytes().splitlines()]
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
    inputs = read(RUN / 'qv_FIT_INPUTS.json')
    assert inputs == read(RUN / 'prior_dynamic_FIT_INPUTS.json')
    assert len(inputs) == 1213 and [x['id'] for x in inputs] == order
    for x in inputs:
        task, row = fit_rows[x['id']]; rhash, contract = raw(task, row)
        assert x['phase'] == 'TRAIN' and x['task'] == task['task'] and x['domain'] == task['domain']
        assert x['raw_input_sha256'] == rhash and x['image_SHA256'] == contract['images'] and x['image_shapes'] == contract['image_shapes']
        digest_string(x['transport_sha256'])

    rows = read(RUN / 'qv_final_predictions.json'); predictions = keyed(rows)
    assert set(predictions) == set(gold)
    old_predictions = {arm: keyed(read(RUN / ('prior_' + arm + '_final_predictions.json'))) for arm in ('dynamic', 'static')}
    old_predictions['wide_qv'] = keyed(read(RUN / 'prior_qv_final_predictions.json'))
    for arm in old_predictions:
        assert set(old_predictions[arm]) == set(gold)
    for rid, x in predictions.items():
        task, row = gold[rid]; rhash, contract = raw(task, row)
        assert x['task'] == task['task'] and x['domain'] == task['domain'] and x['references'] == row['answers']
        assert x['raw_input_sha256'] == rhash and x['image_SHA256'] == contract['images'] and x['image_shapes'] == contract['image_shapes']
        assert x['generation_cap'] == task['max_new_tokens'] and isinstance(x['prediction'], str)
        assert isinstance(x['output_tokens'], list) and len(x['output_tokens']) <= x['generation_cap'] and all(type(t) is int and t >= 0 for t in x['output_tokens'])
        assert x['input_shape'] == [1, x['prompt_tokens']] == x['input_tensor_shapes']['input_ids']
        assert x['input_tensor_shapes']['attention_mask'] == x['input_shape']
        digest_string(x['prepared_sha256']); digest_string(x['transport_sha256'])
        for arm in old_predictions:
            old = old_predictions[arm][rid]
            for key in ('task', 'domain', 'references', 'generation_cap', 'raw_input_sha256', 'prepared_sha256',
                        'transport_sha256', 'image_SHA256', 'image_shapes', 'input_shape', 'input_tensor_shapes'):
                assert x[key] == old[key], key
    memory = read(RUN / 'qv_memory.json'); assert len(memory) == 5
    for index, (domain, x) in enumerate(zip(DOMAINS, memory)):
        rid = min((rid for rid, (t, _) in gold.items() if t['domain'] == domain), key=lambda rid: hashlib.sha256(rid.encode()).digest())
        assert x['id'] == rid and x['domain'] == domain and x['new_tokens'] == 32
        assert all(type(x[k]) is int and x[k] > 0 for k in ('baseline_allocated', 'peak_allocated', 'peak_reserved'))
        assert x['baseline_allocated'] <= x['peak_allocated'] <= x['peak_reserved']
        for key in ('prepared_sha256', 'transport_sha256', 'raw_input_sha256', 'input_shape', 'input_tensor_shapes'):
            assert x[key] == predictions[rid][key]
        for arm in old_predictions:
            old = (wide['memory']['qv'] if arm == 'wide_qv' else pair['memory'][arm])[index]
            for key in ('id', 'domain', 'raw_input_sha256', 'prepared_sha256', 'transport_sha256', 'input_shape', 'input_tensor_shapes', 'new_tokens'):
                assert x[key] == old[key], key

    nodes = [n for n in ast.parse((port / 'runtime_utils.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ('tokens', 'score')]
    assert {n.name for n in nodes} == {'tokens', 'score'}
    tree = ast.Module(body=nodes, type_ignores=[]); scorer_hash = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    assert scorer_hash == pair['scorer_ast_sha256']
    scope = dict(re=re, Counter=collections.Counter); exec(compile(tree, '<original Answer F1>', 'exec'), scope)
    old_quality = {arm: pair['quality'][arm + '_final'] for arm in ('dynamic', 'static')}
    old_quality['wide_qv'] = wide['quality']['qv_final']
    anchor_order = [x['id'] for x in old_quality['dynamic']['per_id']]
    assert len(anchor_order) == 100 and set(anchor_order) == set(gold)
    assert anchor_order == [x['id'] for x in old_quality['static']['per_id']]
    write('VERIFY_STARTED.json', dict(unix=time.time(), ready_sha256=ready_hash, new_scores=100,
                                     scorer_ast_sha256=scorer_hash, both_workers_exited=True,
                                     historical_scores_recomputed=0))
    per_id = []
    for rid in anchor_order:
        row = predictions[rid]; score = scope['score'](row['prediction'], row['references'])['token_f1']
        assert finite(score) and 0 <= score <= 100
        per_id.append(dict(id=rid, domain=row['domain'], F1=score))
    assert len(per_id) == 100
    domains = {d: dict(F1=sum(x['F1'] for x in per_id if x['domain'] == d) / 20, QA=20) for d in DOMAINS}
    qv_quality = dict(domains=domains, macro=sum(x['F1'] for x in domains.values()) / 5, per_id=per_id)
    qv_keyed = keyed(per_id); comparisons = {}
    for arm, prior in old_quality.items():
        prior_keyed = keyed(prior['per_id'])
        assert set(prior_keyed) == set(gold) and set(prior['domains']) == set(DOMAINS)
        delta = prior['macro'] - qv_quality['macro']
        comparisons[arm + '_minus_plain_r8'] = dict(macro=delta,
            domains={d: prior['domains'][d]['F1'] - domains[d]['F1'] for d in DOMAINS},
            per_id=[dict(id=rid, domain=qv_keyed[rid]['domain'], prior_F1=prior_keyed[rid]['F1'],
                         plain_r8_F1=qv_keyed[rid]['F1'], difference=prior_keyed[rid]['F1'] - qv_keyed[rid]['F1']) for rid in anchor_order])
    result = dict(complete=True, ready_sha256=ready_hash, all_exit_sha256=sha(RUN / 'ALL_EXIT.json'),
        source_COMPLETE_sha256=all_exit['source_COMPLETE_sha256'], scorer_ast_sha256=scorer_hash,
        prior_pair_VERIFIED_sha256=sha(RUN / 'prior_pair_VERIFIED.json'), prior_pair_READY_sha256=sha(RUN / 'prior_pair_READY.json'),
        prior_qv_VERIFIED_sha256=sha(RUN / 'prior_qv_VERIFIED.json'), prior_qv_READY_sha256=sha(RUN / 'prior_qv_READY.json'),
        quality=dict(plain_r8_final=qv_quality, wide_qv_final=old_quality['wide_qv'], dynamic_final=old_quality['dynamic'], static_final=old_quality['static']),
        historical_context_minus_plain_r8=comparisons,
        decision='plain_r8_GPU_baseline_measured_no_new_method_claim',
        per_domain_non_decrease_gate_used=False, matched_GPU_plain_r8_control=True,
        training_coordinates=540672, trainable_parameter_tensors=96, gradient_bytes=GRADIENT_BYTES, GPU_optimizer_state_bytes=OPTIMIZER_BYTES,
        complete_checkpoint=tr['checkpoint'], memory=dict(plain_r8=memory, wide_qv=wide['memory']['qv'], dynamic=pair['memory']['dynamic'], static=pair['memory']['static']),
        phase_seconds={p: value['seconds'] for p, value in phases.items()}, model_phase_seconds=sum(value['seconds'] for value in phases.values()),
        controller_seconds=all_exit['controller_seconds'], prior_pair_model_seconds_separate=pair['model_phase_seconds'], prior_wide_model_seconds_separate=wide['model_phase_seconds'],
        new_generations=105, new_scores=100, FIT_backwards=1213, optimizer_updates=87, TEST_QA=0,
        extra_diagnostic_NLL_requests=0, initial_DEV_generations=0, initial_base_output_equivalence_measured=False,
        initial_complete_state_reload_verified=True, historical_quality_reused_without_rescoring=True,
        same_GPU_for_final_resource_windows=1, PAS_or_strongest_baseline_quality_superiority_proven=False,
        framework_or_novel_method_success_proven=False, precision_search=False,
        interpretation='Plain q/v-r8 GPU baseline for a later separately reviewed same-rank CL intervention. Historical wide-QV/dynamic/static values are context, not innovation gains; rank budgets differ. Explored100DEV, single seed, overall macro primary and all domains/IDs retained. No significance, equivalence, noninferiority, or formal strongest-baseline claim.',
        verification_seconds=time.monotonic() - tick)
    write('VERIFIED.json', result)
    print(json.dumps(dict(VERIFIED_sha256=sha(RUN / 'VERIFIED.json'), plain_r8_macro=qv_quality['macro'],
                          historical_context_minus_plain_r8={arm: value['macro'] for arm, value in comparisons.items()},
                          new_scores=100, model_phase_seconds=result['model_phase_seconds'])))


if __name__ == '__main__':
    main()
