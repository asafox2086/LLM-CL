"""Validate six blind phases, then score exactly300 new terminal answers once.

Only standard metadata, source ASTs, hashes and the original scorer are used.
No Torch import, checkpoint deserialization or historical answer rescoring.
"""
from pathlib import Path
import ast
import collections
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import time
from common import ROOT, RUN, ASSET, ARMS, LIMITS, read, sha, write

DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
WEIGHTS = dict(coefficients=893675552, bytes=1787351104)
COORDINATES, PARAMETERS, GRADIENT_BYTES, OPTIMIZER_BYTES = 540672, 96, 2162688, 4325760
PHASES = tuple('train_' + a for a in ARMS) + tuple('eval_' + a for a in ARMS)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(value):
    assert isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def keyed(rows):
    result = {row['id']: row for row in rows}
    assert len(result) == len(rows)
    return result


def rank_plan():
    return {f'model.layers.{i}.self_attn.{kind}_proj': 8 for i in range(24) for kind in ('q', 'v')}


def same_receipt(left, right, device=True):
    for key in ('layout', 'digest', 'all_buffers_digest', 'buffer_manifest',
                'parameter_alias_groups', 'buffer_alias_groups', 'functional_config_digest',
                'generation_config_digest', 'language_generation_config_digest',
                'language_attention_classes', 'vision_attention_classes', 'response'):
        assert left[key] == right[key], key
    if device:
        assert left['device'] == right['device']


def receipt(value, gpu):
    assert value['layout']['weights'] == WEIGHTS and value['layout']['trainable'] == 0
    assert value['layout']['ordinary_buffers']['bytes'] == 9032
    assert value['device'] == f'cuda:{gpu}' and value['response'] is None
    for key in ('digest', 'all_buffers_digest', 'functional_config_digest',
                'generation_config_digest', 'language_generation_config_digest'):
        digest(value[key])
    assert not any(n.endswith(('._ilora_slow_A', '._ilora_slow_B', '.mlp.response_sign', '.mlp.calibration_scale'))
                   for n in value['buffer_manifest'])
    assert all(x['dtype'] == 'torch.float32' for n, x in value['buffer_manifest'].items() if n.endswith('inv_freq'))


def function_ast(path, name):
    return next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def same_function(left, right, name):
    assert ast.dump(function_ast(left, name), include_attributes=False) == ast.dump(function_ast(right, name), include_attributes=False), name


def key(task, row):
    return dict(task=task['task'], id=row['id'], domain=task['domain'])


def simulated_replay(manifest, fit):
    """Independent metadata reconstruction; no worker imports or model calls."""
    pool, seen, global_step = [], 0, 0
    inserts, draws = random.Random(42), random.Random(42)
    stages, steps, memory_events, all_order, all_replay = [], [], [], [], []
    for task in manifest['tasks']:
        stage = task['task']
        order = sorted(fit[stage], key=lambda r: hashlib.sha256(f"42:{stage}:{r['id']}".encode()).digest())
        start = dict(stage=stage, cached_QA=len(pool), seen_past_QA=seen, cached_IDs=list(pool))
        for step, offset in enumerate(range(0, len(order), 16), 1):
            current = [key(task, r) for r in order[offset:offset + 16]]
            replay = [pool[draws.randrange(len(pool))] for _ in current] if pool else []
            assert all(r['task'] < stage for r in replay)
            global_step += 1
            steps.append(dict(stage=stage, step=step, global_update=global_step,
                              stage_updates=(len(order) + 15) // 16, current_QA=current, replay_QA=replay))
            all_order.extend(x['id'] for x in current)
            all_replay.extend(replay)
        for row in order:
            before = seen
            seen += 1
            item = key(task, row)
            if len(pool) < 500:
                slot = len(pool)
                pool.append(item)
                retained = True
            else:
                slot = inserts.randrange(seen)
                retained = slot < 500
                if retained:
                    pool[slot] = item
            memory_events.append(dict(stage=stage, source=item, seen_before=before,
                                      draw=slot, retained=retained, cached_QA=len(pool)))
        encoded = json.dumps(pool, sort_keys=True, separators=(',', ':')).encode()
        end = dict(stage=stage, cached_QA=len(pool), seen_QA=seen,
                   reference_ledger_bytes=len(encoded), reference_ledger_sha256=hashlib.sha256(encoded).hexdigest())
        stages.append((start, end))
    assert seen == 1213 and len(pool) == 500 and len(steps) == 87 and len(all_replay) == 886
    return stages, steps, memory_events, all_order, all_replay


def main():
    tick = time.monotonic()
    assert not (RUN / 'VERIFY_STARTED.json').exists() and not (RUN / 'VERIFIED.json').exists()
    assert not list(RUN.glob('*_FAILURE.json'))
    ready = read(RUN / 'READY.json'); rh = sha(RUN / 'READY.json')
    for field in ('sources', 'prior_files'):
        for name, expected in ready[field].items():
            assert sha(RUN / name) == expected, name
    for name, expected in ready['environment'].items():
        assert sha(name) == expected, name
    for name, expected in ready['runtime_files'].items():
        if not name.endswith('.safetensors'):
            assert sha(ASSET / name) == expected, name
    assert sha(ROOT / 'exp/ffn_static_delete_20261025/v1/core.py') == ready['storage_helper_sha256']
    required_prior = {'calibration_COMPLETE.json', 'prior_naive_READY.json', 'prior_naive_VERIFIED.json',
                      'prior_naive_train_COMPLETE.json', 'prior_naive_eval_COMPLETE.json',
                      'prior_naive_ADAPTER_INITIAL.json', 'prior_naive_FIT_INPUTS.json', 'prior_naive_final_predictions.json'}
    assert required_prior <= set(ready['prior_files'])
    naive_ready = read(RUN / 'prior_naive_READY.json'); naive = read(RUN / 'prior_naive_VERIFIED.json')
    naive_train = read(RUN / 'prior_naive_train_COMPLETE.json'); naive_eval = read(RUN / 'prior_naive_eval_COMPLETE.json')
    naive_initial = read(RUN / 'prior_naive_ADAPTER_INITIAL.json'); naive_inputs = read(RUN / 'prior_naive_FIT_INPUTS.json')
    assert naive['complete'] and naive['matched_GPU_plain_r8_control'] and naive['ready_sha256'] == sha(RUN / 'prior_naive_READY.json')
    assert naive['new_scores'] == 100 and naive['new_generations'] == 105 and naive['TEST_QA'] == 0
    for phase, path, value in (('train_qv', 'prior_naive_train_COMPLETE.json', naive_train),
                               ('eval_qv', 'prior_naive_eval_COMPLETE.json', naive_eval)):
        assert sha(RUN / path) == naive['source_COMPLETE_sha256'][phase] and value['ready_sha256'] == naive['ready_sha256']
    for original, copied in (('qv_ADAPTER_INITIAL.json', 'prior_naive_ADAPTER_INITIAL.json'),
                             ('qv_FIT_INPUTS.json', 'prior_naive_FIT_INPUTS.json')):
        assert naive_train['files'][original] == sha(RUN / copied)
    assert naive_eval['files']['qv_final_predictions.json'] == sha(RUN / 'prior_naive_final_predictions.json')
    assert naive_ready['prior_files']['calibration_COMPLETE.json'] == sha(RUN / 'calibration_COMPLETE.json')
    calibration = read(RUN / 'calibration_COMPLETE.json')
    assert calibration['state_and_buffers_unchanged'] and calibration['layout']['weights'] == WEIGHTS
    prior_source = Path(ready['prior_naive_run'])
    for name in ('control_adaptation.py', 'control_deployment.py', 'readouts.py'):
        assert sha(prior_source / name) == naive_ready['sources'][name], name
    assert sha(RUN / 'deployment.py') == naive_ready['sources']['control_deployment.py']
    assert sha(RUN / 'readouts.py') == naive_ready['sources']['readouts.py']
    for name in ('configure_gpu_math', 'fold'):
        same_function(RUN / 'adaptation.py', prior_source / 'control_adaptation.py', name)

    assert ARMS == ('gate', 'original', 'shuffle')
    assert LIMITS == {**{'train_' + a: 5400 for a in ARMS}, **{'eval_' + a: 600 for a in ARMS}}
    assert ready['phase_GPUs'] == {p: (2 if p.startswith('train_') else 1) for p in PHASES}
    all_exit = read(RUN / 'ALL_EXIT.json')
    assert all_exit['complete'] and all_exit['ready_sha256'] == rh and not all_exit['naive_training_repeated']
    assert set(all_exit['source_COMPLETE_sha256']) == set(PHASES)
    for n, v in dict(new_generations=315, new_scores=0, FIT_backwards=3639, current_QA=3639,
                     replay_QA=2658, optimizer_updates=261, TEST_QA=0).items():
        assert all_exit[n] == v
    assert finite(all_exit['controller_seconds']) and 0 < all_exit['controller_seconds'] <= 18300
    starts, ends, exits = {}, {}, {}
    for phase in PHASES:
        start, end, ex = (read(RUN / (phase + suffix)) for suffix in ('_STARTED.json', '_COMPLETE.json', '_EXIT.json'))
        who = start['identity']; assert who == end['identity']
        assert who['pid'] == who['pgid'] == ex['pid'] and who['uid'] == os.getuid()
        path = Path('/proc') / str(who['pid']) / 'stat'
        if path.exists():
            assert path.read_text().rsplit(') ', 1)[1].split()[19] != str(who['startticks']), 'Original worker still alive'
        assert ex['returncode'] == 0 and not ex.get('controller_interrupted', False)
        assert start['ready_sha256'] == end['ready_sha256'] == rh and start['limit'] == LIMITS[phase]
        assert start['GPU'] == ready['phase_GPUs'][phase]
        assert finite(end['seconds']) and 0 < end['seconds'] <= LIMITS[phase]
        assert finite(start['unix']) and finite(ex['unix']) and ex['unix'] >= start['unix'] + end['seconds']
        assert sha(RUN / (phase + '_COMPLETE.json')) == all_exit['source_COMPLETE_sha256'][phase]
        for name, expected in end['files'].items():
            assert sha(RUN / name) == expected, name
        starts[phase], ends[phase], exits[phase] = start, end, ex
    assert len({x['boot'] for x in starts.values()}) == 1
    assert all(starts[b]['unix'] >= exits[a]['unix'] for a, b in zip(PHASES, PHASES[1:]))
    assert sum(x['seconds'] for x in ends.values()) <= 18000

    port = ASSET / 'exp/two_backbone_native_cl_20261017_v2'
    policy = read(port / 'execution_policy.json'); manifest_path = ASSET / policy['conditions']['pct01']['manifest']
    manifest = read(manifest_path)
    split_path = ASSET / 'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'; split = read(split_path)
    assert sha(manifest_path) == ready['manifest_sha256'] == naive_ready['manifest_sha256']
    assert sha(split_path) == ready['split_sha256'] == naive_ready['split_sha256']
    assert [t['task'] for t in manifest['tasks']] == list(range(1, 20))
    fit, fit_by_id, gold, plan = {}, {}, {}, []
    for task in manifest['tasks']:
        stage = task['task']; path = ASSET / task['files']['train']['path']
        assert sha(path) == task['files']['train']['sha256']
        rows = keyed([json.loads(line) for line in path.read_text(encoding='utf8').splitlines()])
        fit_ids, dev_ids = split['fit_ids'][str(stage)], split['dev_ids'][str(stage)]
        assert set(fit_ids).isdisjoint(dev_ids) and set(fit_ids) | set(dev_ids) == set(rows)
        assert not (set(rows) & (set(fit_by_id) | set(gold)))
        fit[stage] = [rows[rid] for rid in fit_ids]
        fit_by_id.update({rid: (task, rows[rid]) for rid in fit_ids}); gold.update({rid: (task, rows[rid]) for rid in dev_ids})
        order = sorted(fit_ids, key=lambda rid: hashlib.sha256(f'42:{stage}:{rid}'.encode()).digest())
        plan.append(dict(stage=stage, QA=len(order), updates=(len(order) + 15) // 16,
                         order_sha256=hashlib.sha256(''.join(rid + '\n' for rid in order).encode()).hexdigest()))
    assert len(fit_by_id) == 1213 and len(gold) == 100 and sum(x['updates'] for x in plan) == 87
    assert plan == ready['training_plan'] == naive_ready['training_plan']
    assert all(sum(t['domain'] == d for t, r in gold.values()) == 20 for d in DOMAINS)
    original_main = function_ast(port / 'run.py', 'main')
    loop = next(n for n in original_main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'stage')
    cutoff = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'wanted' for t in n.targets))
    prefix = ast.Module(body=loop.body[:cutoff - 1], type_ignores=[])
    assert hashlib.sha256(ast.dump(prefix, include_attributes=False).encode()).hexdigest() == ready['training_ast_sha256'] == naive_ready['training_ast_sha256']
    loss_ast = hashlib.sha256(ast.dump(function_ast(port / 'runtime_utils.py', 'answer_loss'), include_attributes=False).encode()).hexdigest()
    stage_states, steps, reservoir_events, order, replay_order = simulated_replay(manifest, fit)
    assert len(naive_inputs) == 1213 and [x['id'] for x in naive_inputs] == order
    prior_fit = keyed(naive_inputs); naive_predictions = keyed(read(RUN / 'prior_naive_final_predictions.json'))
    assert set(naive_predictions) == set(gold)
    ranks = rank_plan()
    assert naive_initial['rank_plan'] == ranks and naive_initial['coordinates'] == COORDINATES

    # Raw contracts use existing metadata/image headers only, not a processor/model.
    spec = importlib.util.spec_from_file_location('arc_raw_retrieval', port / 'retrieval.py')
    retrieval = importlib.util.module_from_spec(spec); spec.loader.exec_module(retrieval)
    from PIL import Image
    image_cache, raw_cache = {}, {}
    def raw(task, row):
        if row['id'] in raw_cache:
            return raw_cache[row['id']]
        context, indexes = row.get('context', ''), []
        if len(context) > 6000:
            chunks = [context[i:i + 1000] for i in range(0, len(context), 800)]
            indexes = sorted(retrieval.retrieve(row['question'], chunks, 5)); context = '\n'.join(chunks[i] for i in indexes)
        shapes, images = [], []
        for name in row['images']:
            if name not in image_cache:
                assert sha(ASSET / name) == split['image_sha256'][name]
                with Image.open(ASSET / name) as im:
                    width, height = im.size
                before = [width, height]
                if width * height > 401408:
                    ratio = math.sqrt(401408 / (width * height)); width, height = max(1, int(width * ratio)), max(1, int(height * ratio))
                image_cache[name] = dict(original=before, common_input=[width, height])
            shapes.append(image_cache[name]); images.append(dict(path=name, sha256=split['image_sha256'][name]))
        contract = dict(images=images, instruction=task['instruction'], question=row['question'], original_context=row.get('context', ''),
                        resolved_context=context, retrieved_chunk_indexes=indexes, image_shapes=shapes, common_image_area_cap=401408)
        value = hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        raw_cache[row['id']] = (value, contract)
        return value, contract
    def bound(x, task, row):
        h, contract = raw(task, row)
        assert x['id'] == row['id'] and x['task'] == task['task'] and x['domain'] == task['domain']
        assert x['raw_input_sha256'] == h and x['image_SHA256'] == contract['images'] and x['image_shapes'] == contract['image_shapes']
        assert x['retrieved_chunk_indexes'] == contract['retrieved_chunk_indexes'] and x['common_image_area_cap'] == 401408
        assert not x['visual_cache_hit']; digest(x['transport_sha256'])

    masks = {}; gate_summary = {}; new_predictions = {}; memories = {}; checkpoints = {}
    for arm in ARMS:
        tr, ev = ends['train_' + arm], ends['eval_' + arm]
        assert tr['arm'] == ev['arm'] == arm
        for n, v in dict(new_generations=0, FIT_backwards=1213, current_QA=1213, replay_QA=886,
                         working_replay_forwards=886, slow_replay_forwards=886, max_target_head_calls=886,
                         optimizer_updates=87, coordinates=COORDINATES, frozen_EMA_coordinates=COORDINATES,
                         learned_gate_parameters=0, EMA_updates=87, EMA_stage_syncs=19).items():
            assert tr[n] == v
        assert tr['training_plan'] == plan and tr['training_ast_sha256'] == ready['training_ast_sha256']
        assert tr['naive_initialization_exact'] and tr['naive_current_inputs_exact'] and tr['adapter_before'] != tr['adapter_after']
        assert tr['frozen_unchanged'] and tr['ordinary_buffers_unchanged'] and tr['final_default_only'] and tr['final_EMA_removed']
        assert tr['reference_masks'] == (arm == 'shuffle'); digest(tr['final_EMA_digest'])
        needed = {arm + n for n in ('_ADAPTER_INITIAL.json', '_FIT_INPUTS.json', '_REPLAY_INPUTS.json', '_REPLAY_LEDGER.json',
                                    '_TRAIN_MEMORY.json', '_initial.pt', '_complete.pt')}
        needed.add(arm + '_training/logs/events.jsonl')
        for p in plan:
            s = p['stage']; needed.update({f'{arm}_STAGE_{s:02}_COMPLETE.json', f'{arm}_adapter_S{s:02}.pt', f'{arm}_training/validity/stage_{s:02}.json'})
            for i in range(1, p['updates'] + 1):
                needed.update({f'{arm}_grad_S{s:02}_{i:03}.json', f'{arm}_MASK_S{s:02}_{i:03}.json'})
        assert needed <= set(tr['files'])
        assert len(list(RUN.glob(arm + '_grad_S*.json'))) == len(list(RUN.glob(arm + '_MASK_S*.json'))) == 87
        assert len(list(RUN.glob(arm + '_STAGE_*_COMPLETE.json'))) == len(list(RUN.glob(arm + '_adapter_S*.pt'))) == 19
        for f in ('original_base', 'initial_checkpoint', 'initial_reload', 'initial_invariant', 'checkpoint'):
            receipt(tr[f], 2)
        same_receipt(tr['original_base'], naive_train['original_base'], device=False)
        assert tr['original_base']['digest'] == calibration['initial_state_digest']
        assert tr['original_base']['all_buffers_digest'] == calibration['all_buffers_digest'] and tr['original_base']['layout'] == calibration['layout']
        same_receipt(tr['original_base'], tr['initial_checkpoint']); same_receipt(tr['initial_checkpoint'], tr['initial_reload'])
        same_receipt(tr['initial_reload'], tr['initial_invariant']); assert tr['initial_invariant']['complete_state_and_buffers_unchanged']
        for suffix, f in (('initial', 'initial_checkpoint'), ('complete', 'checkpoint')):
            cp, name = tr[f], arm + '_' + suffix + '.pt'
            assert cp['sha256'] == tr['files'][name] and (RUN / name).stat().st_size == cp['bytes']
            assert Path(cp['path']).resolve() == (RUN / name).resolve()
            assert cp['complete_graph'] and not cp['external_base_weights_required'] and cp['no_adapter_optimizer_or_teacher'] and cp['alias_preserving_cpu_snapshot']
        for flag in ('self_contained_meta_construction', 'all_buffers_explicitly_restored', 'strict_complete_reload', 'mixed_parameter_dtypes_preserved'):
            assert tr['initial_reload'][flag] and ev['reloaded'][flag]
        fold = tr['fold']; assert len(fold) == 48 and {x['module']: x['rank'] for x in fold} == ranks
        assert all(x['scale'] == 4 and x['dtype'] == 'torch.float16' and x['merge_dtype'] == 'torch.float32'
                   and x['device'] == 'cuda:2' and not x['autocast_enabled'] and not x['matmul_allow_tf32'] for x in fold)
        init = read(RUN / (arm + '_ADAPTER_INITIAL.json'))
        assert init['coordinates'] == init['slow_coordinates'] == COORDINATES and init['parameters'] == PARAMETERS
        assert init['slow_bytes'] == GRADIENT_BYTES and init['learned_gate_parameters'] == 0 and init['initialization_exact_to_naive']
        assert init['rank_plan'] == ranks and init['initialization'] == naive_initial['initialization']; digest(init['slow_initial_digest'])
        expected_slow = {'language_model.' + n + '.' + suffix for n in ranks for suffix in ('_ilora_slow_A', '_ilora_slow_B')}
        assert len(init['slow_buffers']) == 96 and set(init['slow_buffers']) == expected_slow
        fit_inputs = read(RUN / (arm + '_FIT_INPUTS.json')); assert fit_inputs == naive_inputs
        for x in fit_inputs:
            bound(x, *fit_by_id[x['id']]); assert x['phase'] == 'TRAIN'
        replay_inputs = read(RUN / (arm + '_REPLAY_INPUTS.json'))
        assert len(replay_inputs) == 886 and [key(fit_by_id[x['id']][0], x) for x in replay_inputs] == replay_order
        for x in replay_inputs:
            assert x == dict(prior_fit[x['id']], phase='REPLAY')
            bound(x, *fit_by_id[x['id']])
        ledger = read(RUN / (arm + '_REPLAY_LEDGER.json'))
        for n, v in dict(past_only=True, capacity=500, with_replacement=True, task_end_insert=True, current_QA=1213,
                         replay_QA=886, updates=87, extra_full_decoder_for_gate=0, working_replay_forwards=886,
                         slow_replay_forwards=886, max_target_head_calls=886, reference_mask_counterfactual=arm == 'shuffle').items():
            assert ledger[n] == v
        assert ledger['reservoir_events'] == reservoir_events
        assert ledger['stage_starts'] == [s[0] for s in stage_states] and ledger['stage_ends'] == [s[1] for s in stage_states]
        assert len(ledger['events']) == 87
        masks[arm] = []; accepted = natural = nonzero = 0; zero_steps = all_steps = 0
        for expected, event in zip(steps, ledger['events']):
            s, i = expected['stage'], expected['step']; name = f'{arm}_MASK_S{s:02}_{i:03}.json'
            assert event['file'] == name and event['sha256'] == sha(RUN / name) == tr['files'][name]
            m = read(RUN / name)
            for n, v in expected.items():
                assert m[n] == v, n
            assert m['EMA_updates'] == expected['global_update'] and m['replay_count'] == len(expected['replay_QA'])
            assert m['denominator'] == 'all_replay_QA; unchanged current minibatch accumulation'
            stats = m['stats']; assert len(stats) == len(m['natural_gate']) == len(m['applied_mask']) == len(expected['replay_QA'])
            for x in stats:
                for n in ('working_NLL', 'target_NLL', 'raw_MSE', 'masked_MSE'):
                    assert finite(x[n]) and x[n] >= 0, n
                assert x['natural_gate'] == int(x['target_NLL'] < x['working_NLL'])
                assert type(x['natural_gate']) is int and x['applied_mask'] in (0, 1) and not isinstance(x['applied_mask'], bool)
                assert x['masked_MSE'] == x['raw_MSE'] * x['applied_mask']
                assert x['replay'] and x['arm'] == arm and 2 <= x['answer_tokens'] <= 513
                assert x['prompt_tokens'] > 0 and x['packed_tokens'] == x['prompt_tokens'] + x['answer_tokens']
                assert x['hidden_states_count'] == 25 and x['hidden_width'] == 896
                assert x['all_packed_positions_effective'] and x['target_detached'] and x['teacher_same_training_mode']
                assert x['slow_decoder_forwards'] == x['working_decoder_forwards'] == x['extra_target_head_calls'] == 1
                assert x['slow_answer_head_calls'] == 0 and x['source_answer_loss_AST_sha256'] == loss_ast
            assert m['natural_gate'] == [x['natural_gate'] for x in stats] and m['applied_mask'] == [int(x['applied_mask']) for x in stats]
            if arm == 'original':
                assert m['applied_mask'] == [1] * len(stats) and m['reference_mask_sha256'] is None
            elif arm == 'gate':
                assert m['applied_mask'] == m['natural_gate'] and m['reference_mask_sha256'] is None
            else:
                reference_name = f'gate_MASK_S{s:02}_{i:03}.json'; reference = masks['gate'][len(masks[arm])]
                assert m['reference_mask_sha256'] == sha(RUN / reference_name) == ends['train_gate']['files'][reference_name]
                assert m['current_QA'] == reference['current_QA'] and m['replay_QA'] == reference['replay_QA']
                shuffled = list(reference['natural_gate'])
                seed = int(hashlib.sha256(f'42:reference-mask:{s}:{i}'.encode()).hexdigest(), 16)
                random.Random(seed).shuffle(shuffled)
                assert m['applied_mask'] == shuffled and sum(m['applied_mask']) == sum(reference['natural_gate'])
            assert m['accepted'] == sum(m['applied_mask'])
            assert event == dict(file=name, sha256=sha(RUN / name), stage=s, step=i, current_QA=len(m['current_QA']), replay_QA=len(stats), accepted=m['accepted'])
            if stats:
                zero_steps += int(m['accepted'] == 0); all_steps += int(m['accepted'] == len(stats))
            accepted += m['accepted']; natural += sum(m['natural_gate']); nonzero += sum(x['masked_MSE'] > 0 for x in stats)
            masks[arm].append(m)
        gate_summary[arm] = dict(replay_QA=886, natural_acceptances=natural, applied_acceptances=accepted,
                                 nonzero_applied_MSE_QA=nonzero, replay_steps_all_off=zero_steps, replay_steps_all_on=all_steps,
                                 all_off=accepted == 0, all_on=accepted == 886)
        cumulative = 0
        for p in plan:
            s, updates = p['stage'], p['updates']; cumulative += updates
            st = read(RUN / f'{arm}_STAGE_{s:02}_COMPLETE.json'); valid = read(RUN / f'{arm}_training/validity/stage_{s:02}.json')
            assert st['stage'] == s and st['updates'] == updates and st['bytes'] == st['EMA_bytes'] == GRADIENT_BYTES
            assert st['optimizer_state_bytes'] == OPTIMIZER_BYTES and st['all_optimizer_state_GPU']
            assert st['frozen_unchanged'] and st['ordinary_buffers_unchanged'] and st['EMA_coordinates'] == COORDINATES
            assert st['EMA_updates'] == cumulative and st['EMA_stage_syncs'] == s and st['memory'] == stage_states[s - 1][1]
            digest(st['EMA_digest']); digest(st['adapter_digest'])
            if s == 19:
                assert st['EMA_digest'] == tr['final_EMA_digest']
            assert valid['passed'] and valid['actual_parameters_changed'] and valid['updates'] == valid['positive_lr_updates'] == updates
            assert valid['parameters_before_sha256'] != valid['parameters_after_sha256']
            for i in range(1, updates + 1):
                assert read(RUN / f'{arm}_grad_S{s:02}_{i:03}.json') == dict(parameters=PARAMETERS, finite=True, gradient_bytes=GRADIENT_BYTES)
        events = [json.loads(line) for line in (RUN / (arm + '_training/logs/events.jsonl')).read_bytes().splitlines()]
        assert len(events) == 87
        for p in plan:
            selected = [e for e in events if e['stage'] == p['stage']]; assert len(selected) == p['updates']
            warmup = min(math.floor(.03 * p['updates']), p['updates'] - 1)
            for k, e in enumerate(selected):
                assert e['event'] == 'train_update' and e['step'] == k + 1 and e['qa_seen'] == min((k + 1) * 16, p['QA']) and e['total_qa'] == p['QA']
                assert e['warmup_updates'] == warmup and finite(e['loss'])
                lr = 1e-4 * (.5 * (1 + math.cos(math.pi * (k - warmup) / max(1, p['updates'] - warmup))) if k >= warmup else k / max(1, warmup))
                assert len(e['learning_rate_used']) == 1 and e['learning_rate_used'][0] > 0 and math.isclose(e['learning_rate_used'][0], lr, rel_tol=1e-12, abs_tol=1e-15)
                assert finite(e['gradient_l2_preclip']) and e['gradient_l2_preclip'] >= 0
                assert finite(e['gradient_l2_postclip']) and 0 <= e['gradient_l2_postclip'] <= 1.00001
                assert e['visual_cache_cpu_bytes'] == e['visual_cache_cpu_limit'] == 0
        assert ev['new_generations'] == 105 and ev['FIT_backwards'] == ev['optimizer_updates'] == 0 and ev['final_default_only']
        assert set(ev['files']) == {arm + '_memory.json', arm + '_final_predictions.json'}
        receipt(ev['reloaded'], 1); receipt(ev['invariant'], 1)
        same_receipt(tr['checkpoint'], ev['reloaded'], device=False); same_receipt(ev['reloaded'], ev['invariant'])
        assert ev['invariant']['complete_state_and_buffers_unchanged']
        predictions = keyed(read(RUN / (arm + '_final_predictions.json'))); assert set(predictions) == set(gold)
        for rid, x in predictions.items():
            task, row = gold[rid]; bound(x, task, row)
            assert x['phase'] == 'DEV_FINAL' and x['references'] == row['answers'] and x['generation_cap'] == task['max_new_tokens']
            assert isinstance(x['prediction'], str) and isinstance(x['output_tokens'], list) and len(x['output_tokens']) <= x['generation_cap']
            assert all(type(t) is int and t >= 0 for t in x['output_tokens'])
            assert x['input_shape'] == [1, x['prompt_tokens']] == x['input_tensor_shapes']['input_ids'] == x['input_tensor_shapes']['attention_mask']
            digest(x['prepared_sha256'])
            for n in ('task', 'domain', 'references', 'generation_cap', 'raw_input_sha256', 'transport_sha256', 'prepared_sha256', 'image_SHA256', 'image_shapes', 'input_shape', 'input_tensor_shapes'):
                assert x[n] == naive_predictions[rid][n], n
        memory = read(RUN / (arm + '_memory.json')); assert len(memory) == 5
        for i, (domain, x) in enumerate(zip(DOMAINS, memory)):
            rid = min((rid for rid, (t, r) in gold.items() if t['domain'] == domain), key=lambda rid: hashlib.sha256(rid.encode()).digest())
            assert x['id'] == rid and x['domain'] == domain and x['new_tokens'] == 32
            assert all(type(x[n]) is int and x[n] > 0 for n in ('baseline_allocated', 'peak_allocated', 'peak_reserved'))
            assert x['baseline_allocated'] <= x['peak_allocated'] <= x['peak_reserved']
            old = naive['memory']['plain_r8'][i]
            for n in ('id', 'domain', 'raw_input_sha256', 'transport_sha256', 'prepared_sha256', 'input_shape', 'input_tensor_shapes', 'new_tokens'):
                assert x[n] == old[n], n
            for n in ('raw_input_sha256', 'transport_sha256', 'prepared_sha256', 'input_shape', 'input_tensor_shapes'):
                assert x[n] == predictions[rid][n], n
        new_predictions[arm], memories[arm], checkpoints[arm] = predictions, memory, tr['checkpoint']
    assert gate_summary['gate']['applied_acceptances'] == gate_summary['shuffle']['applied_acceptances']
    reference_differences = [sum(a != b for a, b in zip(reference['applied_mask'], shuffled['applied_mask']))
                             for reference, shuffled in zip(masks['gate'], masks['shuffle'])]
    gate_summary['shuffle'].update(
        reference_mask_Hamming_QA=sum(reference_differences),
        reference_mask_Hamming_steps=sum(n > 0 for n in reference_differences),
        reference_mixed_mask_steps=sum(0 < m['accepted'] < len(m['replay_QA']) for m in masks['gate']),
        replay_steps_size1=sum(len(m['replay_QA']) == 1 for m in masks['shuffle']),
        replay_steps_with_data=sum(bool(m['replay_QA']) for m in masks['shuffle']),
        reference_assignment_changed=any(reference_differences))

    nodes = [n for n in ast.parse((port / 'runtime_utils.py').read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name in ('tokens', 'score')]
    assert {n.name for n in nodes} == {'tokens', 'score'}
    tree = ast.Module(body=nodes, type_ignores=[]); scorer_hash = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    assert scorer_hash == naive['scorer_ast_sha256']
    scope = dict(re=re, Counter=collections.Counter); exec(compile(tree, '<original Answer F1>', 'exec'), scope)
    naive_quality = naive['quality']['plain_r8_final']; score_order = [x['id'] for x in naive_quality['per_id']]
    assert len(score_order) == 100 and set(score_order) == set(gold)
    write('VERIFY_STARTED.json', dict(unix=time.time(), ready_sha256=rh, new_scores=300,
                                     scorer_ast_sha256=scorer_hash, all_six_workers_exited=True, historical_scores_recomputed=0))
    quality = {}
    for arm in ARMS:
        per_id = []
        for rid in score_order:
            x = new_predictions[arm][rid]; value = scope['score'](x['prediction'], x['references'])['token_f1']
            assert finite(value) and 0 <= value <= 100
            per_id.append(dict(id=rid, domain=x['domain'], F1=value))
        domains = {d: dict(F1=sum(x['F1'] for x in per_id if x['domain'] == d) / 20, QA=20) for d in DOMAINS}
        quality[arm] = dict(domains=domains, macro=sum(x['F1'] for x in domains.values()) / 5, per_id=per_id)
    quality['naive'] = naive_quality  # Exact existing scalar copy; no scorer call.
    comparisons = {}
    for left, right in (('gate', 'original'), ('gate', 'shuffle'), ('gate', 'naive'), ('original', 'naive'), ('shuffle', 'naive')):
        a, b = keyed(quality[left]['per_id']), keyed(quality[right]['per_id'])
        comparisons[left + '_minus_' + right] = dict(macro=quality[left]['macro'] - quality[right]['macro'],
            domains={d: quality[left]['domains'][d]['F1'] - quality[right]['domains'][d]['F1'] for d in DOMAINS},
            per_id=[dict(id=rid, domain=a[rid]['domain'], difference=a[rid]['F1'] - b[rid]['F1']) for rid in score_order],
            up=sum(a[rid]['F1'] > b[rid]['F1'] for rid in score_order), down=sum(a[rid]['F1'] < b[rid]['F1'] for rid in score_order),
            tie=sum(a[rid]['F1'] == b[rid]['F1'] for rid in score_order))
    above_original = quality['gate']['macro'] > quality['original']['macro']
    above_shuffle = quality['gate']['macro'] > quality['shuffle']['macro']
    above_naive = quality['gate']['macro'] > quality['naive']['macro']
    nondegenerate = not gate_summary['gate']['all_off'] and not gate_summary['gate']['all_on'] and gate_summary['gate']['nonzero_applied_MSE_QA'] > 0
    assignment_changed = gate_summary['shuffle']['reference_assignment_changed']
    decision = ('count_control_reference_assignment_unchanged_selection_not_identified' if not assignment_changed
                else 'nondegenerate_proxy_selection_over_original_and_count_control_observed' if nondegenerate and above_original and above_shuffle
                else 'fixed_proxy_selection_not_supported_over_both_controls')
    memory_comparison = {arm: [dict(domain=d, new_peak_allocated=x['peak_allocated'], naive_peak_allocated=old['peak_allocated'],
                                   difference=x['peak_allocated'] - old['peak_allocated'])
                              for d, x, old in zip(DOMAINS, memories[arm], naive['memory']['plain_r8'])] for arm in ARMS}
    result = dict(complete=True, ready_sha256=rh, all_exit_sha256=sha(RUN / 'ALL_EXIT.json'), source_COMPLETE_sha256=all_exit['source_COMPLETE_sha256'],
        prior_naive_READY_sha256=sha(RUN / 'prior_naive_READY.json'), prior_naive_VERIFIED_sha256=sha(RUN / 'prior_naive_VERIFIED.json'),
        scorer_ast_sha256=scorer_hash, quality=quality, comparisons=comparisons, gate_statistics=gate_summary,
        gate_over_original=above_original, gate_over_count_matched_shuffle=above_shuffle, gate_over_naive=above_naive,
        nondegenerate_reference_gate=nondegenerate, count_control_reference_assignment_changed=assignment_changed,
        decision=decision, per_domain_non_decrease_gate_used=False,
        complete_checkpoints=checkpoints, memory=memories, naive_memory_copied=naive['memory']['plain_r8'], memory_comparison=memory_comparison,
        training_coordinates=COORDINATES, trainable_parameter_tensors=PARAMETERS, frozen_slow_coordinates=COORDINATES,
        frozen_slow_bytes=GRADIENT_BYTES, learned_gate_parameters=0, gradient_bytes=GRADIENT_BYTES, GPU_optimizer_state_bytes=OPTIMIZER_BYTES,
        FIT_backwards=3639, current_QA=3639, replay_QA=2658, working_replay_forwards=2658, slow_replay_forwards=2658,
        actual_training_decoder_requests=8955, extra_target_head_calls=2658, extra_full_decoder_for_gate=0,
        optimizer_updates=261, new_generations=315, new_scores=300, TEST_QA=0, initial_DEV_generations=0,
        historical_scores_recomputed=0, naive_training_repeated=False, final_default_only=True, final_no_EMA_or_memory_or_gate=True,
        final_weights=WEIGHTS, final_ordinary_buffer_bytes=9032, new_inference_coefficients_relative_to_naive=0,
        phase_seconds={p: x['seconds'] for p, x in ends.items()}, model_phase_seconds=sum(x['seconds'] for x in ends.values()),
        controller_seconds=all_exit['controller_seconds'], prior_naive_model_seconds_separate=naive['model_phase_seconds'],
        terminal_only_no_acquisition_forgetting_identification=True, formal_or_strongest_baseline_superiority_proven=False,
        user_joint_goal_complete_proven=False, same_dtype_optimal_deployment_superiority_proven=False,
        interpretation='Fixed-r8 transplanted I-LoRA controls on explored100DEV, one seed. Last-layer actual-max-target NLL is a stochastic proxy for the whole-hidden constraint, not a guarantee of useful gradients or visual grounding. Gate versus original tests its package; same-count reference-shuffle separates selection from accepting fewer QAs, not all loss/gradient magnitude confounds. Naive comparisons also include replay/EMA and different training GPU. No formal, significance, forgetting-causality, new structural compression or user-goal-complete claim.',
        verification_seconds=time.monotonic() - tick)
    write('VERIFIED.json', result)
    print(json.dumps(dict(VERIFIED_sha256=sha(RUN / 'VERIFIED.json'), macros={a: q['macro'] for a, q in quality.items()},
                          decision=decision, new_scores=300, model_phase_seconds=result['model_phase_seconds'])))


if __name__ == '__main__':
    main()
