"""Verify all acquired-stage readouts, then score exactly80 new answers once.

Standard-library metadata only: no Torch, pickle/checkpoint deserialization,
model execution, historical rescoring, bootstrap or outcome-dependent gate.
"""
from pathlib import Path
import ast
import collections
import hashlib
import json
import math
import os
import re
import time

ROOT = Path(os.environ.get('LLMCL_ROOT', Path(__file__).resolve().parents[3]))
RUN = Path(os.environ.get('LLMCL_RUN', Path(__file__).resolve().parent))
ASSET = Path(os.environ.get('LLMCL_ASSET', ROOT))
METHODS = ('original', 'naive')
DOMAINS = ('medical', 'geography')
WEIGHTS = dict(coefficients=893675552, bytes=1787351104)
COORDINATES, PARAMETERS, ADAPTER_BYTES = 540672, 96, 2162688
RECEIPT_KEYS = ('layout', 'digest', 'all_buffers_digest', 'buffer_manifest',
    'parameter_alias_groups', 'buffer_alias_groups', 'functional_config_digest',
    'generation_config_digest', 'language_generation_config_digest',
    'language_attention_classes', 'vision_attention_classes', 'response')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    with (RUN / name).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(value):
    assert isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def keyed(rows):
    result = {x['id']: x for x in rows}
    assert len(result) == len(rows)
    return result


def same_receipt(left, right):
    for key in RECEIPT_KEYS:
        assert left[key] == right[key], key


def receipt(value, gpu):
    assert value['layout']['weights'] == WEIGHTS and value['layout']['trainable'] == 0
    assert value['layout']['ordinary_buffers']['bytes'] == 9032
    assert value['device'] == f'cuda:{gpu}' and value['response'] is None
    for key in ('digest', 'all_buffers_digest', 'functional_config_digest',
                'generation_config_digest', 'language_generation_config_digest'):
        digest(value[key])
    assert not any(n.endswith(('._ilora_slow_A', '._ilora_slow_B', '.mlp.response_sign',
                              '.mlp.calibration_scale')) for n in value['buffer_manifest'])
    assert all(x['dtype'] == 'torch.float32' for n, x in value['buffer_manifest'].items()
               if n.endswith('inv_freq'))


def function_ast(path, name):
    return next(n for n in ast.parse(Path(path).read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def no_process(who):
    assert who['pid'] == who['pgid'] and who['uid'] == os.getuid()
    p = Path('/proc') / str(who['pid']) / 'stat'
    assert not p.exists() or p.read_text().rsplit(') ', 1)[1].split()[19] != str(who['startticks'])


def arc_chain(directory, verified):
    """Bind both disclosed, model-free implementation repairs, not repeat them."""
    vf = directory / 'VERIFIED.json'
    cv = read(directory / 'CONTINUATION_VERIFIED.json')
    schema = read(directory / 'POST_MODEL_SCHEMA_VERIFIED.json')
    permit = read(directory / 'PRESTART_CONTINUATION_PERMIT.json')
    schema_permit = read(directory / 'POST_MODEL_SCHEMA_PERMIT.json')
    schema_start = read(directory / 'POST_MODEL_SCHEMA_STARTED.json')
    old_ready = read(directory / 'READY.json')
    assert verified['complete'] and verified['new_scores'] == 300
    assert cv['complete'] and cv['VERIFIED_sha256'] == sha(vf) and cv['quality_scores'] == 300
    assert cv['post_model_verifier_schema_repair'] and cv['old_continuation_unit_failed_preserved']
    assert cv['post_model_schema_VERIFIED_sha256'] == sha(directory / 'POST_MODEL_SCHEMA_VERIFIED.json')
    assert schema['complete'] and schema['VERIFIED_sha256'] == sha(vf)
    assert schema['prior_failed_verification_scores'] == 0
    assert schema['first_scoring_execution_new_scores'] == 300 and schema['scoring_executions'] == 1
    assert schema['new_models'] == schema['new_generations'] == 0
    assert schema['old_eight_scientific_files_unchanged'] and schema['all_six_model_phases_preserved']
    assert schema['original_verifier_sha256'] == old_ready['sources']['verify.py']
    correction = schema['correction']
    assert correction['changed_assert_line'] == 403 and correction['all_other_AST_nodes_unchanged']
    assert correction['old_assert'] == "assert x['phase'] == 'DEV_FINAL' and x['references'] == row['answers'] and x['generation_cap'] == task['max_new_tokens']"
    assert correction['new_assert'] == "assert 'phase' not in x and x['references'] == row['answers'] and x['generation_cap'] == task['max_new_tokens']"
    for n in ('original_AST_sha256', 'corrected_AST_sha256'):
        digest(correction[n])
    scientific_sources = {'adaptation.py', 'common.py', 'controller.py', 'deployment.py',
                          'loss_core.py', 'readouts.py', 'verify.py', 'worker.py'}
    assert len(old_ready['sources']) == 14
    assert {n for n in old_ready['sources'] if n.endswith('.py')} == scientific_sources
    for n, h in old_ready['sources'].items():
        assert sha(directory / n) == h, n
    assert schema['permit_sha256'] == sha(directory / 'POST_MODEL_SCHEMA_PERMIT.json')
    assert schema['STARTED_sha256'] == sha(directory / 'POST_MODEL_SCHEMA_STARTED.json')
    assert schema_start['first_attempt_score_marker_absent'] and schema_start['previous_scores'] == 0
    assert schema_start['correction'] == correction
    assert schema['original_failed_files'] == schema_permit['failed_files']
    assert set(schema['original_failed_files']) == {'CONTINUATION_FAILURE.json', 'VERIFIER_EXIT.json'}
    assert schema['original_failure_archive'] == 'post_model_verifier_schema_failure_20261025'
    for n, h in schema['original_failed_files'].items():
        assert sha(directory / schema['original_failure_archive'] / n) == h
    failed = read(directory / schema['original_failure_archive'] / 'VERIFIER_EXIT.json')
    assert failed['returncode'] == 1 and "KeyError: 'phase'" in failed['stderr']
    assert len(schema['old_units_failed_preserved']) == 2
    assert all(v == dict(ActiveState='failed', MainPID='0', ExecMainStatus='1')
               for v in schema['old_units_failed_preserved'].values())
    assert cv['permit_sha256'] == sha(directory / 'PRESTART_CONTINUATION_PERMIT.json')
    assert cv['continuation_model_complete_sha256'] == sha(directory / 'CONTINUATION_MODEL_COMPLETE.json')
    assert permit['ready_sha256'] == verified['ready_sha256'] == sha(directory / 'READY.json')
    assert cv['preSTARTED_model_attempts'] == cv['new_retries_of_started_model_phases'] == 0
    assert cv['actual_model_phases'] == 6 and cv['original_unit_failed_preserved']
    assert cv['old_scientific_sources_unchanged'] and cv['READY_unchanged']
    assert cv['absolute_original_deadline_preserved'] and schema['absolute_original_deadline_preserved']
    assert cv['total_elapsed_since_original_activation'] <= 18300
    for n, h in permit['attempt_files'].items():
        assert sha(directory / 'preSTARTED_guard_attempt_20261025' / n) == h
        assert cv['original_preSTARTED_attempt_archive'][n] == h
    for n, h in permit['preserved_gate_files'].items():
        assert sha(directory / n) == h
    assert verified['all_exit_sha256'] == schema['all_exit_sha256'] == sha(directory / 'ALL_EXIT.json')
    model_end = read(directory / 'CONTINUATION_MODEL_COMPLETE.json')
    assert model_end['all_exit_sha256'] == verified['all_exit_sha256']
    assert model_end['original_gate_preserved'] and model_end['source_and_READY_unchanged']
    return dict(VERIFIED_sha256=sha(vf), CONTINUATION_VERIFIED_sha256=sha(directory / 'CONTINUATION_VERIFIED.json'),
        POST_MODEL_SCHEMA_VERIFIED_sha256=sha(directory / 'POST_MODEL_SCHEMA_VERIFIED.json'),
        both_disclosed_repairs_preserved=True, original_scientific_sources_unchanged=True,
        previous_failed_verifier_scores=0, successful_parent_scores=300)


def main():
    tick = time.monotonic()
    assert not (RUN / 'VERIFY_STARTED.json').exists() and not (RUN / 'VERIFIED.json').exists()
    assert not list(RUN.glob('*_FAILURE.json'))
    ready = read(RUN / 'READY.json'); rh = sha(RUN / 'READY.json')
    for field in ('sources', 'prior_files'):
        for n, h in ready[field].items():
            assert sha(RUN / n) == h, n
    for n, h in ready['environment'].items():
        assert sha(n) == h, n
    for n, h in ready['runtime_files'].items():
        if not n.endswith('.safetensors'):
            assert sha(ASSET / n) == h, n
    assert sha(ROOT / 'exp/ffn_static_delete_20261025/v1/core.py') == ready['storage_helper_sha256']
    assert sha(RUN / 'PROTOCOL.md') == ready['protocol_sha256']
    phase = 'acquisition_readout'
    start, end, ex = (read(RUN / (phase + suffix)) for suffix in ('_STARTED.json', '_COMPLETE.json', '_EXIT.json'))
    assert start['ready_sha256'] == end['ready_sha256'] == rh and start['identity'] == end['identity']
    assert start['GPU'] == 1 and start['limit'] == 1800
    assert start['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert ex['pid'] == start['identity']['pid'] and ex['returncode'] == 0
    assert finite(end['seconds']) and 0 < end['seconds'] <= 1800
    # COMPLETE bounds model work at1800s; process exit additionally stays inside
    # the protocol's2100s outer supervision, without an invented teardown gate.
    assert start['unix'] <= ex['unix'] and ex['unix'] - start['unix'] <= 2100
    no_process(start['identity'])
    for n, value in dict(new_generations=80, new_scores=0, NLL_requests=0, FIT_backwards=0,
                        optimizer_updates=0, TEST_QA=0, S19_new_generations=0,
                        initial_new_generations=0, final_new_generations=0).items():
        assert end[n] == value, n
    assert end['methods'] == list(METHODS) and end['domains'] == list(DOMAINS)
    assert end['both_S19_controls_before_any_new_generation'] and end['frozen_parent_runs_and_sources_unchanged']
    assert not list(RUN.glob('*.pt')), 'This diagnostic must not write new full checkpoints or adapters'
    for n, h in end['files'].items():
        assert sha(RUN / n) == h, n

    arc = Path(ready['arc_source_dir']); assert arc.is_absolute()
    arc_ready = read(arc / 'READY.json')
    for name in ('common.py', 'deployment.py', 'adaptation.py', 'readouts.py'):
        path = arc / name
        assert ready['sources'][str(path)] == arc_ready['sources'][name] == sha(path)
    assert set(ready['acquisition_methods']) == set(METHODS)
    parents = {}
    for name in METHODS:
        source = ready['acquisition_methods'][name]
        directory = Path(source['run_dir']); train_path = Path(source['train_complete']); vf_path = Path(source['verified'])
        assert all(p.is_absolute() for p in (directory, train_path, vf_path))
        assert sha(train_path) == ready['prior_files'][str(train_path)]
        assert sha(vf_path) == ready['prior_files'][str(vf_path)]
        old_ready = read(directory / 'READY.json'); train = read(train_path); vf = read(vf_path)
        prefix = 'original' if name == 'original' else 'qv'
        assert vf['complete'] and train['ready_sha256'] == vf['ready_sha256'] == sha(directory / 'READY.json')
        assert vf['source_COMPLETE_sha256']['train_' + prefix] == sha(train_path)
        assert train['FIT_backwards'] == 1213 and train['optimizer_updates'] == 87
        assert train['coordinates'] == COORDINATES and train['frozen_unchanged']
        for n in ('original_base', 'initial_checkpoint', 'initial_reload', 'initial_invariant'):
            same_receipt(train[n], train['original_base'])
        for suffix, key in (('initial', 'initial_checkpoint'), ('complete', 'checkpoint')):
            container = directory / (prefix + '_' + suffix + '.pt'); cp = train[key]
            assert Path(cp['path']).resolve() == container.resolve()
            assert cp['sha256'] == train['files'][container.name] and container.stat().st_size == cp['bytes']
            assert cp['complete_graph'] and cp['no_adapter_optimizer_or_teacher']
        parents[name] = dict(source=source, directory=directory, train=train, verified=vf, ready=old_ready, prefix=prefix)
    provenance = arc_chain(arc, parents['original']['verified'])
    naive = parents['naive']; original = parents['original']
    assert original['directory'] == arc
    assert original['verified']['prior_naive_VERIFIED_sha256'] == sha(Path(naive['source']['verified']))
    assert original['verified']['prior_naive_READY_sha256'] == sha(naive['directory'] / 'READY.json')
    assert naive['verified']['new_scores'] == 100 and naive['verified']['new_generations'] == 105
    old_predictions_path = arc / 'prior_naive_final_predictions.json'
    assert sha(old_predictions_path) == arc_ready['prior_files']['prior_naive_final_predictions.json']
    old_eval_path = arc / 'prior_naive_eval_COMPLETE.json'
    assert sha(old_eval_path) == arc_ready['prior_files']['prior_naive_eval_COMPLETE.json']
    old_eval = read(old_eval_path)
    assert sha(old_eval_path) == naive['verified']['source_COMPLETE_sha256']['eval_qv']
    assert old_eval['files']['qv_final_predictions.json'] == sha(old_predictions_path)
    old_predictions = keyed(read(old_predictions_path))
    same_receipt(original['train']['original_base'], naive['train']['original_base'])

    port = ASSET / 'exp/two_backbone_native_cl_20261017_v2'
    policy = read(port / 'execution_policy.json')
    manifest_path = ASSET / policy['conditions']['pct01']['manifest']; manifest = read(manifest_path)
    split_path = ASSET / 'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'; split = read(split_path)
    assert sha(manifest_path) == ready['manifest_sha256']
    assert sha(split_path) == ready['split_sha256']
    for parent in parents.values():
        assert parent['ready']['manifest_sha256'] == ready['manifest_sha256']
        assert parent['ready']['split_sha256'] == ready['split_sha256']
    assert [x['task'] for x in manifest['tasks']] == list(range(1, 20))
    selected, gold = [], {}
    for task in manifest['tasks']:
        path = ASSET / task['files']['train']['path']
        assert sha(path) == task['files']['train']['sha256']
        rows = keyed([json.loads(line) for line in path.read_text(encoding='utf8').splitlines()])
        fit_ids, dev_ids = split['fit_ids'][str(task['task'])], split['dev_ids'][str(task['task'])]
        assert set(fit_ids).isdisjoint(dev_ids) and set(fit_ids) | set(dev_ids) == set(rows)
        if task['domain'] in DOMAINS:
            for rid in dev_ids:
                assert rid not in gold
                gold[rid] = (task, rows[rid])
                selected.append(dict(id=rid, task=task['task'], domain=task['domain'], generation_cap=task['max_new_tokens']))
    assert len(gold) == 40 and all(sum(t['domain'] == d for t, r in gold.values()) == 20 for d in DOMAINS)
    assert selected == ready['selected_cases']
    stages = sorted({x['task'] for x in selected})
    assert stages == [2, 3, 4, 5, 6, 10, 11, 13, 14, 15]
    assert set(gold) <= set(old_predictions)

    rank_plan = {f'model.layers.{i}.self_attn.{q}_proj': 8 for i in range(24) for q in ('q', 'v')}
    shapes = {name + '.A.weight': [8, 896] for name in rank_plan}
    shapes.update({name + '.B.weight': [896 if name.endswith('.q_proj') else 128, 8] for name in rank_plan})
    assert len(shapes) == PARAMETERS and sum(math.prod(v) for v in shapes.values()) == COORDINATES

    def reconstruction(proof, name, stage):
        parent = parents[name]; train = parent['train']; directory = parent['directory']; prefix = parent['prefix']
        assert proof['method'] == name and proof['stage'] == stage
        assert Path(proof['initial_checkpoint_path']) == directory / (prefix + '_initial.pt')
        assert proof['initial_checkpoint_declared_sha256'] == train['initial_checkpoint']['sha256']
        assert not proof['initial_container_rehashed'] and proof['initial_complete_state_checked']
        assert proof['source_train_complete_sha256'] == sha(Path(parent['source']['train_complete']))
        assert proof['source_verified_sha256'] == sha(Path(parent['source']['verified']))
        stage_path = directory / f'{prefix}_STAGE_{stage:02}_COMPLETE.json'
        snapshot = directory / f'{prefix}_adapter_S{stage:02}.pt'
        assert Path(proof['stage_proof_path']) == stage_path and Path(proof['snapshot_path']) == snapshot
        assert proof['stage_proof_sha256'] == sha(stage_path) == train['files'][stage_path.name]
        assert proof['snapshot_sha256'] == train['files'][snapshot.name]
        sealed = read(stage_path)
        assert sealed['stage'] == stage and sealed['bytes'] == ADAPTER_BYTES and sealed['frozen_unchanged']
        assert proof['stage_adapter_digest'] == proof['copied_adapter_digest'] == sealed['adapter_digest']
        assert proof['working_parameters'] == PARAMETERS and proof['coordinates'] == COORDINATES
        assert proof['adapter_bytes'] == ADAPTER_BYTES and proof['keys'] == sorted(shapes) and proof['shapes'] == shapes
        for flag in ('copy_not_add', 'same_parameter_objects', 'frozen_unchanged', 'ordinary_buffers_unchanged',
                     'no_EMA_state_restored', 'temporary_slow_buffers_removed', 'fresh_initial_each_stage'):
            assert proof[flag], flag
        assert not proof['cumulative_fold']
        receipt(proof['initial_reload'], 1); same_receipt(proof['initial_reload'], train['original_base'])
        receipt(proof['reloaded'], 1); receipt(proof['invariant'], 1)
        same_receipt(proof['reloaded'], proof['invariant'])
        assert proof['invariant']['complete_state_and_buffers_unchanged']
        fold = proof['fold']; assert len(fold) == 48 and {x['module']: x['rank'] for x in fold} == rank_plan
        assert all(x['scale'] == 4 and x['dtype'] == 'torch.float16' and x['merge_dtype'] == 'torch.float32'
                   and x['device'] == 'cuda:1' and not x['autocast_enabled'] and not x['matmul_allow_tf32'] for x in fold)

    controls = read(RUN / 'S19_CONTROLS.json'); assert set(controls) == set(METHODS)
    for name in METHODS:
        proof = controls[name]; reconstruction(proof, name, 19)
        assert proof['full_final_receipt_exact'] and proof['new_generations'] == proof['FIT_backwards'] == proof['optimizer_updates'] == 0
        same_receipt(proof['reloaded'], parents[name]['train']['checkpoint'])
    expected_files = {'S19_CONTROLS.json', 'acquired_predictions.json'} | {
        f'{name}_S{stage:02}_RECONSTRUCTION.json' for name in METHODS for stage in stages}
    assert set(end['files']) == expected_files
    events, input_records = [], {}
    for name in METHODS:
        for stage in stages:
            filename = f'{name}_S{stage:02}_RECONSTRUCTION.json'; proof = read(RUN / filename)
            reconstruction(proof, name, stage)
            ids = [x['id'] for x in selected if x['task'] == stage]
            assert proof['readout_IDs'] == ids and proof['new_generations'] == len(ids)
            assert proof['new_scores'] == proof['FIT_backwards'] == proof['optimizer_updates'] == 0
            records = proof['input_records']; assert [x['id'] for x in records] == ids
            for x in records:
                assert x['phase'] == 'ACQUIRED_DEV' and x['task'] == stage
                input_records[name, x['id']] = x
            events.append(dict(method=name, stage=stage, new_generations=len(ids), proof=filename, sha256=sha(RUN / filename)))
    assert end['stage_events'] == events and len(input_records) == 80
    predictions = read(RUN / 'acquired_predictions.json')
    expected_order = [(name, x['id']) for name in METHODS for stage in stages for x in selected if x['task'] == stage]
    assert [(x['method'], x['id']) for x in predictions] == expected_order
    assert len(predictions) == len({(x['method'], x['id']) for x in predictions}) == 80
    metadata_keys = ('task', 'domain', 'references', 'generation_cap', 'raw_input_sha256', 'transport_sha256',
                     'prepared_sha256', 'image_SHA256', 'image_shapes', 'input_shape', 'input_tensor_shapes')
    for x in predictions:
        task, row = gold[x['id']]; previous = old_predictions[x['id']]
        assert x['method'] in METHODS and x['phase'] == 'ACQUIRED_DEV'
        assert x['acquired_stage'] == x['task'] == task['task'] and x['domain'] == task['domain']
        assert x['references'] == row['answers'] and x['generation_cap'] == task['max_new_tokens']
        assert isinstance(x['prediction'], str) and isinstance(x['output_tokens'], list)
        assert len(x['output_tokens']) <= x['generation_cap'] and all(type(t) is int and t >= 0 for t in x['output_tokens'])
        assert x['input_shape'] == [1, x['prompt_tokens']] == x['input_tensor_shapes']['input_ids'] == x['input_tensor_shapes']['attention_mask']
        for k in metadata_keys:
            assert x[k] == previous[k], k
        for k in ('raw_input_sha256', 'transport_sha256', 'prepared_sha256'):
            digest(x[k])
        actual_input = input_records[x['method'], x['id']]
        for k in ('task', 'domain', 'raw_input_sha256', 'transport_sha256', 'image_SHA256', 'image_shapes'):
            assert x[k] == actual_input[k], k

    score_nodes = [n for n in ast.parse((port / 'runtime_utils.py').read_bytes()).body
                   if isinstance(n, ast.FunctionDef) and n.name in ('tokens', 'score')]
    assert {n.name for n in score_nodes} == {'tokens', 'score'}
    score_tree = ast.Module(body=score_nodes, type_ignores=[])
    scorer_hash = hashlib.sha256(ast.dump(score_tree, include_attributes=False).encode()).hexdigest()
    assert scorer_hash == original['verified']['scorer_ast_sha256'] == naive['verified']['scorer_ast_sha256']
    scope = dict(re=re, Counter=collections.Counter)
    exec(compile(score_tree, '<original Answer F1>', 'exec'), scope)
    final_rows = {'original': original['verified']['quality']['original']['per_id'],
                  'naive': naive['verified']['quality']['plain_r8_final']['per_id']}
    final = {name: keyed(values) for name, values in final_rows.items()}
    assert all(set(gold) <= set(values) for values in final.values())
    for values in final.values():
        for rid in gold:
            assert values[rid]['domain'] == gold[rid][0]['domain'] and finite(values[rid]['F1'])
    write('VERIFY_STARTED.json', dict(unix=time.time(), ready_sha256=rh, new_scores=80,
        historical_scores_recomputed=0, scorer_ast_sha256=scorer_hash, model_worker_exited=True,
        both_S19_controls_exact=True))
    acquired = {name: {} for name in METHODS}
    for x in predictions:
        value = scope['score'](x['prediction'], x['references'])['token_f1']
        assert finite(value) and 0 <= value <= 100
        acquired[x['method']][x['id']] = value
    quality = {}
    for name in METHODS:
        per_id = [dict(id=x['id'], domain=x['domain'], task=x['task'], acquired_stage=x['task'],
            acquired_F1=acquired[name][x['id']], final_F1=final[name][x['id']]['F1'],
            final_minus_acquired=final[name][x['id']]['F1'] - acquired[name][x['id']]) for x in selected]
        domains = {d: dict(QA=20, **{key: sum(x[key] for x in per_id if x['domain'] == d) / 20
                                   for key in ('acquired_F1', 'final_F1', 'final_minus_acquired')}) for d in DOMAINS}
        quality[name] = dict(domains=domains, per_id=per_id,
            two_domain_macro={key: sum(x[key] for x in domains.values()) / 2
                              for key in ('acquired_F1', 'final_F1', 'final_minus_acquired')})
    def contrast(a, b):
        result = dict(acquired_difference=a['acquired_F1'] - b['acquired_F1'],
                      final_difference=a['final_F1'] - b['final_F1'],
                      subsequent_change_difference=a['final_minus_acquired'] - b['final_minus_acquired'])
        residual = result['final_difference'] - result['acquired_difference'] - result['subsequent_change_difference']
        assert abs(residual) < 1e-12
        result['decomposition_roundoff_residual'] = residual
        return result
    per = {name: keyed(quality[name]['per_id']) for name in METHODS}
    differences = [dict(id=x['id'], domain=x['domain'], task=x['task'],
                        **contrast(per['original'][x['id']], per['naive'][x['id']])) for x in selected]
    comparisons = dict(original_minus_naive=dict(
        domains={d: contrast(quality['original']['domains'][d], quality['naive']['domains'][d]) for d in DOMAINS},
        two_domain_macro=contrast(quality['original']['two_domain_macro'], quality['naive']['two_domain_macro']),
        per_id=differences,
        counts={key: dict(up=sum(x[key] > 0 for x in differences), down=sum(x[key] < 0 for x in differences),
                          tie=sum(x[key] == 0 for x in differences))
                for key in ('acquired_difference', 'final_difference', 'subsequent_change_difference')}))
    result = dict(complete=True, ready_sha256=rh, protocol_sha256=ready['protocol_sha256'],
        source_COMPLETE_sha256=sha(RUN / (phase + '_COMPLETE.json')),
        parent_VERIFIED_sha256={n: sha(Path(p['source']['verified'])) for n, p in parents.items()},
        parent_train_COMPLETE_sha256={n: sha(Path(p['source']['train_complete'])) for n, p in parents.items()},
        parent_repair_provenance=provenance, S19_controls_sha256=sha(RUN / 'S19_CONTROLS.json'),
        predictions_sha256=sha(RUN / 'acquired_predictions.json'), scorer_ast_sha256=scorer_hash,
        quality=quality, comparisons=comparisons, selected_cases=selected,
        selected_tasks=stages, source_stage_reconstruction_sha256={e['proof']: e['sha256'] for e in events},
        methods=list(METHODS), domains=list(DOMAINS), new_generations=80, new_scores=80,
        initial_new_generations=0, final_new_generations=0, S19_new_generations=0,
        FIT_backwards=0, optimizer_updates=0, NLL_requests=0, TEST_QA=0,
        historical_scores_recomputed=0, old_final_F1_copied_directly=True,
        all_sources_and_parents_unchanged=True, both_S19_controls_before_any_new_generation=True,
        snapshot_coordinates=COORDINATES, snapshot_parameters=PARAMETERS, snapshot_FP32_bytes=ADAPTER_BYTES,
        reconstruction_weights=WEIGHTS, reconstruction_ordinary_buffer_bytes=9032,
        new_complete_checkpoints_written=0, model_phase_seconds=end['seconds'],
        parent_model_seconds_separate={n: p['verified']['model_phase_seconds'] for n, p in parents.items()},
        parent_training_GPU=dict(original=2, naive=0), readout_GPU=1,
        no_direction_or_significance_gate=True, new_framework_or_user_goal_complete_proven=False,
        interpretation='Retrospective readout of40 explored DEV IDs at each own task boundary, two methods. Acquired domain averages pool different task-stage models, not a single domain-end model. Final F1 is copied from sealed parents, never rescored. Final-minus-acquired includes later useful/adverse transfer and common FP32-GPUfold-to-FP16 deployment rounding; not pure forgetting, ER/EMA causality or architecture compression. Different original training GPUs and replay exposure remain disclosed. All positive/negative differences are reported; no new direction threshold, test selection, significance claim or automatic experiment authorization.',
        verification_seconds=time.monotonic() - tick)
    write('VERIFIED.json', result)
    print(json.dumps(dict(VERIFIED_sha256=sha(RUN / 'VERIFIED.json'), new_scores=80,
                          new_models=0, old_final_scores_recomputed=0)))


if __name__ == '__main__':
    main()
