"""Verify three exited phases, then score only the 200 new DEV predictions once.

Standard library only: never load a model, rescore a sealed anchor, or bootstrap.
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

ROOT = Path(os.environ['LLMCL_ROOT'])
RUN = Path(os.environ['LLMCL_RUN'])
ASSET = Path(os.environ['LLMCL_ASSET'])
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
PHASES = {'train': (3000, 100), 'compact_eval': (450, 105),
          'parent_resource': (150, 5)}
PARENT_WEIGHTS = {'coefficients': 893675552, 'bytes': 1787351104}
COMPACT_WEIGHTS = {'coefficients': 581213056, 'bytes': 1162426112}
ADAPTER_COORDINATES = 4399104


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    with (RUN / name).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def checked_rows(path, wanted):
    rows = read(path)
    assert len(rows) == len(wanted)
    keyed = {row['id']: row for row in rows}
    assert len(keyed) == len(rows) and set(keyed) == set(wanted)
    return keyed


def main():
    tick = time.monotonic()
    assert not (RUN / 'VERIFY_STARTED.json').exists()
    assert not (RUN / 'VERIFIED.json').exists()
    ready = read(RUN / 'READY.json')
    ready_hash = sha(RUN / 'READY.json')
    assert read(RUN / 'ALL_EXIT.json')['complete'] is True
    assert ready['sources']['verify.py'] == sha(__file__)
    for name, digest in ready['sources'].items():
        assert sha(RUN / name) == digest, name
    for name, digest in ready['runtime_files'].items():
        assert sha(ASSET / name) == digest, name
    for name, digest in ready['prior_files'].items():
        assert sha(RUN / name) == digest, name
    assert sha(ROOT / 'exp/ffn_static_delete_20261025/v1/core.py') == ready['storage_helper_sha256']
    assert sha(RUN / 'fixed_vision_only.pt') == ready['vision_asset']['sha256']

    old = read(RUN / 'prior_VERIFIED.json')
    prior_ready = read(RUN / 'prior_READY.json')
    assert old['complete'] is True
    assert sha(RUN / 'prior_READY.json') == old['ready_sha256']
    previous = {}
    original_predictions = {}
    for name in ('parent', 'candidate'):
        path = RUN / ('prior_' + name + '_COMPLETE.json')
        assert sha(path) == old['source_complete_sha256'][name]
        previous[name] = read(path)
        assert previous[name]['ready_sha256'] == old['ready_sha256']
        predictions = RUN / ('prior_' + name + '_predictions.json')
        assert sha(predictions) == previous[name]['files'][name + '_predictions.json']
        original_predictions[name] = read(predictions)
    assert ready['parent']['digest'] == previous['parent']['digest']
    assert ready['parent']['sha256'] == prior_ready['parent']['sha256']
    assert ready['parent']['bytes'] == prior_ready['parent']['bytes']
    assert previous['parent']['layout']['weights'] == PARENT_WEIGHTS
    assert previous['candidate']['layout']['weights'] == COMPACT_WEIGHTS

    completions = {}
    starts = {}
    last_exit = None
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for case, (limit, generations) in PHASES.items():
        assert not (RUN / (case + '_FAILURE.json')).exists(), case
        start = read(RUN / (case + '_STARTED.json'))
        end = read(RUN / (case + '_COMPLETE.json'))
        exited = read(RUN / (case + '_EXIT.json'))
        assert start['identity'] == end['identity']
        identity = start['identity']
        assert identity['uid'] == os.getuid() and identity['pgid'] == identity['pid']
        assert start['boot'] == boot and start['GPU'] == 1
        assert exited['pid'] == identity['pid'] and exited['returncode'] == 0
        assert not (Path('/proc') / str(identity['pid'])).exists()
        assert start['ready_sha256'] == end['ready_sha256'] == ready_hash
        assert finite(end['seconds']) and 0 < end['seconds'] <= limit
        assert exited['unix'] >= start['unix']
        if last_exit is not None:
            assert start['unix'] >= last_exit
        last_exit = exited['unix']
        assert end['new_generations'] == generations
        assert end['DEV_used_for_optimization'] == end['TEST_QA'] == 0
        assert end['files']
        for name, digest in end['files'].items():
            assert sha(RUN / name) == digest, (case, name)
        starts[case] = start
        completions[case] = end
    assert sum(x['seconds'] for x in completions.values()) <= 3600
    assert len({x['identity']['pid'] for x in starts.values()}) == 3
    assert read(RUN / 'ALL_EXIT.json')['unix'] >= last_exit

    port = ASSET / 'exp/two_backbone_native_cl_20261017_v2'
    split_path = ASSET / 'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
    assert sha(split_path) == ready['split_sha256'] == prior_ready['split_sha256']
    split = read(split_path)
    policy = read(port / 'execution_policy.json')
    manifest_path = ASSET / policy['conditions']['pct01']['manifest']
    assert sha(manifest_path) == ready['manifest_sha256'] == policy['conditions']['pct01']['manifest_sha256']
    manifest = read(manifest_path)
    assert manifest['seed'] == 42 and len(manifest['tasks']) == 19
    gold = {}
    plan = []
    for expected_stage, task in enumerate(manifest['tasks'], 1):
        stage = task['task']
        assert stage == expected_stage and task['domain'] in DOMAINS
        path = ASSET / task['files']['train']['path']
        assert sha(path) == task['files']['train']['sha256']
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        assert len(rows) == task['files']['train']['count']
        by_id = {row['id']: row for row in rows}
        assert len(by_id) == len(rows)
        fit_ids = split['fit_ids'][str(stage)]
        dev_ids = split['dev_ids'][str(stage)]
        assert not set(fit_ids) & set(dev_ids)
        assert set(fit_ids) | set(dev_ids) == set(by_id)
        order = sorted(fit_ids, key=lambda rid: hashlib.sha256(f'42:{stage}:{rid}'.encode()).digest())
        order_hash = hashlib.sha256(''.join(rid + '\n' for rid in order).encode()).hexdigest()
        plan.append(dict(stage=stage, QA=len(order), updates=(len(order) + 15) // 16,
                         order_sha256=order_hash))
        for rid in dev_ids:
            assert rid not in gold
            gold[rid] = (task, by_id[rid])
    assert len(gold) == 100 and sum(x['QA'] for x in plan) == 1213
    assert sum(x['updates'] for x in plan) == 87 and plan == ready['training_plan']
    assert all(sum(task['domain'] == d for task, _ in gold.values()) == 20 for d in DOMAINS)

    train = completions['train']
    evaluation = completions['compact_eval']
    parent = completions['parent_resource']
    init = read(RUN / 'INITIALIZATION.json')
    assert init['no_S19_language_loaded'] is True
    assert init['receipt'] == dict(adapter_parameters=96, adapter_coordinates=ADAPTER_COORDINATES,
                                    identities_bytes_and_RNG_preserved=True)
    assert init['layout']['trainable'] == ADAPTER_COORDINATES
    assert init['layout']['weights']['coefficients'] == 585612160
    assert train['initial_adapter_digest'] == init['adapter_initial_digest']
    assert train['base_language_digest'] == init['base_language_digest']
    assert train['frozen_base_unchanged'] is True and train['vision_digest'] == init['vision_digest']
    assert train['FIT_QA'] == 1213 and train['updates'] == 87
    assert train['training_plan'] == plan
    assert train['training_ast_sha256'] == ready['training_ast_sha256']
    assert len(init['rank_plan']) == len(init['initialization']) == len(train['fold']) == 48
    expected_v = [f'model.layers.{i}.self_attn.v_proj' for i in range(24)]
    extra = set(sorted(expected_v, key=lambda n: hashlib.sha256(('42:coordinate-remainder:' + n).encode()).hexdigest())[:6])
    ranks = {f'model.layers.{i}.self_attn.{projection}_proj':
             65 + int(f'model.layers.{i}.self_attn.{projection}_proj' in extra)
             for i in range(24) for projection in ('q', 'v')}
    assert init['rank_plan'] == ranks
    assert {x['module'] for x in init['initialization']} == set(ranks)
    for row in init['initialization']:
        assert row['rank'] == ranks[row['module']] and row['scale'] == 4
        assert row['alpha'] == 4 * row['rank']
        assert row['prefix_r8_exact'] is True and row['extra_B_zero'] is True
        assert row['extra_A_seed'] == int(hashlib.sha256(('42:coordinate-extra-init:' + row['module']).encode()).hexdigest()[:8], 16)
    assert {x['module'] for x in train['fold']} == set(ranks)
    assert all(x['rank'] == ranks[x['module']] and x['scale'] == 4 and x['dtype'] == 'torch.float16'
               for x in train['fold'])
    assert all(x['merge_device'] == 'cpu' and x['merge_dtype'] == 'torch.float32'
               and str(x['output_device']).startswith('cuda') for x in train['fold'])

    # Recompute the inherited training-prefix AST hash without importing torch.
    tree = ast.parse((port / 'run.py').read_text())
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main_node.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'stage')
    cutoff = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'wanted' for target in n.targets))
    prefix = loop.body[:cutoff - 1]
    assert isinstance(prefix[-1], ast.If)
    actual_training_ast = hashlib.sha256(ast.dump(ast.Module(body=prefix, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    assert actual_training_ast == ready['training_ast_sha256']
    events_path = RUN / 'training/logs/events.jsonl'
    events = [json.loads(line) for line in events_path.read_bytes().splitlines()]
    assert len(events) == 87 and all(x['event'] == 'train_update' for x in events)
    assert len(list((RUN / 'training/validity').glob('stage_*.json'))) == 19
    assert len(list(RUN.glob('STAGE_*_COMPLETE.json'))) == 19
    assert len(list(RUN.glob('grad_S*.json'))) == 87
    assert len(list(RUN.glob('adapter_S*.pt'))) == 19
    assert len(list(RUN.glob('grad_S*_first.pt'))) == 19
    assert [e['stage'] for e in events] == [p['stage'] for p in plan for _ in range(p['updates'])]
    for p in plan:
        stage = p['stage']
        assert train['counts'][str(stage)] == p['updates']
        validity = read(RUN / f'training/validity/stage_{stage:02d}.json')
        assert validity['passed'] is True and validity['actual_parameters_changed'] is True
        assert validity['updates'] == validity['positive_lr_updates'] == p['updates']
        assert validity['parameters_before_sha256'] != validity['parameters_after_sha256']
        stage_record = read(RUN / f'STAGE_{stage:02d}_COMPLETE.json')
        assert stage_record['stage'] == stage and stage_record['updates'] == p['updates']
        assert stage_record['adapter_bytes'] == ADAPTER_COORDINATES * 4
        assert stage_record['frozen_digest'] == init['frozen_digest']
        assert stage_record['vision_digest'] == init['vision_digest']
        stage_events = [e for e in events if e['stage'] == stage]
        for step, event in enumerate(stage_events, 1):
            assert event['step'] == step and event['total_qa'] == p['QA']
            assert event['qa_seen'] == min(step * 16, p['QA'])
            assert event['warmup_updates'] == min(math.floor(.03 * p['updates']), p['updates'] - 1)
            assert event['gradient_clip_threshold'] == 1
            assert all(finite(event[k]) and event[k] >= 0 for k in
                       ('loss', 'gradient_l2_preclip', 'gradient_l2_postclip', 'seconds'))
            assert event['gradient_l2_postclip'] <= 1.0001
            assert event['prompt_tokens'] > 0 and event['target_tokens'] > 0
            assert event['learning_rate_used'] and all(finite(lr) and 0 < lr <= 1e-4 for lr in event['learning_rate_used'])
            assert all(finite(lr) and 0 <= lr <= 1e-4 for lr in event['learning_rate_after_scheduler'])
            gradient = read(RUN / f'grad_S{stage:02d}_{step:03d}.json')
            assert gradient == dict(parameters=96, gradient_bytes=ADAPTER_COORDINATES * 4, finite=True)
    assert len(train['counts']) == 19
    # All records used above must be covered by the successful train receipt.
    required_train = {'INITIALIZATION.json', 'initial_predictions.json', 'training/logs/events.jsonl'}
    required_train.update(f'training/validity/stage_{p["stage"]:02d}.json' for p in plan)
    required_train.update(f'STAGE_{p["stage"]:02d}_COMPLETE.json' for p in plan)
    required_train.update(f'grad_S{p["stage"]:02d}_{step:03d}.json' for p in plan for step in range(1, p['updates'] + 1))
    required_train.update(f'adapter_S{p["stage"]:02d}.pt' for p in plan)
    required_train.update(f'grad_S{p["stage"]:02d}_first.pt' for p in plan)
    assert required_train <= set(train['files'])
    assert {'final_predictions.json', 'compact_eval_memory.json'} <= set(evaluation['files'])
    assert 'parent_resource_memory.json' in parent['files']

    assert train['layout']['weights'] == evaluation['layout']['weights'] == COMPACT_WEIGHTS
    assert train['layout'] == evaluation['layout'] and train['layout']['trainable'] == 0
    assert train['digest'] == evaluation['digest']
    assert evaluation['strict_independent_static_reload'] is True
    assert evaluation['buffer_digest'] == parent['buffer_digest'] == init['buffer_digest']
    assert init['buffer_digest'] == previous['parent']['preserved_nonvision_buffers_digest']
    assert parent['digest'] == ready['parent']['digest'] == previous['parent']['digest']
    assert parent['layout'] == previous['parent']['layout']
    checkpoint = RUN / 'compact_complete.pt'
    assert checkpoint.stat().st_size == train['checkpoint']['bytes']
    assert sha(checkpoint) == train['checkpoint']['sha256']
    package_smaller = checkpoint.stat().st_size < ready['parent']['bytes']
    weight_smaller = COMPACT_WEIGHTS['bytes'] < PARENT_WEIGHTS['bytes']

    old_rows = {name: checked_rows(RUN / ('prior_' + name + '_predictions.json'), gold)
                for name in ('parent', 'candidate')}
    predictions = {label: checked_rows(RUN / (label + '_predictions.json'), gold)
                   for label in ('initial', 'final')}
    for label, by_id in predictions.items():
        for rid, row in by_id.items():
            task, original = gold[rid]
            ref = old_rows['candidate'][rid]
            assert row['task'] == task['task'] and row['domain'] == task['domain']
            assert row['references'] == original['answers'] == ref['references']
            assert row['generation_cap'] == task['max_new_tokens'] == ref['generation_cap']
            assert row['prepared_sha256'] == ref['prepared_sha256']
            assert row['transport_sha256'] == ref['transport_sha256']
            assert row['packed_tokens'] == ref['packed_tokens']
            assert isinstance(row['prediction'], str)
            assert all(isinstance(token, int) and token >= 0 for token in row['output_tokens'])
            assert len(row['output_tokens']) <= row['generation_cap']
            assert 'NLL' not in row

    compact_memory = read(RUN / 'compact_eval_memory.json')
    parent_memory = read(RUN / 'parent_resource_memory.json')
    assert len(compact_memory) == len(parent_memory) == 5
    memory_changes = []
    for domain, a, b in zip(DOMAINS, compact_memory, parent_memory):
        rid = min((rid for rid, (task, _) in gold.items() if task['domain'] == domain),
                  key=lambda rid: hashlib.sha256(rid.encode()).digest())
        for row, reference in ((a, 'candidate'), (b, 'parent')):
            ref = old_rows[reference][rid]
            assert row['id'] == rid and row['domain'] == domain and row['new_tokens'] == 32
            for key in ('prepared_sha256', 'transport_sha256', 'raw_pixels_sha256', 'packed_tokens'):
                assert row[key] == ref[key]
            assert len(row['input_shape']) == 2 and row['input_shape'][0] == 1 and row['input_shape'][1] > 0
            assert row['tiles'] > 0
            assert all(isinstance(row[key], int) and row[key] > 0 for key in
                       ('baseline_allocated', 'peak_allocated', 'peak_reserved'))
            assert row['peak_reserved'] >= row['peak_allocated'] >= row['baseline_allocated']
        for key in ('id', 'domain', 'transport_sha256', 'text_sha256', 'input_shape',
                    'raw_pixels_sha256', 'tiles', 'packed_tokens', 'new_tokens'):
            assert a[key] == b[key], (rid, key)
        memory_changes.append(dict(id=rid, domain=domain,
                                   peak_allocated_saving_bytes=b['peak_allocated'] - a['peak_allocated'],
                                   peak_reserved_saving_bytes=b['peak_reserved'] - a['peak_reserved']))
    all_peaks_lower = all(x['peak_allocated_saving_bytes'] > 0 for x in memory_changes)

    # Bind the exact historical scorer before the one-shot scoring marker.
    scorer_tree = ast.parse((port / 'runtime_utils.py').read_text())
    functions = [n for n in scorer_tree.body if isinstance(n, ast.FunctionDef) and n.name in ('tokens', 'score')]
    assert len(functions) == 2
    scorer_ast = hashlib.sha256(ast.dump(ast.Module(body=functions, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    assert scorer_ast == old['scorer_ast_sha256']
    namespace = dict(re=re, Counter=collections.Counter)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<original-score>', 'exec'), namespace)
    for name in ('parent', 'candidate'):
        sealed = old['quality'][name]
        assert len(sealed['per_id']) == 100 and {r['id'] for r in sealed['per_id']} == set(gold)
        assert all(finite(r['F1']) and 0 <= r['F1'] <= 100 for r in sealed['per_id'])
        assert all(sealed['domains'][d]['QA'] == 20 for d in DOMAINS)
    write('VERIFY_STARTED.json', dict(unix=time.time(), verifier_sha256=sha(__file__),
                                     scorer_ast_sha256=scorer_ast, new_scores=200,
                                     prior_scores_reused_without_rescoring=True))
    quality = {'parent': old['quality']['parent'], 'posthoc': old['quality']['candidate']}
    for label, by_id in predictions.items():
        values = []
        for rid in sorted(gold):
            row = by_id[rid]
            score = namespace['score'](row['prediction'], row['references'])['token_f1']
            assert finite(score) and 0 <= score <= 100
            values.append(dict(id=rid, task=row['task'], domain=row['domain'], F1=score))
        domains = {d: dict(F1=sum(x['F1'] for x in values if x['domain'] == d) / 20,
                           QA=20) for d in DOMAINS}
        quality[label] = dict(domains=domains, macro=sum(x['F1'] for x in domains.values()) / 5,
                              per_id=values)
    differences = {reference: {d: quality['final']['domains'][d]['F1'] - quality[reference]['domains'][d]['F1']
                               for d in DOMAINS} for reference in ('parent', 'posthoc', 'initial')}
    floors_passed = all(x >= 0 for x in differences['parent'].values())
    passed = floors_passed and weight_smaller and package_smaller and all_peaks_lower
    decision = ('retain_compact_first_for_independent_framework_review' if passed else
                'close_fixed_representation_87update_recipe')
    per_id_changes = {}
    for reference in ('parent', 'posthoc', 'initial'):
        prior_values = {r['id']: r['F1'] for r in quality[reference]['per_id']}
        final_values = {r['id']: r['F1'] for r in quality['final']['per_id']}
        per_id_changes[reference] = [dict(id=rid, domain=gold[rid][0]['domain'],
                                           delta_F1=final_values[rid] - prior_values[rid])
                                     for rid in sorted(gold)]
    result = dict(complete=True, ready_sha256=ready_hash,
                  prior_VERIFIED_sha256=sha(RUN / 'prior_VERIFIED.json'),
                  source_complete_sha256={case: sha(RUN / (case + '_COMPLETE.json')) for case in PHASES},
                  quality=quality, final_minus_reference= differences,
                  per_id_F1_changes=per_id_changes, all_domain_parent_floor_passed=floors_passed,
                  all5_GPU_peaks_lower=all_peaks_lower, complete_package_smaller=package_smaller,
                  complete_weight_smaller=weight_smaller, gate_passed=passed, decision=decision,
                  complete_weights_parent=PARENT_WEIGHTS, complete_weights_compact=COMPACT_WEIGHTS,
                  whole_weight_saving_fraction=1 - COMPACT_WEIGHTS['coefficients'] / PARENT_WEIGHTS['coefficients'],
                  checkpoint=train['checkpoint'], initialization=init, training_plan=plan,
                  training_AST_sha256=actual_training_ast, gradient_records=87, gradient_tensors_per_update=96,
                  adapter_coordinates=ADAPTER_COORDINATES, FIT_QA=1213, stages=19, optimizer_updates=87,
                  new_generations=210, new_scores=200, new_NLL_requests=0, TEST_QA=0,
                  prior_scores_reused_without_rescoring=True,
                  model_seconds={case: end['seconds'] for case, end in completions.items()},
                  training_memory=train['training_memory'],
                  phase_RSS_highwater_bytes={case: end['process_RSS_highwater_bytes'] for case, end in completions.items()},
                  memory=dict(parent=parent_memory, compact=compact_memory, changes=memory_changes),
                  prior_bridge_learning_and_teacher_cost_not_zero=True,
                  historical_full_checkpoint_IO_differs_no_training_speed_claim=True,
                  no_significance_equivalence_strongest_or_framework_success_claim=True,
                  scorer_ast_sha256=scorer_ast, verification_seconds=time.monotonic() - tick)
    write('VERIFIED.json', result)
    lines = ['# 紧凑表示先行：完整固定配方结果', '',
             '| 域 | 紧凑初态 | 事后替换 | 紧凑先行末态 | 完整父 | 末态−父pp |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for d in DOMAINS:
        scores = [quality[name]['domains'][d]['F1'] for name in ('initial', 'posthoc', 'final', 'parent')]
        lines.append('| ' + d + ' | ' + ' | '.join(f'{v:.6f}' for v in scores) + f' | {differences["parent"][d]:+.6f} |')
    lines += ['', f'固定出口：`{decision}`。',
              f'五域宏：初态{quality["initial"]["macro"]:.9f}，事后替换{quality["posthoc"]["macro"]:.9f}，末态{quality["final"]["macro"]:.9f}，完整父{quality["parent"]["macro"]:.9f}。',
              f'完整权重893675552→581213056系数；完整包更小={package_smaller}，五个匹配GPU峰均更低={all_peaks_lower}，各域父底线全过={floors_passed}。',
              '原1213 FIT／19阶段／87正LR更新，96个有限梯度张量／更新；仅初末200条新F1唯一评分，原两质量锚仅复制封存标量；0新增DEV NLL／TEST／bootstrap。',
              '公开全部逐域和逐题正负。仅高于事后替换图不足以保留；失败关闭本固定表示和87步配方，不加步／解冻／改桥／追参。',
              '单seed、已探索DEV每域20题；不称显著、等效、非劣、稳定、已胜正式最强或新CL框架完成。训练次序本身非自动创新，也不证明所有有效适应都必须如此。',
              '模型各phase费用及实际GPU窗口见VERIFIED；初始桥293组学习、教师前向、资产提取和历史失败费用另计，不称零教师成本。原父保存每阶段完整图、本臂主要保存adapter，IO不同，不据耗时称训练加速。', '']
    with (RUN / 'FINDINGS.md').open('x', encoding='utf8') as f:
        f.write('\n'.join(lines))
    print(json.dumps(dict(VERIFIED_sha256=sha(RUN / 'VERIFIED.json'), decision=decision,
                          final_macro=quality['final']['macro'],
                          final_minus_parent=differences['parent']), ensure_ascii=False))


if __name__ == '__main__':
    main()
