"""Check five exited snapshot readouts, then score only 80 new answers once.

Standard library only. Never load a model or rescore a sealed prediction.
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
CASES = {'control19': (19, None), 'natural1': (1, 'natural'),
         'medical11': (11, 'medical'), 'geography15': (15, 'geography'),
         'finance16': (16, 'finance')}
BOUNDS = {'natural': 1, 'medical': 11, 'geography': 15,
          'finance': 16, 'traffic': 19}
CF_VERIFIED_SHA256 = '24fe7263efa78dc6a95dfa3a413059dd39229dfce7f788da66d520e7e5d66a54'
COORDINATES = 4399104
WEIGHTS = {'coefficients': 581213056, 'bytes': 1162426112}


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
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def keyed(rows, expected):
    result = {row['id']: row for row in rows}
    assert len(rows) == len(result) == len(expected) and set(result) == set(expected)
    return result


def main():
    tick = time.monotonic()
    assert not (RUN / 'VERIFY_STARTED.json').exists()
    assert not (RUN / 'VERIFIED.json').exists()
    ready = read(RUN / 'READY.json')
    ready_hash = sha(RUN / 'READY.json')
    assert ready['sources']['verify.py'] == sha(__file__)
    for name, digest in ready['sources'].items():
        assert sha(RUN / name) == digest, name
    for name, digest in ready['runtime_files'].items():
        assert sha(ASSET / name) == digest, name
    for name, digest in ready['prior_files'].items():
        assert sha(RUN / name) == digest, name
    assert sha(ROOT / 'exp/ffn_static_delete_20261025/v1/core.py') == ready['storage_helper_sha256']

    old = read(RUN / 'cf_VERIFIED.json')
    assert sha(RUN / 'cf_VERIFIED.json') == CF_VERIFIED_SHA256
    assert old['complete'] is True and old['prior_scores_reused_without_rescoring'] is True
    old_ready = read(RUN / 'cf_READY.json')
    assert sha(RUN / 'cf_READY.json') == old['ready_sha256']
    assert sha(RUN / 'teacher_parent_COMPLETE.json') == old_ready['prior_files']['prior_parent_COMPLETE.json']
    for name in ('adaptation.py', 'vision_interface.py', 'inherited_budget_core.py'):
        assert ready['sources'][name] == old_ready['sources'][name]
    assert ready['runtime_files'] == old_ready['runtime_files']
    assert ready['manifest_sha256'] == old_ready['manifest_sha256']
    assert ready['split_sha256'] == old_ready['split_sha256']
    assert ready['vision_asset'] == old_ready['vision_asset']
    assert sha(RUN / 'fixed_vision_only.pt') == ready['vision_asset']['sha256']
    assert (RUN / 'fixed_vision_only.pt').stat().st_size == ready['vision_asset']['bytes']
    prior_completions = {}
    for phase in ('train', 'compact_eval'):
        path = RUN / ('cf_' + phase + '_COMPLETE.json')
        assert sha(path) == old['source_complete_sha256'][phase]
        prior_completions[phase] = read(path)
        assert prior_completions[phase]['ready_sha256'] == old['ready_sha256']
    train = prior_completions['train']
    prior_eval = prior_completions['compact_eval']
    init = read(RUN / 'cf_INITIALIZATION.json')
    assert sha(RUN / 'cf_INITIALIZATION.json') == train['files']['INITIALIZATION.json']
    assert sha(RUN / 'cf_final_predictions.json') == prior_eval['files']['final_predictions.json']
    assert train['layout'] == prior_eval['layout'] and train['layout']['weights'] == WEIGHTS
    assert train['layout']['trainable'] == 0 and train['digest'] == prior_eval['digest']
    assert train['FIT_QA'] == 1213 and train['updates'] == 87
    assert init['no_S19_language_loaded'] is True and train['frozen_base_unchanged'] is True
    assert train['base_language_digest'] == init['base_language_digest']
    assert train['vision_digest'] == init['vision_digest']
    assert init['receipt'] == dict(adapter_parameters=96, adapter_coordinates=COORDINATES,
                                   identities_bytes_and_RNG_preserved=True)
    assert init['layout']['trainable'] == COORDINATES

    # The original cumulative snapshots and their original training receipts,
    # not reconstructed states, define the five fixed observation points.
    records = {}
    assert set(ready['snapshot_files']) == {f'adapter_S{stage:02}.pt' for stage, _ in CASES.values()}
    for stage, _ in CASES.values():
        adapter = f'adapter_S{stage:02}.pt'
        receipt = f'STAGE_{stage:02}_COMPLETE.json'
        assert receipt in ready['prior_files']
        assert (RUN / adapter).stat().st_size == ready['snapshot_files'][adapter]['bytes']
        assert sha(RUN / adapter) == ready['snapshot_files'][adapter]['sha256'] == train['files'][adapter]
        assert sha(RUN / receipt) == train['files'][receipt]
        records[stage] = read(RUN / receipt)
        row = records[stage]
        assert row['stage'] == stage and row['updates'] == train['counts'][str(stage)]
        assert row['adapter_bytes'] == COORDINATES * 4
        assert row['frozen_digest'] == init['frozen_digest']
        assert row['vision_digest'] == init['vision_digest']

    port = ASSET / 'exp/two_backbone_native_cl_20261017_v2'
    policy = read(port / 'execution_policy.json')
    manifest_path = ASSET / policy['conditions']['pct01']['manifest']
    assert sha(manifest_path) == ready['manifest_sha256'] == policy['conditions']['pct01']['manifest_sha256']
    manifest = read(manifest_path)
    assert manifest['seed'] == 42 and len(manifest['tasks']) == 19
    assert [t['task'] for t in manifest['tasks']] == list(range(1, 20))
    assert list(dict.fromkeys(t['domain'] for t in manifest['tasks'])) == list(BOUNDS)
    bounds = {d: max(t['task'] for t in manifest['tasks'] if t['domain'] == d) for d in BOUNDS}
    assert bounds == BOUNDS
    assert [t['domain'] for t in manifest['tasks']] == (
        ['natural'] + ['medical'] * 10 + ['geography'] * 4 + ['finance'] + ['traffic'] * 3)
    split_path = ASSET / 'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
    assert sha(split_path) == ready['split_sha256']
    split = read(split_path)
    gold = {}
    for task in manifest['tasks']:
        stage = task['task']
        source = ASSET / task['files']['train']['path']
        assert sha(source) == task['files']['train']['sha256']
        rows = [json.loads(line) for line in source.read_bytes().splitlines()]
        by_id = {row['id']: row for row in rows}
        assert len(rows) == len(by_id) == task['files']['train']['count']
        fit = split['fit_ids'][str(stage)]
        dev = split['dev_ids'][str(stage)]
        assert not set(fit) & set(dev) and set(fit) | set(dev) == set(by_id)
        for rid in dev:
            assert rid not in gold
            original = by_id[rid]
            for image in original['images']:
                assert sha(ASSET / image) == split['image_sha256'][image]
            gold[rid] = (task, original)
    assert len(gold) == 100
    assert all(sum(t['domain'] == d for t, _ in gold.values()) == 20 for d in BOUNDS)
    prior_predictions = keyed(read(RUN / 'cf_final_predictions.json'), gold)
    for label in ('final', 'parent'):
        sealed = old['quality'][label]
        values = keyed(sealed['per_id'], gold)
        assert all(finite(r['F1']) and 0 <= r['F1'] <= 100 for r in values.values())
        assert all(values[rid]['domain'] == task['domain'] for rid, (task, _) in gold.items())
        assert all(sealed['domains'][d]['QA'] == 20 for d in BOUNDS)
        assert all(finite(sealed['domains'][d]['F1']) for d in BOUNDS)

    # All five processes must finish before any new quality consumption.
    all_exit = read(RUN / 'ALL_EXIT.json')
    assert all_exit['complete'] is True
    activation_hash = sha(RUN / 'ACTIVATION.json')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    gpu = int(os.environ['LLMCL_GPU'])
    completions = {}
    starts = {}
    last_exit = None
    expected_modules = {f'model.layers.{i}.self_attn.{p}_proj' for i in range(24) for p in ('q', 'v')}
    expected_names = {n + suffix for n in expected_modules for suffix in ('.A.weight', '.B.weight')}
    assert set(init['rank_plan']) == expected_modules
    expected_shapes = {}
    for module in expected_modules:
        rank = init['rank_plan'][module]
        expected_shapes[module + '.A.weight'] = [rank, 896]
        expected_shapes[module + '.B.weight'] = [896 if module.endswith('q_proj') else 128, rank]
    expected_bytes = {n: math.prod(s) * 4 for n, s in expected_shapes.items()}
    assert len(expected_names) == 96 and sum(expected_bytes.values()) == COORDINATES * 4
    predictions = {}
    for case, (stage, domain) in CASES.items():
        assert not (RUN / (case + '_FAILURE.json')).exists()
        start = read(RUN / (case + '_STARTED.json'))
        end = read(RUN / (case + '_COMPLETE.json'))
        exited = read(RUN / (case + '_EXIT.json'))
        identity = start['identity']
        assert identity == end['identity'] and identity['uid'] == os.getuid()
        assert identity['pid'] == identity['pgid'] and exited['pid'] == identity['pid']
        assert exited['returncode'] == 0 and not (Path('/proc') / str(identity['pid'])).exists()
        assert start['boot'] == boot and start['GPU'] == gpu
        assert start['ready_sha256'] == end['ready_sha256'] == ready_hash
        assert start['activation_sha256'] == activation_hash
        assert finite(start['unix']) and finite(exited['unix']) and exited['unix'] >= start['unix']
        assert finite(end['seconds']) and 0 < end['seconds'] <= 180
        if last_exit is not None:
            assert start['unix'] >= last_exit
        last_exit = exited['unix']
        assert end['stage'] == stage and end['domain'] == domain
        assert end['new_generations'] == (0 if domain is None else 20)
        assert all(end[k] == 0 for k in ('backwards', 'optimizer_updates', 'teacher_requests',
                                         'FIT_model_requests', 'NLL_requests', 'TEST_QA'))
        assert end['layout'] == train['layout'] and end['digest_before'] == end['digest_after']
        assert end['vision_digest'] == init['vision_digest'] and end['buffer_digest'] == init['buffer_digest']
        receipt = end['reconstruction']
        assert receipt['stage'] == stage
        assert receipt['adapter_parameters'] == 96 and receipt['adapter_coordinates'] == COORDINATES
        assert receipt['adapter_digest'] == receipt['snapshot_digest'] == records[stage]['adapter_digest']
        assert receipt['snapshot_keys'] == sorted(expected_names)
        assert receipt['snapshot_shapes'] == expected_shapes
        assert receipt['snapshot_dtypes'] == {n: 'torch.float32' for n in expected_names}
        assert receipt['snapshot_tensor_bytes'] == expected_bytes
        assert receipt['snapshot_bytes'] == records[stage]['adapter_bytes'] == COORDINATES * 4
        assert receipt['before_fold_trainable_names'] == sorted(expected_names)
        assert receipt['fixed_frozen_parameter_count'] > 0
        assert all(receipt[k] == init['frozen_digest'] for k in
                   ('frozen_digest', 'frozen_digest_before', 'frozen_digest_after'))
        assert receipt['vision_digest'] == init['vision_digest'] and receipt['buffer_digest'] == init['buffer_digest']
        assert all(receipt[k] is True for k in ('adapter_parameter_identities_preserved',
                   'original_trainable_mask_preserved', 'complete_cumulative_snapshot_copied_not_added'))
        fold = end['fold']
        assert len(fold) == 48 and {r['module'] for r in fold} == expected_modules
        for row in fold:
            module = row['module']
            assert row['rank'] == init['rank_plan'][module] and row['scale'] == 4
            assert row['shape'] == [896 if module.endswith('q_proj') else 128, 896]
            assert row['dtype'] == 'torch.float16'
            assert row['merge_device'] == 'cpu' and row['merge_dtype'] == 'torch.float32'
            assert str(row['output_device']).startswith('cuda')
        name = case + '_predictions.json'
        assert set(end['files']) == {name}
        assert sha(RUN / name) == end['files'][name]
        rows = read(RUN / name)
        if domain is None:
            assert rows == [] and end['digest_before'] == train['digest']
        else:
            wanted = {rid for rid, (t, _) in gold.items() if t['domain'] == domain}
            by_id = keyed(rows, wanted)
            for rid, row in by_id.items():
                task, original = gold[rid]
                reference = prior_predictions[rid]
                assert row['stage'] == stage and row['domain'] == domain and row['task'] == task['task']
                assert row['references'] == original['answers'] == reference['references']
                assert row['generation_cap'] == task['max_new_tokens'] == reference['generation_cap']
                for key in ('prepared_sha256', 'transport_sha256', 'packed_tokens'):
                    assert row[key] == reference[key]
                assert isinstance(row['prediction'], str)
                assert isinstance(row['output_tokens'], list) and len(row['output_tokens']) <= row['generation_cap']
                assert all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in row['output_tokens'])
                assert 'NLL' not in row
            predictions[domain] = by_id
        completions[case] = end
        starts[case] = start
    assert len({x['identity']['pid'] for x in starts.values()}) == 5
    assert all_exit['unix'] >= last_exit
    assert sum(x['seconds'] for x in completions.values()) <= 900
    assert sum(x['new_generations'] for x in completions.values()) == 80

    scorer_tree = ast.parse((port / 'runtime_utils.py').read_text())
    functions = [n for n in scorer_tree.body if isinstance(n, ast.FunctionDef) and n.name in ('tokens', 'score')]
    assert len(functions) == 2
    scorer_ast = hashlib.sha256(ast.dump(ast.Module(body=functions, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    assert scorer_ast == old['scorer_ast_sha256']
    namespace = dict(re=re, Counter=collections.Counter)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<original-score>', 'exec'), namespace)
    write('VERIFY_STARTED.json', dict(unix=time.time(), verifier_sha256=sha(__file__),
                                     scorer_ast_sha256=scorer_ast, new_scores=80,
                                     prior_scores_reused_without_rescoring=True))
    final_values = {r['id']: r for r in old['quality']['final']['per_id']}
    domains = {}
    for domain, by_id in predictions.items():
        values = []
        for rid in sorted(by_id):
            row = by_id[rid]
            score = namespace['score'](row['prediction'], row['references'])['token_f1']
            assert finite(score) and 0 <= score <= 100
            final = final_values[rid]['F1']
            values.append(dict(id=rid, task=row['task'], domain=domain, stage=BOUNDS[domain],
                               domain_end_F1=score, final_F1=final, final_minus_domain_end=final - score))
        domain_end = sum(r['domain_end_F1'] for r in values) / 20
        final = old['quality']['final']['domains'][domain]['F1']
        target = old['quality']['parent']['domains'][domain]['F1']
        if domain_end >= target and final < target:
            interpretation = 'target_attained_at_domain_end_then_lost_by_final'
        elif domain_end < target and final < target:
            interpretation = 'domain_end_already_below_target_not_all_deficit_arose_after_boundary'
        elif domain_end < target and final >= target:
            interpretation = 'domain_end_below_target_final_at_or_above_target'
        else:
            interpretation = 'domain_end_and_final_at_or_above_target'
        domains[domain] = dict(stage=BOUNDS[domain], QA=20, domain_end_F1=domain_end,
            sealed_final_F1=final, sealed_parent_final_target_F1=target,
            final_minus_domain_end=final - domain_end,
            domain_end_minus_parent_final_target=domain_end - target,
            final_higher=sum(r['final_minus_domain_end'] > 0 for r in values),
            final_lower=sum(r['final_minus_domain_end'] < 0 for r in values),
            tied=sum(r['final_minus_domain_end'] == 0 for r in values),
            interpretation=interpretation, per_id=values)
    result = dict(complete=True, decision='descriptive_domain_end_readout_complete_no_new_candidate',
        ready_sha256=ready_hash, prior_VERIFIED_sha256=CF_VERIFIED_SHA256,
        source_complete_sha256={case: sha(RUN / (case + '_COMPLETE.json')) for case in CASES},
        domain_boundaries=BOUNDS, domains=domains,
        traffic_final_only=old['quality']['final']['domains']['traffic'],
        sealed_final_quality=old['quality']['final'],
        sealed_parent_final_target_quality=old['quality']['parent'],
        positive_control_digest=completions['control19']['digest_before'],
        positive_control_layout=completions['control19']['layout'],
        new_generations=80, new_scores=80, new_NLL_requests=0, optimizer_updates=0,
        backwards=0, FIT_model_requests=0, teacher_requests=0, TEST_QA=0,
        prior_scores_reused_without_rescoring=True,
        model_seconds={c: r['seconds'] for c, r in completions.items()},
        scorer_ast_sha256=scorer_ast, old_fixed_recipe_stays_closed=True,
        domain_end_not_strictly_just_acquired=True,
        parent_final_target_not_matched_stage_control=True,
        no_pure_forgetting_capacity_significance_or_framework_success_claim=True,
        verification_seconds=time.monotonic() - tick)
    write('VERIFIED.json', result)
    lines = ['# 固定域末功能读数', '',
             '全部五个重建phase成功退出，S19零生成重建与已封末态完整digest及layout完全一致。仅80条新域末回答执行一次原Answer F1；旧末态100及父末态目标只复用封存标量。', '',
             '| 域 | 固定阶段 | 域末F1 | 已封末态F1 | 父末态目标 | 末态−域末 | 末态逐题高/低/平 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for domain, row in domains.items():
        lines.append(f"| {domain} | {row['stage']} | {row['domain_end_F1']:.8f} | {row['sealed_final_F1']:.8f} | {row['sealed_parent_final_target_F1']:.8f} | {row['final_minus_domain_end']:+.8f} | {row['final_higher']}/{row['final_lower']}/{row['tied']} |")
    lines += ['', '交通S19只复用末态，不新增请求；四个域末值来自不同阶段，不作为同一checkpoint的五域宏分。逐题标量与固定解释分叉见VERIFIED。', '',
              '域末达目标后最终跌落支持存在后续净损失；域末已低仅拒绝把全部缺口归因于域后变化。医学和遥感含多个task，域末不是严格刚习得，不能排除域内遗忘。',
              '父末态只是设计目标，不是匹配阶段控制；不识别纯遗忘、容量或接口的唯一原因，不称显著、等效或保质新框架。',
              '0训练、恢复、反向、teacher、FIT模型请求、NLL或TEST。旧固定表示＋87步配方仍关闭，不据结果自动改桥、训练或选择新候选。',
              '本次五phase全部加载、fold、摘要和生成费用单列于VERIFIED，原试验及视觉桥教师/拟合历史费用仍保留。', '']
    with (RUN / 'FINDINGS.md').open('x', encoding='utf8') as f:
        f.write('\n'.join(lines))
    print(json.dumps(dict(VERIFIED_sha256=sha(RUN / 'VERIFIED.json'),
                          decision=result['decision'], domains={d: {k: r[k] for k in
                          ('domain_end_F1', 'sealed_final_F1', 'final_minus_domain_end', 'interpretation')}
                          for d, r in domains.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
