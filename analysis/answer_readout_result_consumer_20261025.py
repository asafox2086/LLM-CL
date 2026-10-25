"""Consume only completed ARC scalar proofs; never run while results are partial.

Writes OUTCOME.json/FINDINGS.md once. No model, scorer, prediction file,
checkpoint payload, network access, bootstrap or data-dependent metric choice.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / 'summary/answer_reliability_consolidation_20261025/v1'
ARMS = ('gate', 'original', 'shuffle')
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
PHASES = tuple('train_' + a for a in ARMS) + tuple('eval_' + a for a in ARMS)
PROTOCOL_SHA = '388c19f12ea036a8b5bb2192a6c8bd0b66f824b3b1cd0b2ff90ff820d324363c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate(exposures):
    """Each already-recorded replay occurrence has one equal descriptive weight."""
    size = len(exposures)
    def mean(name):
        return math.fsum(x[name] for x in exposures) / size if size else None
    return dict(replay_exposures=size, unique_replay_QA=len({x['id'] for x in exposures}),
        mean_working_NLL=mean('working_NLL'), mean_target_NLL=mean('target_NLL'),
        mean_target_minus_working_NLL=mean('target_minus_working_NLL'),
        mean_raw_MSE=mean('raw_MSE'), mean_applied_MSE=mean('masked_MSE'),
        natural_gate_on=sum(x['natural_gate'] for x in exposures),
        applied_gate_on=sum(x['applied_mask'] for x in exposures),
        applied_gate_on_fraction=sum(x['applied_mask'] for x in exposures) / size if size else None,
        nonzero_applied_MSE_exposures=sum(x['masked_MSE'] > 0 for x in exposures),
        note='QA-exposure means of recorded token-average NLL and whole-hidden MSE; repeated replay QAs count repeatedly. Descriptive, not causal, F1 or significance evidence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(); report = args.report.resolve(); tick = time.monotonic()
    # Do not open MASK, training resources or any other running outputs before
    # both final proofs exist and their completion/binding checks pass.
    required = (report / 'VERIFIED.json', report / 'CONTINUATION_VERIFIED.json')
    if not all(p.is_file() for p in required):
        raise SystemExit('Not consumed: both local VERIFIED.json and CONTINUATION_VERIFIED.json are required. No partial statistics were read.')
    if any((report / n).exists() for n in ('OUTCOME.json', 'FINDINGS.md')):
        raise SystemExit('Not consumed: OUTCOME.json/FINDINGS.md already exists; no overwrite.')
    v, cv = (read(p) for p in required)
    assert v['complete'] and cv['complete'] and v['new_scores'] == cv['quality_scores'] == 300
    assert cv['VERIFIED_sha256'] == sha(required[0])
    assert cv['preSTARTED_model_attempts'] == cv['new_retries_of_started_model_phases'] == 0
    assert cv['actual_model_phases'] == 6 and cv['old_scientific_sources_unchanged'] and cv['READY_unchanged']
    assert cv['absolute_original_deadline_preserved'] and cv['original_unit_failed_preserved']
    assert cv['post_model_verifier_schema_repair'] and cv['old_continuation_unit_failed_preserved']
    schema_path = report / 'POST_MODEL_SCHEMA_VERIFIED.json'
    assert schema_path.is_file() and sha(schema_path) == cv['post_model_schema_VERIFIED_sha256']
    schema = read(schema_path)
    assert schema['complete'] and schema['VERIFIED_sha256'] == sha(required[0])
    assert schema['prior_failed_verification_scores'] == 0
    assert schema['first_scoring_execution_new_scores'] == 300 and schema['scoring_executions'] == 1
    assert schema['new_models'] == schema['new_generations'] == 0
    assert schema['old_eight_scientific_files_unchanged'] and schema['all_six_model_phases_preserved']
    assert schema['absolute_original_deadline_preserved']
    assert schema['original_verifier_sha256'] == '68802ceb7d4debb15b4047ade830c043f2ecf3cf1627454d8ddd319795800dee'
    correction = schema['correction']
    assert correction['old_assert'] == "assert x['phase'] == 'DEV_FINAL' and x['references'] == row['answers'] and x['generation_cap'] == task['max_new_tokens']"
    assert correction['new_assert'] == "assert 'phase' not in x and x['references'] == row['answers'] and x['generation_cap'] == task['max_new_tokens']"
    assert correction['changed_assert_line'] == 403 and correction['all_other_AST_nodes_unchanged']
    assert all(len(correction[n]) == 64 for n in ('original_AST_sha256', 'corrected_AST_sha256'))
    assert len(schema['old_units_failed_preserved']) == 2
    assert all(x == dict(ActiveState='failed', MainPID='0', ExecMainStatus='1') for x in schema['old_units_failed_preserved'].values())
    schema_permit = read(report / 'POST_MODEL_SCHEMA_PERMIT.json')
    assert sha(report / 'POST_MODEL_SCHEMA_PERMIT.json') == schema['permit_sha256']
    assert finite(schema['seconds']) and schema['seconds'] >= 0
    assert schema['original_post_model_check_endpoint_interval_seconds'] == schema_permit['original_post_model_check_endpoint_interval_seconds']
    assert schema['wrapper_sha256'] == schema_permit['wrapper_sha256'] == sha(ROOT / 'analysis/arc_post_model_schema_verification_20261025.py')
    schema_started = read(report / 'POST_MODEL_SCHEMA_STARTED.json')
    assert sha(report / 'POST_MODEL_SCHEMA_STARTED.json') == schema['STARTED_sha256']
    assert schema_started['first_attempt_score_marker_absent'] and schema_started['previous_scores'] == 0
    assert schema_started['correction'] == correction and schema_started['old_units_failed_preserved'] == schema['old_units_failed_preserved']
    assert schema['original_failure_archive'] == 'post_model_verifier_schema_failure_20261025'
    assert set(schema['original_failed_files']) == {'CONTINUATION_FAILURE.json', 'VERIFIER_EXIT.json'}
    assert schema['original_failed_files'] == schema_permit['failed_files']
    for name, expected in schema['original_failed_files'].items():
        assert sha(report / schema['original_failure_archive'] / name) == expected
    schema_failed_exit = read(report / schema['original_failure_archive'] / 'VERIFIER_EXIT.json')
    assert schema_failed_exit['returncode'] == 1 and "KeyError: 'phase'" in schema_failed_exit['stderr']
    ready = read(report / 'READY.json'); rh = sha(report / 'READY.json')
    assert {n for n in ready['sources'] if n.endswith('.py')} == {'adaptation.py', 'common.py', 'controller.py', 'deployment.py', 'loss_core.py', 'readouts.py', 'verify.py', 'worker.py'}
    assert ready['sources']['verify.py'] == schema['original_verifier_sha256']
    assert schema_permit['ready_sha256'] == rh
    assert v['ready_sha256'] == rh and sha(report / 'PROTOCOL.md') == PROTOCOL_SHA
    assert sha(report / 'ALL_EXIT.json') == v['all_exit_sha256']
    assert schema['all_exit_sha256'] == v['all_exit_sha256']
    permit = read(report / 'PRESTART_CONTINUATION_PERMIT.json')
    assert sha(report / 'PRESTART_CONTINUATION_PERMIT.json') == cv['permit_sha256']
    assert permit['ready_sha256'] == rh
    assert permit['absolute_wall_deadline'] == permit['first_ACTIVATION_unix'] + 18300
    assert schema_permit['absolute_wall_deadline'] == permit['absolute_wall_deadline']
    model_end = read(report / 'CONTINUATION_MODEL_COMPLETE.json')
    assert sha(report / 'CONTINUATION_MODEL_COMPLETE.json') == cv['continuation_model_complete_sha256']
    assert model_end['all_exit_sha256'] == v['all_exit_sha256'] and model_end['source_and_READY_unchanged']
    assert model_end['original_gate_preserved'] and model_end['old_preSTARTED_model_QA'] == 0
    assert cv['total_elapsed_since_original_activation'] <= 18300
    archive = report / 'preSTARTED_guard_attempt_20261025'
    for name, expected in permit['attempt_files'].items():
        assert sha(archive / name) == expected == cv['original_preSTARTED_attempt_archive'][name]
    for name, expected in permit['preserved_gate_files'].items():
        assert sha(report / name) == expected
    old_exit = read(archive / 'train_original_EXIT.json')
    old_failure = read(archive / 'train_original_FAILURE.json')
    old_controller_failure = read(archive / 'CONTROLLER_FAILURE.json')
    assert old_exit['returncode'] == 1 and old_exit['pid'] == permit['failed_preSTARTED_pid']
    assert 'common.py", line 28, in __init__' in old_failure['traceback']
    assert 'card=ids.inventory()[self.gpu];assert not card' in old_failure['traceback']
    assert not (report / 'CONTINUATION_FAILURE.json').exists()

    inputs = {'VERIFIED.json': sha(required[0]), 'CONTINUATION_VERIFIED.json': sha(required[1]),
              'READY.json': rh, 'PROTOCOL.md': PROTOCOL_SHA,
              'PRESTART_CONTINUATION_PERMIT.json': cv['permit_sha256'],
              'CONTINUATION_MODEL_COMPLETE.json': cv['continuation_model_complete_sha256'],
              'POST_MODEL_SCHEMA_VERIFIED.json': sha(schema_path),
              'POST_MODEL_SCHEMA_PERMIT.json': schema['permit_sha256'],
              'POST_MODEL_SCHEMA_STARTED.json': schema['STARTED_sha256']}
    phases, starts, exits = {}, {}, {}
    for phase in PHASES:
        complete = report / (phase + '_COMPLETE.json')
        assert sha(complete) == v['source_COMPLETE_sha256'][phase]
        e, st, ex = read(complete), read(report / (phase + '_STARTED.json')), read(report / (phase + '_EXIT.json'))
        assert e['ready_sha256'] == st['ready_sha256'] == rh and st['identity'] == e['identity']
        assert ex['returncode'] == 0 and ex['pid'] == st['identity']['pid']
        assert e['seconds'] == v['phase_seconds'][phase]
        phases[phase], starts[phase], exits[phase] = e, st, ex
        inputs[complete.name] = sha(complete)

    training, resources, update_times = {}, {}, {}
    for arm in ARMS:
        e = phases['train_' + arm]; files = e['files']; exposures = []; masks = []
        expected_names = {n for n in files if n.startswith(arm + '_MASK_S') and n.endswith('.json')}
        assert len(expected_names) == 87
        for name in sorted(expected_names):
            path = report / name; assert sha(path) == files[name]; m = read(path); masks.append(m)
            assert len(m['stats']) == len(m['replay_QA'])
            for row, stats in zip(m['replay_QA'], m['stats']):
                assert row['domain'] in DOMAINS and row['task'] < m['stage']
                assert all(finite(stats[n]) for n in ('working_NLL', 'target_NLL', 'raw_MSE', 'masked_MSE'))
                delta = stats['target_NLL'] - stats['working_NLL']
                assert stats['natural_gate'] == int(delta < 0)
                assert stats['applied_mask'] in (0, 1)
                assert stats['masked_MSE'] == stats['raw_MSE'] * stats['applied_mask']
                exposures.append(dict(id=row['id'], domain=row['domain'], replay_source_stage=row['task'],
                    current_training_stage=m['stage'], working_NLL=stats['working_NLL'], target_NLL=stats['target_NLL'],
                    target_minus_working_NLL=delta, raw_MSE=stats['raw_MSE'], masked_MSE=stats['masked_MSE'],
                    natural_gate=int(stats['natural_gate']), applied_mask=int(stats['applied_mask'])))
        assert len(exposures) == 886
        training[arm] = dict(overall=aggregate(exposures),
            by_current_training_stage={str(s): aggregate([x for x in exposures if x['current_training_stage'] == s]) for s in range(1, 20)},
            by_replay_source_stage={str(s): aggregate([x for x in exposures if x['replay_source_stage'] == s]) for s in range(1, 20)},
            by_replay_domain={d: aggregate([x for x in exposures if x['domain'] == d]) for d in DOMAINS},
            by_current_training_stage_and_replay_domain={str(s): {d: aggregate([x for x in exposures if x['current_training_stage'] == s and x['domain'] == d]) for d in DOMAINS} for s in range(1, 20)},
            acceptance_counts_are_exposures_not_unique_QAs=True)
        assert training[arm]['overall']['applied_gate_on'] == v['gate_statistics'][arm]['applied_acceptances']
        assert training[arm]['overall']['natural_gate_on'] == v['gate_statistics'][arm]['natural_acceptances']
        summary_name = arm + '_TRAIN_MEMORY.json'
        assert sha(report / summary_name) == files[summary_name]
        summary = read(report / summary_name)
        events_name = arm + '_training/logs/events.jsonl'
        assert sha(report / events_name) == files[events_name]
        events = [json.loads(line) for line in (report / events_name).read_bytes().splitlines()]
        assert len(events) == 87
        update_times[arm] = {str(s): dict(updates=sum(x['stage'] == s for x in events),
            recorded_update_seconds_sum=math.fsum(x['seconds'] for x in events if x['stage'] == s),
            scope='Update-loop timings only; exclude stage auxiliary/checkpoint/I/O intervals, not whole-stage wall time.') for s in range(1, 20)}
        resources[arm] = dict(sampled_process_summary=summary,
            maximum_update_recorded_CUDA_peak_allocated_bytes=max(x['gpu_peak_allocated_bytes'] for x in events),
            scope='Training process samples and stage-reset CUDA peak counters; includes working/slow/hidden/replay. Not controlled inference windows or a matched CPU peak.')

    quality = v['quality']; gate = quality['gate']['macro']
    over_original = gate > quality['original']['macro']; over_shuffle = gate > quality['shuffle']['macro']
    over_naive = gate > quality['naive']['macro']
    hamming = v['gate_statistics']['shuffle']['reference_mask_Hamming_QA']
    selection_support = over_original and over_shuffle and hamming > 0 and v['nondegenerate_reference_gate']
    assert over_original == v['gate_over_original'] and over_shuffle == v['gate_over_count_matched_shuffle'] and over_naive == v['gate_over_naive']
    observed_best = [a for a in ('naive', 'original', 'gate', 'shuffle') if quality[a]['macro'] == max(q['macro'] for q in quality.values())]
    operational_best = [a for a in ('naive', 'original', 'gate') if quality[a]['macro'] == max(quality[n]['macro'] for n in ('naive', 'original', 'gate'))]
    statements = [
        '本次只检验固定r8、末层答案读出代理对整包隐藏约束的净作用；原法移植不是新发明。',
        ('门同时高于original及同数shuffle，且对应掩码实际变化、门非退化，支持本固定代理的样本选择点收益。' if selection_support
         else '未同时满足原协议的original、同数shuffle及实际掩码变化条件，不支持本固定代理的样本选择收益。'),
        ('门高于朴素r8锚的总体点估计；该差异仍包含ER/EMA和训练GPU差异，不能全归给门。' if over_naive
         else '门未高于朴素r8锚，不声称新门带来总体净收益。'),
        '末态100已探索DEV、单seed不能区分刚习得和遗忘，不证明显著性、视觉语义或正式最强优势。',
        '打乱仅匹配每步接受数，不匹配不同轨迹的MSE/梯度总量；Hamming为0时没有样本对应干预。',
        '三臂与朴素r8均为折叠后原稠密图，不以门新增参数0冒充相对它们的新结构减重。'
    ]
    if 'original' in operational_best:
        statements.append('original是可执行配方中的最高宏之一；可如实保留更好的已知移植作为后续设计依据，不强推新门，不自动启动正式实验。')
    if 'shuffle' in observed_best:
        statements.append('参考轨迹打乱控制达到观测最高宏之一，但它不是独立在线算法，不能将该控制直接命名为方法。')
    waits = []
    for phase in PHASES[1:]:
        name = 'CONTINUATION_WAIT_' + phase + '.json'; path = report / name
        w = read(path); assert w['phase'] == phase and finite(w['seconds']) and w['seconds'] >= 0
        waits.append(dict(phase=phase, seconds=w['seconds'], metadata_checks=len(w['checks']), sha256=sha(path)))
    assert math.isclose(math.fsum(x['seconds'] for x in waits), model_end['queue_wait_seconds'], abs_tol=1e-9)
    costs = dict(phase_seconds=v['phase_seconds'], model_phase_seconds=v['model_phase_seconds'],
        controller_elapsed_through_model_completion_seconds=v['controller_seconds'],
        total_elapsed_through_continuation_verified_seconds=cv['total_elapsed_since_original_activation'],
        unique_verification_seconds=v['verification_seconds'], continuation_waits=waits,
        continuation_resource_queue_wait_seconds=model_end['queue_wait_seconds'],
        original_failed_controller_seconds=old_controller_failure.get('controller_seconds'),
        failure_to_continuation_start_elapsed_seconds=read(report / 'CONTINUATION_STARTED.json')['unix'] - old_failure['unix'],
        gap_gate_exit_to_preSTARTED_attempt_exit_seconds=old_exit['unix'] - exits['train_gate']['unix'],
        preSTARTED_failed_worker_runtime_seconds=None, preSTARTED_model_QA=0,
        original_absolute_deadline=permit['absolute_wall_deadline'], original_first_activation_unix=permit['first_ACTIVATION_unix'],
        original_deadline_preserved=True, original_attempt_archive_sha256=cv['original_preSTARTED_attempt_archive'],
        original_failure_guard_subcondition_unknown=True,
        post_model_schema_repair_seconds=schema['seconds'],
        original_post_model_check_endpoint_interval_seconds=schema['original_post_model_check_endpoint_interval_seconds'],
        post_model_original_failed_files_sha256=schema['original_failed_files'],
        prior_naive_model_seconds_separate=v['prior_naive_model_seconds_separate'],
        note='Phase durations include load/hash/save/cleanup; are not pure CUDA compute. Queue/repair/failed-controller/verification intervals overlap controller elapsed and must not be added twice. The preSTARTED gap is not measured failed-worker runtime. Historical exploration and transfer/collection costs are separate, not zeroed.')
    outcome = dict(VERIFIED_sha256=inputs['VERIFIED.json'], CONTINUATION_VERIFIED_sha256=inputs['CONTINUATION_VERIFIED.json'],
        evidence_sha256=inputs, scientific_protocol_sha256=PROTOCOL_SHA, verifier_decision_copied=v['decision'],
        question='固定r8内，末层实际max目标的答案NLL代理能否比无条件和同接受数打乱的隐藏约束更有效？',
        quality=quality, comparisons=v['comparisons'], gate_statistics=v['gate_statistics'],
        predicates=dict(gate_over_original=over_original, gate_over_count_matched_shuffle=over_shuffle,
            reference_mask_assignment_Hamming_QA=hamming, nondegenerate_reference_gate=v['nondegenerate_reference_gate'],
            selection_supported_under_fixed_protocol=selection_support, gate_over_naive=over_naive,
            new_gate_overall_net_gain_not_claimed=not over_naive),
        observed_highest_macro_arms=observed_best, highest_macro_operational_recipes=operational_best,
        interpretation=statements, training_descriptive_statistics=training,
        post_model_verifier_schema_repair=dict(receipt=schema,
            original_verifier_and_source_reviews_missed_unserialized_phase_field=True,
            all_original_scientific_files_predictions_and_checkpoints_unchanged=True,
            repair_scope='External in-memory AST replacement of one nonexistent phase assertion; explicit absence replaces the false schema assumption. Original producer call chain, COMPLETE files, IDs/references/caps/input bindings and scoring remain checked.',
            both_original_services_remain_failed=True, previous_failed_verification_scored_QA=0,
            first_scoring_execution_scored_QA=300, new_model_or_generation_retries=0),
        parameter_and_checkpoint=dict(working_coordinates=v['training_coordinates'], working_parameter_tensors=v['trainable_parameter_tensors'],
            frozen_slow_coordinates=v['frozen_slow_coordinates'], frozen_slow_bytes=v['frozen_slow_bytes'],
            learned_gate_parameters=v['learned_gate_parameters'], gradient_bytes=v['gradient_bytes'],
            GPU_optimizer_state_bytes=v['GPU_optimizer_state_bytes'], final_weights=v['final_weights'],
            final_ordinary_buffer_bytes=v['final_ordinary_buffer_bytes'], complete_checkpoints=v['complete_checkpoints'],
            final_no_EMA_or_memory_or_gate=v['final_no_EMA_or_memory_or_gate'], new_inference_coefficients_relative_to_naive=0),
        fixed_inference_windows_copied=v['memory'], naive_inference_windows_copied=v['naive_memory_copied'],
        inference_comparison_copied=v['memory_comparison'], sampled_training_resources=resources,
        per_training_stage_update_timing=update_times, costs=costs,
        work={n:v[n] for n in ('current_QA', 'replay_QA', 'FIT_backwards', 'actual_training_decoder_requests',
            'extra_target_head_calls', 'optimizer_updates', 'new_generations', 'new_scores', 'TEST_QA', 'initial_DEV_generations')},
        new_models_by_consumer=0, new_scores_by_consumer=0, bootstrap_calls=0,
        training_NLL_is_already_recorded_not_new_model_evaluation=True, selection_or_threshold_search=False,
        README_or_scientific_source_modified=False, followup_model_authorized=False,
        scalar_consumer_seconds_before_output=time.monotonic()-tick)

    lines = ['# 固定r8答案读出门：完整后置结论', '',
        '原始问题是共享小容量如何选择值得保留的历史约束，不预设旧教师更好，也不以扩容取得收益。', '',
        '三臂固定同rank／训练更新／回放／EMA；只有整包MSE准入规则不同。所有训练完成后才统一生成与唯一评分。', '',
        '## 总体与五域Answer F1', '', '| 条件 | natural | medical | geography | finance | traffic | 五域宏 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ('naive', 'original', 'gate', 'shuffle'):
        q = quality[arm]
        lines.append('| ' + arm + ' | ' + ' | '.join(f"{q['domains'][d]['F1']:.6f}" for d in DOMAINS) + f" | {q['macro']:.6f} |")
    lines += ['', *['- ' + s for s in statements], '', '逐题正负与原VF全部比较直接复制到[OUTCOME](OUTCOME.json)，没有重新评分、挑题或bootstrap。', '',
        '## 掩码对应与退化', '', '| 条件 | 自然接受/886 | 实际接受/886 | 非零实际MSE QA曝光 |', '| --- | ---: | ---: | ---: |']
    for arm in ARMS:
        g = v['gate_statistics'][arm]
        lines.append(f"| {arm} | {g['natural_acceptances']} | {g['applied_acceptances']} | {g['nonzero_applied_MSE_QA']} |")
    g = v['gate_statistics']['shuffle']
    lines += ['', f"shuffle相对reference：QA Hamming={g['reference_mask_Hamming_QA']}，改变对应的step={g['reference_mask_Hamming_steps']}；reference混合step={g['reference_mixed_mask_steps']}，单replay-QA step={g['replay_steps_size1']}。全关／全开退化保留在OUTCOME，不视作选择机制成功。", '',
        '## 已记录训练统计（描述性）', '',
        '下表每次replay曝光等权；NLL本身是该QA答案token均值，重复QA按实际曝光重复计数。更低训练目标NLL不证明生成、视觉依赖或参数更新有益。19个当前stage、19个回放来源stage及其域交叉全部保存在OUTCOME，不按结果筛选。', '',
        '| 条件／回放域 | 曝光数 | 平均target−working NLL | 平均原MSE | 实际接受数 |', '| --- | ---: | ---: | ---: | ---: |']
    def display(value):
        return '未发生' if value is None else f'{value:.6f}'
    for arm in ARMS:
        for domain in DOMAINS:
            x = training[arm]['by_replay_domain'][domain]
            lines.append(f"| {arm}/{domain} | {x['replay_exposures']} | {display(x['mean_target_minus_working_NLL'])} | {display(x['mean_raw_MSE'])} | {x['applied_gate_on']} |")
    lines += ['', '## 完整参数、包与实际资源', '',
        f"每臂working训练坐标{v['training_coordinates']:,}，另有{v['frozen_slow_coordinates']:,}冻结EMA系数，门新增0。最终完整图{v['final_weights']['coefficients']:,}系数／{v['final_weights']['bytes']:,}权重bytes，普通buffers{v['final_ordinary_buffer_bytes']:,}bytes。最终去掉EMA、回放、门及A/B；相对同样fold的朴素LoRA没有新增结构减重。", '',
        '| 条件 | 完整CP bytes |', '| --- | ---: |']
    for arm in ARMS:
        lines.append(f"| {arm} | {v['complete_checkpoints'][arm]['bytes']} |")
    lines += ['', '五个同物理GPU1、固定32新token窗口先于100DEV；下表是推理窗口，训练进程资源另列OUTCOME。五窗不代表全分布峰，也不报告未测速度。', '',
        '| 条件／域 | baseline MiB | peak allocated MiB | peak reserved MiB |', '| --- | ---: | ---: | ---: |']
    for arm, memory in [('naive', v['naive_memory_copied'])] + [(a, v['memory'][a]) for a in ARMS]:
        for x in memory:
            lines.append(f"| {arm}/{x['domain']} | {x['baseline_allocated']/2**20:.3f} | {x['peak_allocated']/2**20:.3f} | {x['peak_reserved']/2**20:.3f} |")
    lines += ['', '训练的sampled RSS／CUDA allocated、reserved／NVML及更新记录的stage-reset highwater分别保留，不当作受控CPU峰或端侧推理优势。', '',
        '## 实际成本与原失败', '', '| phase | 秒（含加载、保存和摘要） |', '| --- | ---: |']
    for phase in PHASES:
        lines.append(f"| {phase} | {v['phase_seconds'][phase]:.6f} |")
    lines += ['', f"六phase和{v['model_phase_seconds']:.6f}秒；从原激活到模型完成{v['controller_seconds']:.6f}秒，续接核验完成累计{cv['total_elapsed_since_original_activation']:.6f}秒。后两者含重叠等待／维修，不能再与phase相加。原绝对截止{permit['absolute_wall_deadline']}保持，资源队列等待{model_end['queue_wait_seconds']:.6f}秒。", '',
        '原original调度在STARTED前因owner／memory／util组合资源guard拒绝，0模型QA；失败子条件及是否NVML滞后未知。旧日志／FAILURE／EXIT／controller失败按原字节SHA归档，gate完整训练未重做，只续接五个未模型启动phase。该过程不是无故障首试，也不是科学质量阴性。', '',
        '六模型phase成功后，原验证器又因要求producer从未序列化的phase字段而失败；此前源码与交叉评审共同漏检了这一契约。失败发生在VERIFY_STARTED及任何评分之前，原两份失败文件按SHA保留，两个原服务均维持failed／exit1。外部wrapper仅在内存AST中把这一断言改为明确phase缺失，保留引用、帽、ID、文件及输入全链；八个冻结科学源、READY、生成、CP和阈值均未改，模型未重跑。随后原scorer首次且仅一次评分300条。', '',
        f"此次后模型schema修复含核验耗时{schema['seconds']:.6f}秒；原模型完成至旧验证失败端点区间{schema['original_post_model_check_endpoint_interval_seconds']:.6f}秒另记。这些区间已包含在原激活至最终核验累计墙钟中，不能重复相加或把整个区间都当纯评分时间；仍遵守原绝对截止。", '',
        f"原唯一评分300条；本消费者新增模型0／评分0。训练共{v['current_QA']} current、{v['replay_QA']} replay曝光，{v['actual_training_decoder_requests']} decoder请求及{v['extra_target_head_calls']}额外target-head，{v['optimizer_updates']}更新；不是只有{v['FIT_backwards']}个有梯度QA。朴素锚及既往探索费用分列，不重复相加也不抹去。", '',
        '本报告不授权新实验，不回写事前问题、README或冻结源。', '',
        f"证据：VERIFIED SHA `{inputs['VERIFIED.json']}`；CONTINUATION_VERIFIED SHA `{inputs['CONTINUATION_VERIFIED.json']}`。", '']
    # Only these two new files are written, after all proof/statistic reads.
    with (report / 'OUTCOME.json').open('x', encoding='utf8') as f:
        json.dump(outcome, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    with (report / 'FINDINGS.md').open('x', encoding='utf8') as f:
        f.write('\n'.join(lines))
    print(json.dumps(dict(OUTCOME_sha256=sha(report/'OUTCOME.json'), FINDINGS_sha256=sha(report/'FINDINGS.md'),
        new_models=0, new_scores=0, selection_supported=selection_support, gate_over_naive=over_naive)))


if __name__ == '__main__':
    main()
