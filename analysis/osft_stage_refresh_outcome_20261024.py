"""Consume the new, complete VERIFIED proof once; no model, QA, scoring or bootstrap.

Original boundary/resource observations were already checked by the unique verifier.
This consumer summarizes them without recomputing tensors or reading checkpoints.
"""
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1'
WORK = ROOT / 'exp/osft_stage_refresh_mechanism_20261024/v1/private'
CASES = ('prefix', 'Fixed', 'Refresh')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def bound_bytes(path, complete):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == complete['files'][str(path.relative_to(WORK))]
    return data


def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def summary(values):
    assert values and all(math.isfinite(x) for x in values)
    return dict(minimum=min(values), median=statistics.median(values), maximum=max(values))


def compare(a, b):
    """Pair existing scalar F1 rows: b minus a, without re-running the scorer."""
    aa = {row['id']: row for row in a['per_id']}
    bb = {row['id']: row for row in b['per_id']}
    assert len(aa) == len(bb) == 100 and aa.keys() == bb.keys()
    pairs = []
    for key, row in aa.items():
        other = bb[key]
        assert row['domain'] == other['domain']
        pairs.append(dict(id=key, domain=row['domain'], a_F1=row['F1'], b_F1=other['F1'],
                          b_minus_a_pp=other['F1'] - row['F1']))
    return dict(macro_b_minus_a_pp=b['macro'] - a['macro'],
                domains_b_minus_a_pp={d: b['domains'][d] - a['domains'][d] for d in a['domains']},
                higher=sum(x['b_minus_a_pp'] > 0 for x in pairs),
                lower=sum(x['b_minus_a_pp'] < 0 for x in pairs),
                same=sum(x['b_minus_a_pp'] == 0 for x in pairs), all_100_ID_scalar_pairs=pairs)


def boundary_summary(case, complete):
    stages = []
    all_rows = []
    for stage in range(2, 20):
        value = json.loads(bound_bytes(WORK / case / 'boundaries' / f'S{stage:02d}.json', complete))
        rows = value['matrices']
        assert value['stage'] == stage and len(rows) == value['modules'] == 168
        assert value['actual_refactor_calls'] == (168 if case == 'Refresh' else 0)
        all_rows.extend(rows)
        stages.append(dict(stage=stage, observation_seconds=value['seconds'],
                           actual_refactor_calls=value['actual_refactor_calls'],
                           matrices_with_FP16_weight_change=sum(x['actual_FP16_changed_coordinates'] > 0 for x in rows),
                           FP16_changed_coordinates=sum(x['actual_FP16_changed_coordinates'] for x in rows),
                           FP16_squared_change=sum(x['actual_FP16_squared_change'] for x in rows),
                           FP16_before_energy=sum(x['actual_FP16_before_energy'] for x in rows),
                           factor_states_changed=sum(x['factor_state_before'] != x['factor_state_after'] for x in rows)))
    projector = {}
    for side in ('U_high', 'V_high'):
        raw = [x[side]['raw_squared_projector_distance'] for x in all_rows]
        projector[side] = dict(raw_squared_projector_distance=summary(raw),
                              negative_raw_count=sum(x < 0 for x in raw),
                              clamped_nonnegative_estimate=summary([x[side]['nonnegative_estimate'] for x in all_rows]),
                              gram_error_before=summary([x[side]['gram_error_before'] for x in all_rows]),
                              gram_error_after=summary([x[side]['gram_error_after'] for x in all_rows]),
                              squared_projector_normalizer=summary([x[side]['squared_projector_normalizer'] for x in all_rows]),
                              products='FP32', scalar_reductions='FP64',
                              zero_clamp_not_proof_of_exactly_unchanged_subspace=True)
    return dict(stages=stages, matrix_observations=len(all_rows), projector=projector,
                total_boundary_observation_seconds=sum(x['observation_seconds'] for x in stages),
                original_observations_already_verified_no_tensor_recomputation=True,
                package_effect_not_pure_subspace_or_optimizer_causality=True)


def main():
    tick = time.monotonic()
    assert not (REPORT / 'OUTCOME.json').exists()
    assert not (REPORT / 'OUTCOME_STARTED.json').exists()
    proof = read(REPORT / 'VERIFIED.json')
    assert proof['complete'] and proof['ready_sha256'] == sha(REPORT / 'READY.json')
    assert proof['actual_optimizer_updates'] == 153 and proof['actual_FIT_exposures'] == 2099
    assert proof['new_generation_count'] == 480 and proof['TEST_QA'] == 0
    complete = {}
    exits = {}
    for case in CASES:
        assert sha(REPORT / f'{case}_COMPLETE.json') == proof['case_complete_sha256'][case]
        complete[case] = read(REPORT / f'{case}_COMPLETE.json')
        exits[case] = read(REPORT / f'{case}_EXIT.json')
        assert exits[case]['returncode'] == 0 and exits[case]['worker'] == complete[case]['worker']
    assert sum(exits[case]['observed_elapsed_seconds'] for case in CASES) == proof['observed_model_seconds_sum']
    write(REPORT / 'OUTCOME_STARTED.json', dict(unix=time.time(), source_VERIFIED_sha256=sha(REPORT / 'VERIFIED.json'),
                                                consumer_sha256=sha(Path(__file__)), no_new_scoring=True))
    quality = proof['quality']
    acquired = compare(quality['Fixed']['acquired'], quality['Refresh']['acquired'])
    final = compare(quality['Fixed']['final'], quality['Refresh']['final'])
    assert acquired['macro_b_minus_a_pp'] == proof['acquired_Refresh_minus_Fixed_pp']
    assert final['macro_b_minus_a_pp'] == proof['final_Refresh_minus_Fixed_pp']
    costs = {}
    for case in CASES:
        comp = complete[case]
        samples = [json.loads(line) for line in bound_bytes(WORK / case / 'resources.jsonl', comp).decode('utf8').splitlines()]
        assert samples and all(row['pid'] == comp['worker']['pid'] for row in samples)
        maxima = {}
        for name in ('VmRSS_bytes', 'cpu_uss_bytes', 'gpu_allocated_bytes', 'gpu_reserved_bytes',
                     'gpu_process_nvml_bytes'):
            values = [row[name] for row in samples if row.get(name) is not None]
            maxima[name] = max(values) if values else None
        costs[case] = dict(RUN_seconds=comp['seconds'],
                           exit_observation_seconds=exits[case]['observed_elapsed_seconds'],
                           actual_updates=comp['actual_updates'], actual_FIT=comp['actual_FIT'],
                           layout=comp['layout'], checkpoint=comp['checkpoint'],
                           whole_phase_sampled_resource_maxima=maxima, resource_sample_count=len(samples),
                           not_controlled_inference_peak_or_speed=True)
        if 'reload_layout' in comp:
            costs[case]['reload_layout'] = comp['reload_layout']
    boundaries = {case: boundary_summary(case, complete[case]) for case in CASES[1:]}
    outcome = dict(source_VERIFIED_sha256=sha(REPORT / 'VERIFIED.json'), consumer_sha256=sha(Path(__file__)),
                   quality=quality, acquired_Refresh_minus_Fixed=acquired, final_Refresh_minus_Fixed=final,
                   acquired_minus_initial={case: compare(quality['initial'], quality[case]['acquired']) for case in CASES[1:]},
                   final_minus_acquired={case: compare(quality[case]['acquired'], quality[case]['final']) for case in CASES[1:]},
                   final_minus_acquired_not_formal_BWT_or_pure_forgetting=True,
                   boundaries=boundaries, costs=costs, decision=proof['decision'],
                   total_model_RUN_seconds=sum(v['RUN_seconds'] for v in costs.values()),
                   total_exit_observation_seconds=sum(v['exit_observation_seconds'] for v in costs.values()),
                   actual_optimizer_updates=153, actual_FIT_exposures=2099, new_generation_count=480,
                   no_new_model_QA_scoring_bootstrap_or_tensor_analysis=True,
                   no_final_weight_reduction_or_new_framework_success_claim=True,
                   consumer_seconds_before_write=time.monotonic() - tick)
    write(REPORT / 'OUTCOME.json', outcome)
    write(REPORT / 'DECISION.json', dict(action=proof['decision'], source_OUTCOME_sha256=sha(REPORT / 'OUTCOME.json'),
                                        tested_package_only_not_universal_necessity=True,
                                        no_equivalence_significance_or_small_data_specific_cause_claim=True,
                                        no_automatic_compression_confirmation_or_Ours=True,
                                        no_rank_LR_extra_QA_time_or_version_rescue=True, goal_achieved=False))
    lines = ['# OSFT阶段刷新：习得与末态结果', '',
             '两分支从同一段新S1已学权重与RNG开始，低U/S/V均继续训练，阶段optimizer均重置。唯一干预是是否执行后续18次原refactor／保护参照刷新整包。', '',
             '| 状态 | 刚习得五域宏 | 末态五域宏 | 末态−刚习得 |', '| --- | ---: | ---: | ---: |']
    for case in CASES[1:]:
        aq = quality[case]['acquired']['macro']
        fq = quality[case]['final']['macro']
        lines.append(f'| {case} | {aq:.10f} | {fq:.10f} | {fq-aq:+.10f} |')
    lines += ['', f"Refresh−Fixed：刚习得{acquired['macro_b_minus_a_pp']:+.10f}pp，末态{final['macro_b_minus_a_pp']:+.10f}pp。",
              f"冻结出口：`{proof['decision']}`。", '',
              '| 领域 | Fixed刚习得 | Refresh刚习得 | Fixed末态 | Refresh末态 |', '| --- | ---: | ---: | ---: | ---: |']
    for domain in quality['initial']['domains']:
        values = [quality[case][phase]['domains'][domain]
                  for phase in ('acquired', 'final') for case in CASES[1:]]
        lines.append('| ' + str(domain) + ' | ' + ' | '.join(f'{x:.10f}' for x in values) + ' |')
    lines += ['',
              'OUTCOME保存全部100ID既评分标量、逐域正负和边界观测汇总；不重新评分或bootstrap。S1刚习得20题仅评分一次并在两臂中复制。',
              '刚习得仍含此前任务迁移，末态−刚习得不是正式BWT或纯遗忘；当前两臂不是相同总计算量。',
              '边界投影诊断使用FP32小Gram与FP64标量归约，负值与夹0均公开；夹0不是子空间精确未变。任何分差都只支持刷新整包的点结果，不能直接归因为移动子空间或解释全部OSFT优势。',
              'DEV已经探索，单seed、20QA/域、89已知组，T7/8/9/12无DEV；不称新留出、显著、稳定、等效或1%特有因果。', '',
              f"模型RUN总和{outcome['total_model_RUN_seconds']:.6f}秒；退出观察总和{outcome['total_exit_observation_seconds']:.6f}秒与之重叠，不相加。153真实更新、2099FIT呈现、480新生成、保存重载与边界观察全部计费。",
              'CPU准入、运输、唯一核验、实现、当前消费者和全部历史费用另列。全流程资源采样不是受控推理峰或速度。',
              '两臂完整USV权重均990913568系数；没有减少最终权重，也未形成保质紧凑新框架。最终trainable=0来自训练后冻结，不等于没有训练。',
              '本固定机制对照结束，不自动压缩、1313/21454或填Ours，不延时／续权重／追加rankLR或QA。', '']
    with (REPORT / 'README.md').open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps(dict(decision=proof['decision'], OUTCOME_sha256=sha(REPORT / 'OUTCOME.json'),
                          seconds=outcome['consumer_seconds_before_write'])))


if __name__ == '__main__':
    main()
