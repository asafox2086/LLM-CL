"""Independent original-F1 rescoring and fixed 2x2 bookkeeping, no model calls."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/edge_deployment_20261019/width_component_split_20261021/v1'
PRIVATE = ROOT / 'exp/edge_width_component_split_20261021/v1'
PARENT = ROOT / 'summary/edge_deployment_20261019/structured_width_20261020/v1'
OLD_PRIVATE = ROOT / 'exp/edge_structured_width_20261020/v1'
TEACHER = ROOT / 'exp/edge_teacher_transfer_20261019/onevision_osft_final'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from cpu_function_space_20261020_v2 import selection
    assert not (REPORT / 'VERIFIED_RESULTS.json').exists()
    ready = json.loads((REPORT / 'READY.json').read_text())
    for n, expected in ready['sources'].items():
        assert sha(ROOT / 'analysis' / n) == expected
    assert sha(REPORT / 'PROTOCOL.md') == ready['protocol_sha256']
    assert sha(REPORT / 'ADMISSION_REVIEW.json') == ready['admission_review_sha256']
    for n, expected in ready['parent_files'].items():
        assert sha(ROOT / n) == expected
    old_ready = json.loads((PARENT / 'READY.json').read_text())
    for n, expected in old_ready['sources'].items():
        assert sha(ROOT / 'analysis' / n) == expected
    assert sha(PARENT / 'PROTOCOL.md') == old_ready['protocol_sha256']
    assert sha(PARENT / 'CPU_GATE.json') == old_ready['CPU_gate_sha256']
    prepare_path = REPORT / 'prepare_COMPLETE.json'
    prepare = json.loads(prepare_path.read_text())
    assert prepare['ready_sha256'] == sha(REPORT / 'READY.json')
    assert prepare['source_sha256'] == ready['sources']['edge_width_split_20261021.py']
    selected, sources, rows = selection()
    assert selected == old_ready['selected'] and sources == old_ready['input_source_sha256']
    helper = TEACHER / 'runner_source/runtime_utils.py'
    assert sha(helper) == old_ready['teacher_runner_sources']['runtime_utils.py']
    functions = [n for n in ast.parse(helper.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in ['tokens', 'score']]
    assert len(functions) == 2
    env = {'Counter': Counter, 're': re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(helper), 'exec'), env)
    expected_ids = [x['id'] for x in selected if x['role'] == 'probe']
    summaries = {}
    for name in ['dense', 'vision_only', 'text_only', 'fluctuation']:
        old = name in ['dense', 'fluctuation']
        directory = PARENT if old else REPORT
        private = OLD_PRIVATE if old else PRIVATE
        complete_path = directory / (name + '_COMPLETE.json')
        complete = json.loads(complete_path.read_text())
        assert complete['complete'] and complete['fresh_parameter_digest_exact']
        assert complete['ready_sha256'] == sha(directory / 'READY.json')
        assert complete['source_sha256'] == (old_ready['sources']['edge_width_pilot_20261020.py'] if old else ready['sources']['edge_width_split_20261021.py'])
        if not old:
            assert complete['prepare_sha256'] == sha(prepare_path)
            assert complete['package'] == prepare['packages'][name]
        predictions = private / (name + '_predictions.json')
        assert sha(predictions) == complete['raw_predictions_sha256']
        records = json.loads(predictions.read_text())
        assert len(records) == len(complete['requests']) == 10
        free, speed = [], []
        for i, qid in enumerate(expected_ids):
            task, original = rows[qid]
            for j, is_speed in enumerate([True, False]):
                item, public = records[2 * i + j], complete['requests'][2 * i + j]
                assert item['id'] == public['id'] == qid
                assert item['speed_only'] == public['speed_only'] == is_speed
                assert item['domain'] == task['domain']
                assert all(math.isfinite(item[k]) and item[k] > 0 for k in ['seconds', 'TTFT_seconds'])
                if is_speed:
                    assert item['output_token_count'] == 8 and item['raw_AnswerF1'] is None
                    speed.append(item)
                else:
                    assert item['references'] == original['answers']
                    f1 = env['score'](item['prediction'], original['answers'])['token_f1']
                    assert f1 == item['raw_AnswerF1'] == public['raw_AnswerF1']
                    free.append({'id': qid, 'domain': task['domain'], 'F1': f1})
        package = complete['package']
        for n, info in package['files'].items():
            path = ROOT / package['package'] / n
            assert sha(path) == info['sha256'] and path.stat().st_size == info['bytes']
        assert package['actual_package_bytes'] == sum(x['bytes'] for x in package['files'].values())
        summaries[name] = {
            'inherited_old_control': old, 'complete_sha256': sha(complete_path),
            'package_bytes': package['actual_package_bytes'], 'parameters': package['parameters'],
            'parameter_digest': package['parameter_digest'], 'per_QA': free,
            'mean_F1_five_seen_TRAIN': math.fsum(x['F1'] for x in free) / 5,
            'memory_after_load': complete['memory_after_load'],
            'all10_generation_peak_RSS_bytes': max(x['generation_peak_RSS_bytes'] for x in records),
            'forced8_TTFT_sum_seconds': math.fsum(x['TTFT_seconds'] for x in speed),
            'forced8_decode_tps': [x['decode_tokens_per_second'] for x in speed],
            'case_seconds': complete['seconds'],
        }
    factorial = []
    for i, qid in enumerate(expected_ids):
        f00, f10, f01, f11 = [summaries[n]['per_QA'][i]['F1'] for n in ['dense', 'vision_only', 'text_only', 'fluctuation']]
        dv, dl, interaction = f10 - f00, f01 - f00, f11 - f10 - f01 + f00
        assert math.isclose(f11 - f00, dv + dl + interaction, abs_tol=1e-10)
        factorial.append({'id': qid, 'domain': summaries['dense']['per_QA'][i]['domain'],
                          'F00': f00, 'F10': f10, 'F01': f01, 'F11': f11,
                          'vision_only_difference': dv, 'text_only_difference': dl,
                          'interaction_on_F1_scale': interaction})
    launched = json.loads((REPORT / 'LAUNCH.json').read_text())
    result = {'cases': summaries, 'factorial_per_QA': factorial,
              'factorial_means': {k: math.fsum(x[k] for x in factorial) / 5 for k in
                                  ['vision_only_difference', 'text_only_difference', 'interaction_on_F1_scale']},
              'source_sha256': sha(__file__), 'ready_sha256': sha(REPORT / 'READY.json'),
              'new_internal_prepare_and_two_case_seconds': prepare['seconds'] + sum(summaries[n]['case_seconds'] for n in ['vision_only', 'text_only']),
              'pipeline_wall_seconds_through_report_including_prechecks': time.time() - launched['started_unix'],
              'gate_admission_and_original_teacher_costs_separate': True,
              'scope': 'fixed 2x2 feasibility on five teacher-seen TRAIN QAs; not independent DEV, domain F1, equivalence or significant effect',
              'old_new_latency_not_a_controlled_speedup_comparison': True,
              'original_TEST_requests': 0, 'new_optimizer_updates': 0, 'goal_achieved': False}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    (REPORT / 'VERIFIED_RESULTS.json').write_text(text)
    lines = ['# 视觉/语言FFN隔离：固定2×2结果', '',
             '五个teacher已见TRAIN探针，不是域测试或独立DEV；医学/金融原完整参考为0，不能据均分声称保留五域能力。旧/新CPU时序不同，时延只列观察，不称受控加速。', '',
             '| 条件 | 实际完整包MiB | 原5TRAIN平均F1 | 加载RSS MiB | 全10生成峰值RSS MiB |',
             '|---|---:|---:|---:|---:|']
    for n, x in summaries.items():
        lines.append(f"| {n} | {x['package_bytes']/1024**2:.3f} | {x['mean_F1_five_seen_TRAIN']:.6f} | {x['memory_after_load']['rss']/1024**2:.3f} | {x['all10_generation_peak_RSS_bytes']/1024**2:.3f} |")
    lines += ['', '| 原probe领域 | 完整 | 仅视觉半宽 | 仅语言半宽 | 两支半宽 | F1交互记账 |',
              '|---|---:|---:|---:|---:|---:|']
    for x in factorial:
        lines.append(f"| {x['domain']} | {x['F00']:.6f} | {x['F10']:.6f} | {x['F01']:.6f} | {x['F11']:.6f} | {x['interaction_on_F1_scale']:.6f} |")
    lines += ['', '交互只表示这些题在F1尺度的非加性，不是机制/显著性。全部条件保留；按冻结协议决定停止固定无恢复条件或仅保留未来FIT-teacher/独立DEV候选，不自动21454，不填Ours。',
              '', '新增流水线墙钟秒数（含前置hash与至报告阶段）：' + str(result['pipeline_wall_seconds_through_report_including_prechecks']) + '；阶段内计时：' + str(result['new_internal_prepare_and_two_case_seconds']) + '；准入门槛/原teacher/校准/历史费用另计。']
    (REPORT / 'VERIFIED_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'complete_verified': True, 'result_sha256': sha(REPORT / 'VERIFIED_RESULTS.json')}))


if __name__ == '__main__':
    main()
