"""Independent CPU-only score/provenance publication, without model forwards."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/edge_deployment_20261019/structured_width_20261020/v1'
PRIVATE = ROOT / 'exp/edge_structured_width_20261020/v1'
TEACHER = ROOT / 'exp/edge_teacher_transfer_20261019/onevision_osft_final'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from cpu_function_space_20261020_v2 import selection
    ready = json.loads((REPORT / 'READY.json').read_text())
    for name, h in ready['sources'].items():
        assert sha(ROOT / 'analysis' / name) == h
    assert sha(REPORT / 'PROTOCOL.md') == ready['protocol_sha256']
    selected, sources, data = selection()
    assert sources == ready['input_source_sha256'] and selected == ready['selected']
    expected = [x for x in selected if x['role'] == 'probe']
    helper = TEACHER / 'runner_source/runtime_utils.py'
    assert sha(helper) == ready['teacher_runner_sources']['runtime_utils.py']
    fns = [x for x in ast.parse(helper.read_text()).body if isinstance(x, ast.FunctionDef) and x.name in ['tokens', 'score']]
    assert len(fns) == 2
    env = {'re': re, 'Counter': Counter}
    exec(compile(ast.fix_missing_locations(ast.Module(body=fns, type_ignores=[])), str(helper), 'exec'), env)
    summaries = {}
    for name in ['dense', 'weight', 'fluctuation']:
        complete = json.loads((REPORT / (name + '_COMPLETE.json')).read_text())
        assert complete['complete'] and complete['fresh_parameter_digest_exact']
        assert complete['ready_sha256'] == sha(REPORT / 'READY.json')
        assert complete['source_sha256'] == ready['sources']['edge_width_pilot_20261020.py']
        predictions = PRIVATE / (name + '_predictions.json')
        assert sha(predictions) == complete['raw_predictions_sha256']
        rows = json.loads(predictions.read_text())
        assert len(rows) == len(complete['requests']) == 10
        quality, speed = [], []
        for i, item in enumerate(expected):
            task, original = data[item['id']]
            for j, is_speed in enumerate([True, False]):
                row = rows[2 * i + j]
                public = complete['requests'][2 * i + j]
                assert row['id'] == public['id'] == original['id']
                assert row['speed_only'] == public['speed_only'] == is_speed
                assert row['domain'] == task['domain']
                assert all(math.isfinite(row[k]) and row[k] > 0 for k in ['TTFT_seconds', 'seconds'])
                if is_speed:
                    assert row['output_token_count'] == 8 and row['raw_AnswerF1'] is None
                    speed.append(row)
                else:
                    assert row['references'] == original['answers']
                    f1 = env['score'](row['prediction'], original['answers'])['token_f1']
                    assert f1 == row['raw_AnswerF1'] == public['raw_AnswerF1']
                    quality.append(row)
        package = complete['package']
        folder = ROOT / package['package']
        for rel, info in package['files'].items():
            assert sha(folder / rel) == info['sha256']
            assert (folder / rel).stat().st_size == info['bytes']
        assert package['actual_package_bytes'] == sum(x['bytes'] for x in package['files'].values())
        summaries[name] = {
            'package_bytes': package['actual_package_bytes'], 'parameters': package['parameters'],
            'F1_five_seen_TRAIN_probes': {x['domain']: x['raw_AnswerF1'] for x in quality},
            'mean_F1_five_seen_TRAIN_probes': math.fsum(x['raw_AnswerF1'] for x in quality) / 5,
            'sum_TTFT_five_forced8_requests': math.fsum(x['TTFT_seconds'] for x in speed),
            'maximum_observed_generation_RSS': max(x['generation_peak_RSS_bytes'] for x in speed),
            'fresh_process_loaded_memory': complete['memory_after_load'],
            'source_complete_sha256': sha(REPORT / (name + '_COMPLETE.json')),
        }
    baseline = summaries['dense']
    for name in ['weight', 'fluctuation']:
        summaries[name]['whole_package_reduction'] = 1 - summaries[name]['package_bytes'] / baseline['package_bytes']
        summaries[name]['five_TRAIN_F1_difference'] = summaries[name]['mean_F1_five_seen_TRAIN_probes'] - baseline['mean_F1_five_seen_TRAIN_probes']
    result = {'scope': 'structural feasibility on five fixed TRAIN probes; teacher saw all; not independent DEV or formal quality',
              'report_source_sha256': sha(__file__), 'ready_sha256': sha(REPORT / 'READY.json'),
              'all_cases': summaries, 'new_model_forwards': 0, 'goal_achieved': False,
              'single_pass_CPU_proxy_not_final_latency_or_mobile_result': True}
    (REPORT / 'VERIFIED_RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# CPU完整结构删减：三条件全部结果', '',
             '以下为已见TRAIN题的可行性探针，不是域测试F1、独立DEV或手机／GPU结果。全部条件公开，不用这里填Ours。', '',
             '| 条件 | 完整包MiB | 5条TRAIN平均F1 | 五次强制8token TTFT合计秒 | 实测推理RSS峰值MiB |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, x in summaries.items():
        lines.append(f"| {name} | {x['package_bytes']/1024**2:.3f} | {x['mean_F1_five_seen_TRAIN_probes']:.3f} | {x['sum_TTFT_five_forced8_requests']:.3f} | {x['maximum_observed_generation_RSS']/1024**2:.3f} |")
    lines += ['', '文件实际字节／严格重载可验证结构确实缩小，但质量仅5个teacher已见TRAIN探针、max64生成；单轮服务器CPU时延受负载影响，不能当最终加速比例。decode速度逐请求另见原完整证据。', '',
              '不能据此声称某领域普遍变好、结构无用或本目标已实现。只有后续相同FIT训练teacher＋独立DEV筛选、完整1313重训／21454确认及统一设备部署测量，才能判断正式方法的质量成本收益。']
    (REPORT / 'VERIFIED_RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
