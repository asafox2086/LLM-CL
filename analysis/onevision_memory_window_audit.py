"""CPU read-only accounting of closed TRAIN windows in knowledge controls.

Append-only resource logs are referenced by immutable byte-prefix SHA. Sampled
RSS/NVML maxima, CUDA allocator high-water marks and parameter bytes differ.
"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'
KEYS = ['VmRSS_bytes', 'Pss_bytes', 'cpu_uss_bytes', 'gpu_allocated_bytes', 'gpu_reserved_bytes', 'gpu_process_nvml_bytes']


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def closed_windows(rows):
    result = []
    for window in [1, 2]:
        before = [r for r in rows if r['phase'] == f'fit_window_{window}_before']
        after = [r for r in rows if r['phase'] == f'fit_window_{window}_after']
        if not before or not after: continue
        assert len(before) == len(after) == 1
        a, b = before[0], after[0]; assert a['pid'] == b['pid'] and a['unix'] < b['unix']
        selected = [r for r in rows if r['pid'] == a['pid'] and a['unix'] <= r['unix'] <= b['unix']]
        result.append({'window': window, 'before': a, 'after': b,
                       'growth_bytes': {k: b[k] - a[k] if a.get(k) is not None and b.get(k) is not None else None for k in KEYS},
                       'sampled_window_max_bytes': {k: max(r[k] for r in selected if r.get(k) is not None) if any(r.get(k) is not None for r in selected) else None for k in KEYS},
                       'allocator_peak_after_reset_bytes': b.get('gpu_peak_allocated_bytes'),
                       'reserved_peak_after_reset_bytes': b.get('gpu_peak_reserved_bytes'),
                       'trainable_parameter_count_growth': b['trainable_parameters'] - a['trainable_parameters'],
                       'trainable_parameter_bytes_growth': b['trainable_parameter_bytes'] - a['trainable_parameter_bytes'],
                       'model_parameter_bytes_growth': b['model_parameter_bytes'] - a['model_parameter_bytes'],
                       'sample_count': len(selected)})
    return result


def main():
    cases = []
    for pointer in sorted((REPORT / 'knowledge_fit').glob('*_attempt.json')):
        info = json.loads(pointer.read_text()); run = ROOT / info['run']; path = run / 'resources.jsonl'
        if not path.exists(): continue
        for source, expected in info['spec']['source_sha256'].items(): assert digest(ROOT / source) == expected
        raw = path.read_bytes(); lines = raw.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'): lines = lines[:-1]
        prefix = b''.join(lines); rows = [json.loads(line) for line in lines]
        windows = closed_windows(rows)
        if not windows: continue
        cases.append({'run': info['run'], 'case': run.parts[-2], 'resource_log': str(path.relative_to(ROOT)),
                      'immutable_prefix_bytes': len(prefix), 'immutable_prefix_sha256': hashlib.sha256(prefix).hexdigest(),
                      'source_sha256': info['spec']['source_sha256'], 'closed_training_windows': windows})
    result = {'collector_sha256': digest(Path(__file__)), 'unit': 'bytes', 'cases': cases,
              'scope': 'only completed TRAIN windows; not completion of clinical/fact evaluation cases',
              'limitations': ['TRAIN-after points retain optimizer state; before has optimizer created but Adam states not yet initialized',
                              'parameter bytes exclude gradients, Adam states, activations, runtime buffers and CUDA cache',
                              'RSS/PSS/USS and processNVML peaks are periodic sampled maxima; may miss short spikes',
                              'CUDA allocator peaks after reset come from the endpoint-after snapshot, not old pre-reset before peak',
                              'VmHWM is whole-process lifetime and is NOT a per-training-window CPU peak',
                              'same PID/phase/time range excludes endpoint inference between windows; retained caches can affect baseline',
                              'memory increase with fixed parameter count is not evidence of new modules or a memory leak']}
    (REPORT / 'MEMORY_WINDOW_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    text = ['# 训练窗口内存审计', '', '只纳入before／after均已落盘的训练窗口；不公布未完成知识病例的准确率。所有原始量单位bytes，MiB=bytes/1048576。', '']
    for case in cases:
        for window in case['closed_training_windows']:
            a, b = window['before'], window['after']
            text += [f"## {case['case']}：窗口{window['window']}", '',
                     f"可训练参数{a['trainable_parameters']}→{b['trainable_parameters']}；其字节{a['trainable_parameter_bytes']}→{b['trainable_parameter_bytes']}。", '',
                     '| 指标 | before bytes | after bytes | 增量MiB | 窗口采样最大bytes |',
                     '| --- | ---: | ---: | ---: | ---: |']
            for k in KEYS:
                delta = window['growth_bytes'][k]
                text.append(f"| {k} | {a.get(k)} | {b.get(k)} | {delta / 1048576:.3f} | {window['sampled_window_max_bytes'][k]} |" if delta is not None else f'| {k} | unavailable | unavailable | unavailable | unavailable |')
            text += ['', f"CUDA allocator重置后峰值allocated={window['allocator_peak_after_reset_bytes']}、reserved={window['reserved_peak_after_reset_bytes']}。这是after端点读取的重置后峰值，不能取before中此前推理留下的peak。", '']
    text += ['## 解释限制', ''] + ['- ' + s for s in result['limitations']]
    text += ['', '参数量不变时，占用增加不能直接解释为新增模块，亦不能仅凭一次增长排除或确认泄漏。逐PID、时间范围、原始端点和追加日志前缀SHA见[JSON](MEMORY_WINDOW_AUDIT.json)。']
    (REPORT / 'MEMORY_WINDOW_AUDIT.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({'closed_windows': sum(len(c['closed_training_windows']) for c in cases),
                      'cases': [{ 'case': c['case'], 'growth': [w['growth_bytes'] for w in c['closed_training_windows']]} for c in cases]}))


if __name__ == '__main__': main()
