"""Read only sealed state-cache results; original Answer F1, no model/GPU."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from answer_f1 import answer_f1

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'summary/onevision_diagnostics_20261016'
REPORT = BASE/'functional_state_cache_20261018/v1'
DOMAINS = ['natural', 'medical', 'geography', 'finance', 'traffic']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    proofs = {p.name: sha(p) for p in REPORT.glob('*_complete.json')}
    destination = REPORT/'VERIFIED_EVIDENCE.json'
    if destination.exists() and json.loads(destination.read_text())['proof_sha256'] == proofs:
        return
    spec = json.loads((REPORT/'CPU_PLAN_CHECK.json').read_text())['spec']
    for name, expected in spec['source_sha256'].items():
        assert sha(ROOT/name) == expected
    native = json.loads((BASE/'deterministic_native/complete.json').read_text())
    assert sha(BASE/'deterministic_native/complete.json') == spec['native_completion_sha256']
    manifest = json.loads((ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json').read_text())
    assert spec['manifest_sha256'] == sha(ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json')
    gold, full_ids = {}, defaultdict(list)
    for task in manifest['tasks']:
        info = task['files']['test'];p = ROOT/info['path'];assert sha(p) == info['sha256']
        for line in p.read_text().splitlines():
            row = json.loads(line);gold[row['id']] = row['answers']
            full_ids[task['domain']].append(row['id'])
    results = []
    for name, expected in sorted(proofs.items()):
        path = REPORT/name;assert sha(path) == expected
        proof = json.loads(path.read_text());assert proof['complete'] and proof['spec'] == spec
        run = ROOT/proof['run'];assert run.resolve().is_relative_to((ROOT/'exp/onevision_state_cache_20261018/v1').resolve())
        for n, expected_file in proof['artifact_sha256'].items():
            p = run/n;assert p.resolve().is_relative_to(run.resolve()) and sha(p) == expected_file
        assert json.loads((run/'original_exact_gate.json').read_text())['passed']
        if proof['phase'] == 'build':
            for t in range(1, 20):
                gate = json.loads((run/f'exact_gates/T{t:02d}.json').read_text())
                assert gate['passed'] and gate['all48_AB_bitwise_equal']
            events = [json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
            events = [e for e in events if e.get('event') == 'train']
            assert len(events) == 93 and all(e['learning_rate_used'] > 0 and math.isfinite(e['loss']) and
                math.isfinite(e['gradient_l2_preclip']) and e['gradient_l2_preclip'] > 0 for e in events)
            decisions = json.loads((run/'decisions.json').read_text())
            assert len(decisions['events']) == 19
            for e in decisions['events']:
                assert e['stage'] in e['selected_states'] and len(e['selected_states']) <= 3
            for e in decisions['actual_TRAIN_NLL_calls']:
                assert spec['TRAIN_memory_arrivals'][e['TRAIN_id']] <= e['decision_stage']
                assert e['state'] <= e['decision_stage']
            views = spec['views']
        else:
            assert proof['full_QA'] == 21454
            assert json.loads((run/'candidate_reload_exact.json').read_text())['passed']
            views = [proof['view']]
        for view in views:
            label = view if proof['phase'] == 'build' else 'final_full_input_router'
            totals, tasks, count, generated = {}, defaultdict(list), 0, 0
            for domain in DOMAINS:
                rows = [json.loads(s) for s in (run/f'evaluations/{label}/{domain}_image.jsonl').read_text().splitlines()]
                if proof['phase'] == 'build':
                    ref = [json.loads(s) for s in (ROOT/native['run']/f'evaluations/sequential_T19/{domain}_image.jsonl').read_text().splitlines()]
                    assert [r['id'] for r in rows] == [r['id'] for r in ref]
                    assert {r['id'] for r in rows if 'scores' in r} == set(spec['generation_ids'][domain])
                else:
                    assert [r['id'] for r in rows] == full_ids[domain]
                scored = []
                for row in rows:
                    assert row['references'] == gold[row['id']] and math.isfinite(row['nll'])
                    if 'scores' in row:
                        f1 = answer_f1(row['prediction'], row['references'])
                        assert f1 == row['scores']['token_f1']
                        scored.append(f1);tasks[row['native_task']].append(f1)
                assert scored
                totals[domain] = sum(scored)/len(scored)
                count += len(rows);generated += len(scored)
            assert len(tasks) == 19
            assert (count, generated) == ((500, 155) if proof['phase'] == 'build' else (21454, 21454))
            results.append({'scope': 'diagnostic155' if proof['phase'] == 'build' else 'full21454', 'view': view,
                'domains': totals, 'domain_macro': sum(totals.values())/5,
                'AP19': sum(sum(v)/len(v) for v in tasks.values())/19,
                'FWT': None, 'BWT': None, 'run': proof['run']})
    out = {'CPU_only': True, 'proof_sha256': proofs, 'source_sha256': sha(Path(__file__)),
           'all_candidates_reported_without_TEST_selection': True, 'rows': results,
           'goal_large_F1_gain_not_inferred_from_completion': True}
    destination.write_text(json.dumps(out, indent=2)+'\n')
    text = '# 连续LoRA＋有界功能状态：原始Answer F1\n\n'
    for scope in ['diagnostic155', 'full21454']:
        text += '## '+scope+'\n\n| 方法 | 自然 | 医学 | 遥感 | 金融 | 交通 | 域宏 | AP19 | FWT | BWT |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for v in spec['views'] if scope == 'diagnostic155' else spec['full_candidates']:
            found = next((r for r in results if r['scope'] == scope and r['view'] == v), None)
            values = [f"{found['domains'][d]:.3f}" for d in DOMAINS]+[f"{found['domain_macro']:.3f}", f"{found['AP19']:.3f}"] if found else ['—']*7
            text += '| '+v+' | '+' | '.join(values+['—', '—'])+' |\n'
        text += '\n'
    text += '单状态两视图函数相同；诊断不冒充完整。最终单点不能计算完整FWT/BWT；保留全部既定确认。状态数量/额外计算/实际内存见运行decisions、pure_TRAIN_windows与memory_summary，不据零路由参数宣布零成本。\n'
    (REPORT/'VERIFIED_RESULTS.md').write_text(text)


if __name__ == '__main__':
    main()
