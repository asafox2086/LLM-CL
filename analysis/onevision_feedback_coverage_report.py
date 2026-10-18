"""Read-only publisher of all four allocation-by-metric conditions."""
import json
import math
from pathlib import Path

import onevision_answer_compatibility_report as previous
from answer_f1 import answer_f1

ROOT = previous.ROOT
REPORT = ROOT/'summary/onevision_diagnostics_20261016/feedback_coverage_20261018/v1'
VIEWS = ['coverage_AnswerF1', 'coverage_AnswerF1_metric']
read, sha = previous.read, previous.sha


def complete(phase, view, spec):
    name = 'build' if phase == 'build' else 'full_'+view
    path = REPORT/(name+'_complete.json')
    if not path.exists():
        return False
    proof = read(path)
    assert proof['spec'] == spec
    run = previous.verify(proof, ROOT/'exp/onevision_feedback_coverage_20261018/v1')
    assert read(run/'original_feedback_exact.json')['passed']
    if phase == 'full':
        assert proof['full_QA'] == 21454 and read(run/'coverage_reload_exact.json')['passed']
    return True


def main():
    spec = read(REPORT/'CPU_WORKER_PLAN.json')['spec']
    for name, expected in spec['source_sha256'].items():
        assert sha(ROOT/name) == expected
    assert sha(REPORT/'PROTOCOL.md') == spec['protocol_sha256']
    proofs = {p.name: sha(p) for p in REPORT.glob('*_complete.json')}
    destination = REPORT/'VERIFIED_EVIDENCE.json'
    if destination.exists() and read(destination)['proof_sha256'] == proofs:
        return
    previous.main()
    old_proof = read(previous.REPORT/'build_complete.json')
    assert sha(previous.REPORT/'build_complete.json') == spec['previous_feedback_complete_sha256']
    old = previous.verify(old_proof, ROOT/'exp/onevision_answer_compatibility_20261018/v1')
    manifest = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest) == spec['previous_spec']['parent_spec']['manifest_sha256']
    gold, train, task_ids, full_ids = {}, {}, {}, {}
    for task in read(manifest)['tasks']:
        for split, target in [('train', train), ('test', gold)]:
            info = task['files'][split]
            path = ROOT/info['path']
            assert sha(path) == info['sha256']
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            target.update({row['id']: row for row in rows})
            if split == 'test':
                full_ids.setdefault(task['domain'], []).extend(row['id'] for row in rows)
                task_ids.update({row['id']: task['task'] for row in rows})
    generation_ids = spec['previous_spec']['parent_spec']['generation_ids']
    def score(run, label, full=False):
        per_task, domains, count = {}, {}, 0
        for domain, generated_ids in generation_ids.items():
            rows = [json.loads(line) for line in (run/f'evaluations/{label}/{domain}_image.jsonl').read_text().splitlines()]
            expected = full_ids[domain] if full else [json.loads(line)['id'] for line in
                (old/f'evaluations/cache_AnswerF1_metric/{domain}_image.jsonl').read_text().splitlines()]
            assert [r['id'] for r in rows] == expected and all(math.isfinite(r['nll']) for r in rows)
            generated = [r for r in rows if r.get('prediction') is not None]
            assert full or {r['id'] for r in generated} == set(generated_ids)
            values = []
            for row in generated:
                assert row['references'] == gold[row['id']]['answers']
                assert row['native_task'] == task_ids[row['id']]
                f1 = answer_f1(row['prediction'], row['references'])
                assert f1 == row['scores']['token_f1']
                per_task.setdefault(str(row['native_task']), []).append(f1)
                values.append(f1)
                count += 1
            domains[domain] = sum(values)/len(values)
        assert count == (21454 if full else 155) and len(per_task) == 19
        return dict(QA=count, domains=domains, AP19=sum(sum(v)/len(v) for v in per_task.values())/19,
                    domain_macro=sum(domains.values())/5, full_FWT=None, full_BWT=None)
    old_report = read(previous.REPORT/'VERIFIED_EVIDENCE.json')
    conditions = {v: old_report['diagnostic_conditions'][v] for v in previous.VIEWS}
    teacher, resources = None, None
    if complete('build', None, spec):
        run = ROOT/read(REPORT/'build_complete.json')['run']
        for view in VIEWS:
            conditions[view] = dict(source='new coverage build', scores=score(run, view))
        event = read(run/'router_TRAIN.json')
        assert event['TRAIN_ids'] == spec['TRAIN_ids'] and len(event['events']) == 40
        assert event['teacher_generation_QA'] == 60 and event['no_TEST_input_to_fit']
        assert all(e['LR'] == .03 and math.isfinite(e['loss']) and math.isfinite(e['raw_gradient_l2']) for e in event['events'])
        teacher = []
        for key in [12, 18, 19]:
            path = run/f'evaluations/TRAIN_teacher_state{key}/TRAIN_memory_image.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert [r['id'] for r in rows] == spec['TRAIN_ids'] and sha(path) == event['teacher_sha256'][str(key)]
            for row in rows:
                assert row['references'] == train[row['id']]['answers']
                assert row['scores']['token_f1'] == answer_f1(row['prediction'], row['references'])
            teacher.append(dict(state=key, QA=20, TRAIN_F1=sum(r['scores']['token_f1'] for r in rows)/20))
        resources = dict(process=read(run/'memory_summary.json'), TRAIN_feedback_window=read(run/'TRAIN_feedback_window.json'))
    full = {}
    for view in VIEWS:
        if complete('full', view, spec):
            run = ROOT/read(REPORT/('full_'+view+'_complete.json'))['run']
            full[view] = score(run, 'final_full_input_router', True)
    result = dict(CPU_only=True, proof_sha256=proofs, reporter_sha256=sha(Path(__file__)),
        diagnostic_conditions=conditions, full_conditions=full, actual_TRAIN_teacher=teacher,
        actual_resources=resources, all_conditions_retained=True, TEST_selection=False,
        limits=['Native155 diagnostic is not full21454 or formal V2.',
                'TRAIN native task only allocates memory; it is not an inference input.',
                'Both new fulls required regardless diagnostic; missing FWT/BWT remain null.',
                'No memory advantage or coverage-causes-improvement claim before measured controls.'])
    destination.write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# 同预算反馈覆盖×距离学习：四格受控比较', '',
        '| 条件 | 自然 | 医学 | 遥感 | 金融 | 交通 | AP19（155诊断） |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for view in previous.VIEWS+VIEWS:
        item = conditions.get(view)
        if item:
            s = item['scores']
            values = [s['domains'][d] for d in ['natural', 'medical', 'geography', 'finance', 'traffic']]+[s['AP19']]
            lines.append('| '+view+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
        else:
            lines.append('| '+view+' | — | — | — | — | — | — |')
    lines += ['', '前两格明确引用原完整封存结果，后两格仅收新完整SHA证明。缺失为待测，不是零分。主TRAIN1313／20锚／60生成／池／头预算相同；仅反馈分配改变。两个新完整21454与原全部条件保留。',
              '', '[协议](PROTOCOL.md) · [原ID／评分／资源证明](VERIFIED_EVIDENCE.json)']
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
