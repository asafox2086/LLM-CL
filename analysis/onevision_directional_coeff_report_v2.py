"""Publish sealed continuous coefficient controls only; never infer performance from CPU checks."""
import hashlib
import json
import math
from pathlib import Path

from answer_f1 import answer_f1

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/onevision_diagnostics_20261016/directional_coeff_20261018/v2'
FAMILY = ROOT/'exp/onevision_directional_coeff_20261018/v2'
VIEWS = ['principal_coeff_r16', 'random_coeff_r16']


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def complete(phase, view, spec):
    pointer = REPORT/(phase+'_'+view+'_complete.json')
    if not pointer.exists():
        return False
    proof = read(pointer)
    assert proof['complete'] and proof['spec'] == spec
    run = (ROOT/proof['run']).resolve()
    assert run.is_relative_to(FAMILY.resolve())
    for rel, expected in proof['artifact_sha256'].items():
        p = (run/rel).resolve()
        assert p.is_relative_to(run) and sha(p) == expected
    assert read(run/'renderer_after_direction.json')['passed']
    if phase == 'train':
        records = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
        assert len(records) == proof['positive_finite_updates'] == 93
        assert {r['stage'] for r in records} == set(range(1, 20)) and proof['trained_stages'] == 19
        for stage in range(1, 20):
            local = [r for r in records if r['stage'] == stage]
            count = len(spec['train_ids'][str(stage)])
            assert len(local) == math.ceil(count/16) and sum(r['QA'] for r in local) == count
            assert all(r['LR'] > 0 and math.isfinite(r['loss']) and r['raw_gradient_l2'] > 0
                       and r['actual_coefficient_displacement_l2'] > 0 for r in local)
            assert not read(run/f'TRAIN_weight_changes/T{stage:02d}.json')['whole_stage_zero_functional_update']
        assert read(run/'initial_exact.json')['passed']
        if view == VIEWS[1]:
            assert read(run/'initial_exact.json')['paired_variant_exact']
    else:
        assert proof['full_QA'] == 21454 and read(run/'exact_final_reload.json')['passed']
        assert read(run/'exact_basis_reload.json')['passed']
    return True


def main():
    spec = read(REPORT/'CPU_PLAN.json')['spec']
    for name, expected in spec['source_sha256'].items():
        assert sha(ROOT/name) == expected
    assert sha(REPORT/'PROTOCOL.md') == spec['protocol_sha256']
    proofs = {p.name: sha(p) for p in REPORT.glob('*_complete.json')}
    output = REPORT/'VERIFIED_EVIDENCE.json'
    if output.exists() and read(output)['proof_sha256'] == proofs:
        return
    manifest = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest) == spec['manifest_sha256']
    gold, task_of, full_ids = {}, {}, {}
    for task in read(manifest)['tasks']:
        info = task['files']['test']; path = ROOT/info['path']
        assert sha(path) == info['sha256']
        for line in path.read_text().splitlines():
            row = json.loads(line)
            gold[row['id']] = row['answers']; task_of[row['id']] = task['task']
            full_ids.setdefault(task['domain'], []).append(row['id'])
    def score(run, label, full=False):
        tasks, domains = {}, {}
        for domain in spec['diagnostic_ids']:
            path = run/f'evaluations/{label}/{domain}_image.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert [r['id'] for r in rows] == (full_ids[domain] if full else spec['diagnostic_ids'][domain])
            selected = []
            for row in rows:
                assert math.isfinite(row['nll']) and row['references'] == gold[row['id']]
                if row.get('prediction') is None:
                    assert not full and row['id'] not in spec['generation_ids'][domain]
                    continue
                f1 = answer_f1(row['prediction'], row['references'])
                assert f1 == row['scores']['token_f1'] and row['native_task'] == task_of[row['id']]
                tasks.setdefault(row['native_task'], []).append(f1); selected.append(f1)
            domains[domain] = sum(selected)/len(selected)
        assert len(tasks) == 19 and sum(map(len, tasks.values())) == (21454 if full else 155)
        return dict(domains=domains, AP19=sum(sum(v)/len(v) for v in tasks.values())/19,
            domain_macro=sum(domains.values())/5, QA=sum(map(len, tasks.values())), full_FWT=None, full_BWT=None)
    conditions = {}
    for view in VIEWS:
        item = {}
        if complete('train', view, spec):
            run = ROOT/read(REPORT/('train_'+view+'_complete.json'))['run']
            item['initial_diagnostic'] = score(run, 'initial')
            item['final_diagnostic'] = score(run, 'final')
            item['resources'] = read(run/'memory_summary.json')
        if complete('full', view, spec):
            run = ROOT/read(REPORT/('full_'+view+'_complete.json'))['run']
            item['final_full'] = score(run, 'final_full_input_router', True)
        conditions[view] = item
    reference = None
    if any(conditions.values()):
        pointer = ROOT/'summary/onevision_diagnostics_20261016/deterministic_native/complete.json'
        assert sha(pointer) == spec['native_complete_sha256']
        native = ROOT/read(pointer)['run']
        reference = score(native, 'sequential_T19')
    output.write_text(json.dumps(dict(CPU_only=True, reporter_sha256=sha(Path(__file__)), proof_sha256=proofs,
        conditions=conditions, native_SeqFT_diagnostic_reference=reference, no_TEST_selection=True, both_fulls_retained=True,
        limits=['Native155 is not full21454 or formal V2.', 'Two candidates match budgets but use different directions; SeqFT has44x trainable coordinates.',
                'Frozen bases/Gram/SVD/checkpoint and activation costs must all be counted.', 'Missing own sparse references leave FWT/BWT null.']), indent=2)+'\n')
    lines = ['# 连续方向系数更新：论文借鉴的轻量实验', '', '| 条件 | 自然 | 医学 | 遥感 | 金融 | 交通 | AP19 | 范围 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    if reference:
        lines.append('| native SeqFT reference | '+' | '.join(f'{reference["domains"][d]:.3f}' for d in ['natural','medical','geography','finance','traffic'])+f' | {reference["AP19"]:.3f} | 同155诊断，非完整主表 |')
    for view, item in conditions.items():
        if item:
            key = 'final_full' if 'final_full' in item else 'final_diagnostic'
            s = item[key]
            lines.append('| '+view+' | '+' | '.join(f'{s["domains"][d]:.3f}' for d in ['natural','medical','geography','finance','traffic'])+f' | {s["AP19"]:.3f} | {key} |')
        else:
            lines.append('| '+view+' | — | — | — | — | — | — | 待测 |')
    lines += ['', '仅接受新完整SHA证明；准备／CPU检查不是GPU收益。两组均完整确认，原全部作业保留。',
              '', '[协议](PROTOCOL.md) · [原始核验](VERIFIED_EVIDENCE.json)']
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
