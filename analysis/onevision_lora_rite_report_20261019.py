"""Read-only, stdlib-only publication of sealed native LoRA-RITE results."""
import hashlib
import json
import math
from pathlib import Path
from answer_f1 import answer_f1

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016/lora_rite_continuous_20261019/v1'
FAMILY = ROOT / 'exp/onevision_lora_rite_continuous_20261019/v1'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def complete(phase, spec):
    p = REPORT / (phase + '_complete.json')
    if not p.exists():
        return False
    proof = read(p); assert proof['complete'] and proof['spec'] == spec
    run = (ROOT / proof['run']).resolve(); assert run.is_relative_to(FAMILY.resolve())
    for rel, expected in proof['artifact_sha256'].items():
        path = (run / rel).resolve(); assert path.is_relative_to(run) and sha(path) == expected
    assert read(run / 'renderer_gpu_gate.json')['passed']
    if phase == 'train':
        events = [json.loads(s) for s in (run / 'events.jsonl').read_text().splitlines()]
        assert len(events) == proof['positive_finite_updates'] == 93 and proof['trained_stages'] == 19
        assert read(run / 'initial_exact.json')['passed']
        for stage in range(1, 20):
            local = [r for r in events if r['stage'] == stage]
            n = len(spec['train_ids'][str(stage)])
            assert len(local) == math.ceil(n / 16) and sum(r['QA'] for r in local) == n
            assert local == read(run / f'stages/T{stage:02d}.json')['updates']
            assert all(r['LR'] > 0 and math.isfinite(r['loss']) and r['raw_factor_gradient_l2'] > 0
                       and r['actual_FP32_deltaW_l2'] > 0 for r in local)
            assert run / f'states/stage_{stage:02d}.pt' in [run / n for n in proof['artifact_sha256']]
    else:
        assert complete('train', spec)
        assert proof['full_QA'] == 21454 and read(run / 'exact_final_reload.json')['passed']
    return True


def main():
    spec = read(REPORT / 'CPU_PLAN.json')['spec']
    for rel, expected in spec['source_sha256'].items():
        assert sha(ROOT / rel) == expected
    assert sha(REPORT / 'PROTOCOL.md') == spec['protocol_sha256']
    valid = {phase: complete(phase, spec) for phase in ('train', 'full')}
    if not any(valid.values()):
        return {'train': False, 'full': False, 'no_performance_claim': True}
    proofs = {phase: sha(REPORT / (phase + '_complete.json')) for phase in valid if valid[phase]}
    output = REPORT / 'VERIFIED_EVIDENCE.json'
    if output.exists() and read(output)['proof_sha256'] == proofs:
        return valid
    manifest = ROOT / 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest) == spec['manifest_sha256']
    gold, tasks, full_ids = {}, {}, {}
    for task in read(manifest)['tasks']:
        info = task['files']['test']; p = ROOT / info['path']; assert sha(p) == info['sha256']
        for s in p.read_text().splitlines():
            row = json.loads(s); gold[row['id']] = row['answers']; tasks[row['id']] = task['task']
            full_ids.setdefault(task['domain'], []).append(row['id'])
    def score(run, label, is_full=False):
        task_scores, domain_scores = {}, {}
        for domain in spec['diagnostic_ids']:
            records = [json.loads(s) for s in (run / f'evaluations/{label}/{domain}_image.jsonl').read_text().splitlines()]
            assert [r['id'] for r in records] == (full_ids[domain] if is_full else spec['diagnostic_ids'][domain])
            scores = []
            for r in records:
                assert r['references'] == gold[r['id']] and math.isfinite(r['nll']) and r['native_task'] == tasks[r['id']]
                if r.get('prediction') is None:
                    assert not is_full and r['id'] not in spec['generation_ids'][domain]
                    continue
                value = answer_f1(r['prediction'], r['references']); assert value == r['scores']['token_f1']
                scores.append(value); task_scores.setdefault(r['native_task'], []).append(value)
            domain_scores[domain] = sum(scores) / len(scores)
        assert len(task_scores) == 19 and sum(map(len, task_scores.values())) == (21454 if is_full else 155)
        return dict(domains=domain_scores, AP19=sum(sum(v) / len(v) for v in task_scores.values()) / 19,
                    QA=sum(map(len, task_scores.values())), full_FWT=None, full_BWT=None)
    result = dict(proof_sha256=proofs, source_sha256=spec['source_sha256'], complete=valid,
                  no_TEST_selection=True, native_not_formal_V2=True, one_expert_no_memory_advantage_claim=True)
    if valid['train']:
        run = ROOT / read(REPORT / 'train_complete.json')['run']
        result['initial_diagnostic'] = score(run, 'initial'); result['final_diagnostic'] = score(run, 'final')
        pointer = ROOT / 'summary/onevision_diagnostics_20261016/deterministic_native/complete.json'
        assert sha(pointer) == spec['native_completion_sha256']
        result['native_SeqFT_diagnostic_reference'] = score(ROOT / read(pointer)['run'], 'sequential_T19')
        result['resources'] = read(run / 'memory_summary.json')
    if valid['full']:
        result['final_full'] = score(ROOT / read(REPORT / 'full_complete.json')['run'], 'final_full_input_router', True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return valid


if __name__ == '__main__':
    print(json.dumps(main()))
