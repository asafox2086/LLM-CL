"""Prespecified same-20-anchor task-coverage control, TRAIN-only CPU plan.

Does not change the frozen1313 training stream, pool retention or GPU queue.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1'
REPORT = ROOT/'summary/onevision_diagnostics_20261016/feedback_coverage_20261018/v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def select20(task_ids):
    assert sorted(task_ids) == list(range(1, 20))
    assert all(task_ids[t] for t in task_ids)
    flat = [i for t in sorted(task_ids) for i in task_ids[t]]
    assert len(flat) == len(set(flat)) >= 20
    def key(ident):
        return hashlib.sha256(('42\x00'+ident).encode()).hexdigest(), ident
    chosen = {min(task_ids[t], key=key) for t in task_ids}
    chosen.add(min((i for i in flat if i not in chosen), key=key))
    assert len(chosen) == 20
    return sorted(chosen)


def checks():
    sample = {t: [f'{t}:a', f'{t}:b'] for t in range(1, 20)}
    selected = select20(sample)
    assert len(selected) == 20
    assert {int(i.split(':')[0]) for i in selected} == set(range(1, 20))
    assert selected == select20({t: list(reversed(sample[t])) for t in reversed(sample)})
    for bad in [{t: v for t, v in sample.items() if t != 9}, dict(sample, **{})]:
        if 9 in bad:
            bad[9] = []
        try:
            select20(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Missing/empty TRAIN task silently accepted')
    return 5


def main():
    count = checks()
    proof = read(SOURCE/'build_complete.json')
    assert proof['complete']
    spec = proof['spec']
    manifest = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest) == spec['manifest_sha256']
    task_ids, rows = {}, {}
    task_metadata = {}
    train_hashes = {}
    for task in read(manifest)['tasks']:
        t = task['task']
        path = ROOT/task['files']['train']['path']
        assert sha(path) == task['files']['train']['sha256']
        values = [json.loads(s) for s in path.read_text().splitlines()]
        task_ids[t] = [v['id'] for v in values]
        for row in values:
            assert row['id'] not in rows
            rows[row['id']] = dict(task=t, domain=task['domain'], images=tuple(row['images']))
        task_metadata[t] = dict(task=t, domain=task['domain'], TRAIN_QA=len(values))
        train_hashes[str(t)] = sha(path)
    assert len(rows) == 1313
    old = sorted(spec['TRAIN_memory_arrivals'])
    assert len(old) == 20 and set(old) <= set(rows)
    new = select20(task_ids)
    old_counts = Counter(rows[i]['task'] for i in old)
    new_counts = Counter(rows[i]['task'] for i in new)
    assert len(new_counts) == 19 and len(new) == len(old) == 20
    for task in task_metadata:
        task_metadata[task].update(old_feedback_QA=old_counts[task], new_feedback_QA=new_counts[task])
    result = dict(CPU_only=True, planner_sha256=sha(Path(__file__)), mechanics_checks=count,
        manifest_sha256=sha(manifest), TRAIN_file_sha256=train_hashes,
        original_pool_build_sha256=sha(SOURCE/'build_complete.json'),
        original_pool_tensor_sha256=proof['artifact_sha256']['final_state.pt'],
        actual_TRAIN_QA_unchanged=1313, original_backbone_updates_charged=93,
        old_anchor_ids=old, new_anchor_ids=new,
        old_covered_native_tasks=sorted(old_counts), new_covered_native_tasks=sorted(new_counts),
        old_missing_native_tasks=sorted(set(task_ids)-set(old_counts)),
        coverage=list(task_metadata.values()),
        selection='One minSHA256(seed42,NUL,id) per original native TRAIN task; one remaining global minSHA; sort IDs.',
        selection_inputs=['TRAIN id', 'original native TRAIN task'],
        TEST_question_answer_prediction_or_score_read=False,
        native_task_not_an_inference_router_input=True,
        controlled_change='Feedback memory allocation only; same20 anchors, pool12/18/19, 60 generations,1792/40 head and fixed constants.',
        fixed_pool_retention_and_prototypes_unchanged=True,
        planned_views=['coverage_AnswerF1', 'coverage_AnswerF1_metric'],
        original_feedback_views_and_fulls_retained=True,
        each_new_view_requires_500NLL_155generation_and_full21454=True,
        GPU_or_model_training_started=False, queue_admitted=False,
        required_actual_gates=['Fresh old feedback500/155/routes exact same frozen build.',
            'New20 frozen TRAIN features and60 raw teacher outputs/score IDs/references exact.',
            'Actual40 raw gradients/theta/loss/resources recorded; gold/native task excluded from routing.',
            'Pool weights and prototypes before/after unchanged; original1313/19 split untouched.'],
        limits=['Coverage audit does not prove missed tasks caused TEST loss.',
            'Finite20-anchor discrimination is a design limitation, not an intrinsic1% data theorem.',
            'Task-stratified memory is a controlled diagnostic, not claimed new algorithm or final vector representation.',
            'No current GPU/queued source or scheduling change occurs in this plan.'])
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'CPU_PLAN.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(old_covered_tasks=len(old_counts), new_covered_tasks=len(new_counts),
        old_missing=result['old_missing_native_tasks'], old_anchors=20, new_anchors=20,
        changed_TRAIN_QA=0, started_GPU=False, admitted=False)))


if __name__ == '__main__':
    main()
