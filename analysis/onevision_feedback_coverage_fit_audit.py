"""Read-only verification of the new allocation's completed TRAIN window.

No TEST predictions are read. A completed teacher/head window is not a completed
GPU build or evidence of a generalization gain. CPU replay is audit overhead.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from answer_f1 import answer_f1
import onevision_answer_compatibility_core as core

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/onevision_diagnostics_20261016/feedback_coverage_20261018/v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    assert sha(path) == read(path.with_suffix('.json'))['sha256']
    return torch.load(path, map_location='cpu', weights_only=True)


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    attempt = read(REPORT/'build_attempt.json')
    run = (ROOT/attempt['run']).resolve()
    assert run.is_relative_to((ROOT/'exp/onevision_feedback_coverage_20261018/v1').resolve())
    required = ['coverage_state.pt', 'feedback_state.pt', 'router_raw_gradients.pt',
                'router_TRAIN.json', 'TRAIN_feedback_window.json', 'original_feedback_exact.json',
                'feature_identity.json']
    if any(not (run/name).is_file() for name in required):
        print(json.dumps(dict(pending_completed_TRAIN_window=True,
                              partial_teacher_or_TEST_predictions_read=False)))
        return
    spec = read(REPORT/'CPU_WORKER_PLAN.json')['spec']
    assert attempt['spec'] == read(run/'spec.json') == spec
    for name, expected in spec['source_sha256'].items():
        assert sha(ROOT/name) == expected
    assert sha(REPORT/'PROTOCOL.md') == spec['protocol_sha256']
    signatures = {name: sha(run/name) for name in required}
    destination = REPORT/'TRAIN_FIT_VERIFIED.json'
    if destination.exists():
        old = read(destination)
        if old['TRAIN_artifact_sha256'] == signatures and old['auditor_sha256'] == sha(Path(__file__)):
            print(json.dumps(dict(unchanged_TRAIN_window=True, no_repeated_analysis=True)))
            return
    assert read(run/'original_feedback_exact.json')['passed']
    assert read(run/'feature_identity.json')['passed']
    state, saved, raw = [load(run/name) for name in required[:3]]
    event = read(run/'router_TRAIN.json')
    ids = spec['TRAIN_ids']
    assert len(ids) == len(set(ids)) == 20
    assert ids == state['memory_ids'] == saved['TRAIN_ids'] == raw['TRAIN_ids'] == event['TRAIN_ids']
    assert saved['coverage_state_sha256'] == sha(run/'coverage_state.pt')
    assert saved['source_pool_sha256'] == spec['fixed_pool_sha256']
    manifest = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest) == spec['previous_spec']['parent_spec']['manifest_sha256']
    train, tasks = {}, {}
    for task in read(manifest)['tasks']:
        info = task['files']['train']
        path = ROOT/info['path']
        assert sha(path) == info['sha256']
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['id'] not in train
            train[row['id']], tasks[row['id']] = row, task['task']
    assert len(train) == 1313 and len({tasks[i] for i in ids}) == 19
    keys = sorted(state['pool'])
    assert keys == [12, 18, 19]
    scores, raw_scores, teacher_hashes = [], [], {}
    for key in keys:
        directory = run/f'evaluations/TRAIN_teacher_state{key}'
        path = directory/'TRAIN_memory_image.jsonl'
        teacher_hashes[str(key)] = sha(path)
        assert sha(path) == read(directory/'durable_complete.json')['files'][path.name]
        assert sha(path) == saved['teacher_sha256'][str(key)] == event['teacher_sha256'][str(key)]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r['id'] for r in rows] == ids
        for row in rows:
            assert row['references'] == train[row['id']]['answers']
            assert answer_f1(row['prediction'], row['references']) == row['scores']['token_f1']
        values = [r['scores']['token_f1'] for r in rows]
        raw_scores.append(values)
        # Preserve the worker's Python division BEFORE FP32 conversion.
        scores.append([v/100 for v in values])
    matrix = torch.tensor(scores, dtype=torch.float32)
    assert torch.equal(matrix, saved['scores'])
    groups = [tuple(train[i]['images']) for i in ids]
    theta, events, gradients = core.fit(state['memory_features'], matrix, groups)
    assert torch.equal(theta, saved['theta']) and events == event['events']
    assert len(gradients) == len(raw['gradients']) == 40
    assert all(torch.equal(a, b) for a, b in zip(gradients, raw['gradients']))
    prototypes = torch.stack([state['prototypes'][key] for key in keys])
    routes, masked = {}, {}
    allowed = core.image_mask(groups)
    for label, parameter in [('fixed', torch.zeros_like(theta)), ('learned', theta)]:
        selected = [core.infer(f, state['memory_features'], matrix, parameter, keys, prototypes)[0]
                    for f in state['memory_features']]
        routes[label] = dict(states=selected, counts=dict(Counter(selected)))
        expected = core.weights(state['memory_features'], state['memory_features'], parameter, allowed)@matrix.T
        probabilities = torch.softmax(expected/core.EXPERT_TEMPERATURE, 1)
        masked[label] = dict(soft_TRAIN_F1=float(100*(probabilities*matrix.T).sum(1).mean()),
                            maximum_tie_counts=(expected == expected.max(1, keepdim=True).values).sum(1).tolist())
    means = {str(key): sum(values)/20 for key, values in zip(keys, raw_scores)}
    oracle = sum(max(values) for values in zip(*raw_scores))/20
    result = dict(CPU_only=True, auditor_sha256=sha(Path(__file__)),
        TRAIN_artifact_sha256=signatures, TRAIN_teacher_sha256=teacher_hashes,
        source_pool_sha256=spec['fixed_pool_sha256'], TRAIN_QA=20, covered_native_tasks=19,
        teacher_predictions=60, informative_TRAIN_QA=int((matrix.max(0).values > matrix.min(0).values).sum()),
        actual_metric_parameters=theta.numel(),
        actual_nonzero_gradient_steps=sum(bool(torch.count_nonzero(g)) for g in gradients),
        actual_changed_dimensions=int(torch.count_nonzero(theta)),
        all40_raw_gradients_parameters_and_logs_exactly_replayed=True,
        first_event=events[0], last_event=events[-1],
        global_TRAIN_means=means, oracle_TRAIN_F1=oracle,
        oracle_minus_best_fixed_pp=oracle-max(means.values()),
        resubstitution_TRAIN_routes=routes,
        resubstitution_TRAIN_route_changes=sum(a != b for a, b in zip(routes['fixed']['states'], routes['learned']['states'])),
        image_excluded_TRAIN_objective=masked,
        TRAIN_feedback_resources=read(run/'TRAIN_feedback_window.json'),
        TEST_questions_answers_predictions_parsed=False, new_GPU_or_model_training=False,
        limits=['Completed TRAIN window is not completed GPU build or full-test performance.',
                'All20 score labels train theta; image exclusion is not independent validation.',
                'Resubstitution routes can include their own anchor, unlike the masked training objective.',
                'The finite oracle is only for these20 TRAIN anchors and3 states, not a TEST upper bound.',
                'Allocation also changes content; neither coverage count nor 1% sufficiency is isolated.',
                'Parameter updates are not evidence of realized TEST route or Answer F1 gain.',
                'The CPU exact replay is audit cost, not a new method or extra GPU run.'])
    destination.write_text(json.dumps(result, indent=2)+'\n')
    assert not torch.cuda.is_initialized()
    fields = ['teacher_predictions', 'covered_native_tasks', 'informative_TRAIN_QA',
              'actual_nonzero_gradient_steps', 'actual_changed_dimensions',
              'resubstitution_TRAIN_route_changes', 'oracle_minus_best_fixed_pp']
    print(json.dumps({name: result[name] for name in fields}))


if __name__ == '__main__':
    main()
