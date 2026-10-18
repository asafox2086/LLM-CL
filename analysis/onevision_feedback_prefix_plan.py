"""CPU plan for chronological sparse evaluation of the frozen feedback method.

No model/QA generation, no TEST-answer parsing, no GPU or scheduler admission.
Final learned heads must never stand in for earlier, unseen TRAIN prefixes.
"""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1'
FEEDBACK = ROOT/'summary/onevision_diagnostics_20261016/answer_compatibility_20261018/v1'
REPORT = FEEDBACK/'prefix_sparse'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def validate_prefix(stage, state, arrivals, cap=3):
    assert state['stage'] == stage
    keys = sorted(state['pool'])
    assert 1 <= len(keys) <= cap and stage in keys and all(1 <= k <= stage for k in keys)
    ids = state['memory_ids']
    assert len(ids) == len(set(ids)) and set(ids) == {i for i, t in arrivals.items() if t <= stage}
    assert all(arrivals[i] <= stage for i in ids)
    assert state['memory_features'].shape == (len(ids), 1792)
    assert torch.isfinite(state['memory_features']).all()
    assert set(state['prototypes']) == set(state['losses']) == set(keys)
    assert all(set(state['losses'][k]) == set(ids) for k in keys)
    return keys, ids


def mechanics_checks():
    arrivals = {'old': 1, 'future': 3}
    sample = dict(stage=1, pool={1: {}}, memory_ids=['old'],
                  memory_features=torch.ones(1, 1792), prototypes={1: {}}, losses={1: {'old': 0.}})
    assert validate_prefix(1, sample, arrivals) == ([1], ['old'])
    for changes in [dict(pool={1: {}, 3: {}}), dict(memory_ids=['old', 'future']), dict(stage=2)]:
        try:
            validate_prefix(1, dict(sample, **changes), arrivals)
        except AssertionError:
            pass
        else:
            raise AssertionError('Chronology failure not detected')
    return 4


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    checks = mechanics_checks()
    proof = read(SOURCE/'build_complete.json')
    assert proof['complete']
    run = (ROOT/proof['run']).resolve()
    assert run.is_relative_to((ROOT/'exp/onevision_state_cache_20261018/v1').resolve())
    spec = proof['spec']
    for name, digest in spec['source_sha256'].items():
        assert sha(ROOT/name) == digest
    assert sha(SOURCE/'PROTOCOL.md') == spec['protocol_sha256']
    manifest = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest) == spec['manifest_sha256']
    tasks = read(manifest)['tasks']
    assert [t['task'] for t in tasks] == list(range(1, 20))
    counts, train_ids = {}, {}
    test_file_hashes = {}
    for task in tasks:
        stage = task['task']
        # TEST bytes are hashed only. No TEST question, answer or prediction is parsed.
        test = task['files']['test']
        path = ROOT/test['path']
        assert sha(path) == test['sha256']
        counts[stage] = sum(1 for line in path.open('rb') if line.strip())
        test_file_hashes[str(stage)] = test['sha256']
        train = task['files']['train']
        path = ROOT/train['path']
        assert sha(path) == train['sha256']
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['id'] not in train_ids
            train_ids[row['id']] = stage
    assert len(train_ids) == 1313 and sum(counts.values()) == 21454
    arrivals = spec['TRAIN_memory_arrivals']
    assert all(i in train_ids and train_ids[i] <= stage for i, stage in arrivals.items())
    decisions_path = run/'decisions.json'
    assert sha(decisions_path) == proof['artifact_sha256']['decisions.json']
    decisions = read(decisions_path)['events']
    assert [e['stage'] for e in decisions] == list(range(1, 20))
    prefixes, teacher_calls, shared_singleton_QA = [], 0, 0
    for stage in range(1, 20):
        rel = f'pool/stage_{stage:02d}.pt'
        path = run/rel
        assert sha(path) == proof['artifact_sha256'][rel]
        state = torch.load(path, map_location='cpu', weights_only=True)
        keys, ids = validate_prefix(stage, state, arrivals)
        assert keys == decisions[stage-1]['selected_states']
        assert ids == decisions[stage-1]['arrived_TRAIN_ids']
        assert all(train_ids[i] <= stage for i in ids)
        singleton = len(keys) == 1
        calls = 0 if singleton else len(keys)*len(ids)
        teacher_calls += calls
        # Final all-task predictions supply R_19,19 too; no repeated acquired cell.
        cell_tasks = [stage, stage+1] if stage < 19 else list(range(1, 20))
        if singleton and stage < 19:
            shared_singleton_QA += sum(counts[t] for t in cell_tasks)
        prefixes.append(dict(stage=stage, pool_states=keys, memory_TRAIN_ids=ids,
            source_pool_path=str(path.relative_to(ROOT)), source_pool_sha256=sha(path),
            singleton_exact_function_share_possible=singleton,
            extra_teacher_generations=calls, metric_steps=0 if singleton else 40,
            metric_parameters=0 if singleton else 1792,
            evaluation_native_tasks=cell_tasks,
            evaluation_QA=sum(counts[t] for t in cell_tasks),
            final_head_allowed_only_at_stage19=True))
    final_path = run/'final_state.pt'
    assert sha(final_path) == proof['artifact_sha256']['final_state.pt']
    final = torch.load(final_path, map_location='cpu', weights_only=True)
    stage19 = torch.load(run/'pool/stage_19.pt', map_location='cpu', weights_only=True)
    assert exact(final, stage19)
    current = sum(counts[t] for t in range(1, 19))
    following = sum(counts[t] for t in range(2, 20))
    final_QA = sum(counts.values())
    baseline_initial_QA = final_QA
    standard_two_view_QA = baseline_initial_QA+2*(current+following+final_QA)
    # Singleton sharing requires explicit same-function/source/renderer/scoring proof;
    # this CPU plan is not a substitute for those GPU gates or completed predictions.
    possible_shared_QA = standard_two_view_QA-shared_singleton_QA
    result = dict(CPU_only=True, planner_sha256=sha(Path(__file__)), mechanics_checks=checks,
        source_complete_sha256=sha(SOURCE/'build_complete.json'),
        manifest_sha256=sha(manifest), TEST_file_sha256=test_file_hashes,
        actual_prefix_states_audited=19, all_prefix_TRAIN_and_pool_chronology_passed=True,
        actual_stage19_pool_exact_final=True, TEST_answers_or_predictions_parsed=False,
        new_GPU_or_model_training=False, queue_admitted=False,
        required_views=['cache_AnswerF1', 'cache_AnswerF1_metric'], prefixes=prefixes,
        budgets=dict(prefix_teacher_generations=teacher_calls,
            already_required_stage19_generations=prefixes[-1]['extra_teacher_generations'],
            additional_earlier_prefix_generations=teacher_calls-prefixes[-1]['extra_teacher_generations'],
            current_1_through18_QA=current, next_2_through19_QA=following,
            final_QA_per_view=final_QA, shared_initial_QA=baseline_initial_QA,
            sparse_74_cells_per_view_without_forgetting=True,
            standard_both_view_required_prediction_QA=standard_two_view_QA,
            singleton_share_possible_prediction_QA=shared_singleton_QA,
            possible_prediction_QA_after_singleton_sharing=possible_shared_QA),
        pending_required_gates=[
            'Actual prefix TRAIN feedback and 40-step heads using only available prefix memories.',
            'Actual stage19 feedback/theta/routes/predictions exactly reload the current build.',
            'Official native image forward and each sparse full TEST identity/gold/F1 proof.',
            'Base-only S0 shared initial, 18 acquired, 18 next, and 19 final cells per view.',
            'Shared singleton predictions require explicit exact compatible-function proofs.',
            'Every teacher/feature/head/checkpoint/CPU-GPU resource and extra compute charged.'],
        limits=['Plan and chronology audit are not GPU outcomes or an FWT/BWT result.',
                'Final TEST predictions cannot choose representation, head fit, threshold or sample IDs.',
                'No original job/confirmation is cancelled or reordered by this standalone plan.',
                'NativeAnyRes/BF16 cannot fill formal V2 main-table results.'])
    assert not torch.cuda.is_initialized()
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'CPU_PLAN.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(prefixes=19, teacher_generations=teacher_calls,
        additional_earlier_TRAIN_generations=result['budgets']['additional_earlier_prefix_generations'],
        budgets=result['budgets'], GPU_started=False, admitted=False)))


if __name__ == '__main__':
    main()
