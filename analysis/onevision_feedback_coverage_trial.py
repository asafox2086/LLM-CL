"""Same20 TRAIN anchors, native-task coverage control of frozen feedback.

Reuses the exact old experiment class, metric and image/score implementation.
Only feedback IDs/features/teacher scores change; no new backbone updates.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import fcntl
import json
import time

import torch

import onevision_answer_compatibility_trial as previous
import onevision_feedback_coverage_plan as sampling
from onevision_knowledge_intervention import reset_seed, case_complete
from onevision_research_worker import atomic_torch, digest
from onevision_capacity_probe import finish_resources, stratify_full_results
from run import write

ROOT, BASE, core = previous.ROOT, previous.BASE, previous.core
REPORT = BASE/'feedback_coverage_20261018/v1'
FAMILY = ROOT/'exp/onevision_feedback_coverage_20261018/v1'
VIEWS = ['coverage_AnswerF1', 'coverage_AnswerF1_metric']
FeedbackExperiment = previous.FeedbackExperiment
read = previous.read


def plan():
    old_spec, values, source_build = previous.plan()
    assert previous.complete('build', None, old_spec)
    old_proof = read(previous.REPORT/'build_complete.json')
    old_build = ROOT/old_proof['run']
    frozen = read(REPORT/'CPU_PLAN.json')
    assert frozen['actual_TRAIN_QA_unchanged'] == 1313
    data = values[1]
    by_task = {t: [r['id'] for r in data[t, 'train']] for t in range(1, 20)}
    ids = sampling.select20(by_task)
    assert ids == frozen['new_anchor_ids'] and len(ids) == 20
    sources = dict(old_spec['source_sha256'])
    for n in ['onevision_feedback_coverage_trial.py', 'onevision_feedback_coverage_entry.py',
              'onevision_feedback_coverage_plan.py']:
        sources['analysis/'+n] = digest(ROOT/'analysis'/n)
    spec = dict(previous_spec=old_spec, source_sha256=sources,
        protocol_sha256=digest(REPORT/'PROTOCOL.md'), sampling_plan_sha256=digest(REPORT/'CPU_PLAN.json'),
        previous_feedback_complete_sha256=digest(previous.REPORT/'build_complete.json'),
        fixed_pool_sha256=old_spec['fixed_pool_sha256'], TRAIN_ids=ids, views=VIEWS,
        teacher_generation_QA=60, metric_parameters=1792, metric_steps=core.STEPS,
        metric_lr=core.LR, metric_L2=core.L2, metric_log_weight_limit=core.LOG_WEIGHT_LIMIT,
        kernel_temperature=core.KERNEL_TEMPERATURE, expert_temperature=core.EXPERT_TEMPERATURE,
        changed_variable='TRAIN feedback memory allocation only', new_backbone_updates=0,
        original_backbone_updates_charged=93, original_TRAIN_QA=1313,
        both_full21454_regardless_diagnostic=True, native_task_only_TRAIN_sampling=True,
        TEST_selection=False, old_pool_and_prototypes_unchanged=True)
    return spec, values, source_build, old_build


def make_state(source, ids, features):
    assert features.shape == (20, 1792) and len(ids) == len(set(ids)) == 20
    assert torch.isfinite(features).all()
    return dict(stage=19, pool=source['pool'], prototypes=source['prototypes'],
                memory_ids=list(ids), memory_features=features)


def complete(phase, view, spec):
    name = 'build' if phase == 'build' else 'full_'+view
    p = REPORT/(name+'_complete.json')
    if not p.exists():
        return False
    proof = read(p)
    assert proof['complete'] and proof['spec'] == spec
    run = (ROOT/proof['run']).resolve()
    assert run.is_relative_to(FAMILY.resolve())
    for rel, expected in proof['artifact_sha256'].items():
        path = (run/rel).resolve()
        assert path.is_relative_to(run) and digest(path) == expected
    assert read(run/'original_feedback_exact.json')['passed']
    if phase == 'full':
        assert proof['full_QA'] == 21454 and read(run/'coverage_reload_exact.json')['passed']
    return True


def run_trial(phase, view):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0', '1']
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    reset_seed()
    spec, values, source_build, old_build = plan()
    assert spec == read(REPORT/'CPU_WORKER_PLAN.json')['spec'] == read(REPORT/'ADMISSION.json')['spec']
    assert phase == 'build' or view in VIEWS
    lock = (REPORT/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    name = 'build' if phase == 'build' else 'full_'+view
    assert not (REPORT/(name+'_attempt.json')).exists(), 'Preserve failure; no blind retry'
    manifest, data, domains, tests, full, native, native_proof = values
    run = FAMILY/name/time.strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    write(REPORT/(name+'_attempt.json'), dict(spec=spec, run=str(run.relative_to(ROOT)), pid=os.getpid()))
    write(run/'spec.json', spec)
    import onevision_research_device_entry as device
    cards = device.inventory()
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert device.select(cards, os.getuid()) == gpu
    write(run/'physical_device_execution.json', dict(physical_GPU=gpu, admission_inventory=cards,
        actual_CUDA_VISIBLE_DEVICES=str(gpu), single_research_stream=True, max_our_GPU_cards=3))
    x = FeedbackExperiment(manifest, run)
    source = torch.load(source_build/'final_state.pt', map_location='cpu', weights_only=True)
    x.configure(source['pool'][19], finetune_shared=False)
    assert not any(p.requires_grad for p in x.model.parameters())
    generation_ids = spec['previous_spec']['parent_spec']['generation_ids']
    generation = {d: [r for r in rows if r['id'] in set(generation_ids[d])] for d, rows in tests.items()}
    assert sum(map(len, generation.values())) == 155
    x.monitor.snapshot('coverage_process_before', model=x.model)
    if phase == 'build':
        old = torch.load(old_build/'feedback_state.pt', map_location='cpu', weights_only=True)
        x.set_feedback(source, old['scores'], old['theta'], 'cache_AnswerF1_metric')
        x.evaluate('control_original_feedback', tests, generation)
        assert case_complete(run, 'control_original_feedback', tests)
        previous.compare(run, 'control_original_feedback', old_build, 'cache_AnswerF1_metric', tests)
        write(run/'original_feedback_exact.json', dict(passed=True, QA=500, generated=155,
            semantic_fields_and_routes_exact=True, original_build_sha256=spec['previous_feedback_complete_sha256']))
        train = {r['id']: r for t in range(1, 20) for r in data[t, 'train']}
        teacher = [train[i] for i in spec['TRAIN_ids']]
        assert len(train) == 1313 and len(teacher) == 20
        torch.cuda.reset_peak_memory_stats()
        feedback_before = x.monitor.snapshot('TRAIN_feedback_before', model=x.model)
        misses_before = x.metric_feature_misses
        features = torch.stack([x.feature(row).clone() for row in teacher])
        old_features = dict(zip(source['memory_ids'], source['memory_features']))
        overlap = [i for i in spec['TRAIN_ids'] if i in old_features]
        for i, feature in zip(spec['TRAIN_ids'], features):
            if i in old_features:
                assert torch.equal(feature, old_features[i])
        write(run/'feature_identity.json', dict(passed=True, old_overlap_ids=overlap,
            overlapping_frozen_features_exact=True, new_logical_feature_queries=20,
            new_actual_feature_misses=x.metric_feature_misses-misses_before))
        state = make_state(source, spec['TRAIN_ids'], features)
        assert state['pool'] is source['pool'] and state['prototypes'] is source['prototypes']
        atomic_torch(run/'coverage_state.pt', state)
        scores, teacher_sha = [], {}
        for key in sorted(state['pool']):
            x.set_feedback()
            x.configure(state['pool'][key], finetune_shared=False)
            label = 'TRAIN_teacher_state'+str(key)
            selected = {'TRAIN_memory': teacher}
            x.evaluate(label, selected, selected)
            assert case_complete(run, label, selected) and previous.parent.same(x.snapshot(), source['pool'][key])
            path = run/f'evaluations/{label}/TRAIN_memory_image.jsonl'
            teacher_sha[str(key)] = digest(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert [r['id'] for r in rows] == spec['TRAIN_ids']
            from answer_f1 import answer_f1
            for row, original in zip(rows, teacher):
                assert row['references'] == original['answers']
                assert row['scores']['token_f1'] == answer_f1(row['prediction'], row['references'])
            scores.append([r['scores']['token_f1']/100 for r in rows])
        scores = torch.tensor(scores, dtype=torch.float32)
        x.monitor.snapshot('CPU_router_TRAIN_before', model=x.model)
        tick = time.monotonic()
        theta, events, gradients = core.fit(features, scores, [tuple(r['images']) for r in teacher])
        seconds = time.monotonic()-tick
        x.monitor.snapshot('CPU_router_TRAIN_after', model=x.model)
        atomic_torch(run/'router_raw_gradients.pt', dict(gradients=gradients, TRAIN_ids=spec['TRAIN_ids']))
        atomic_torch(run/'feedback_state.pt', dict(theta=theta, scores=scores, TRAIN_ids=spec['TRAIN_ids'],
            source_pool_sha256=spec['fixed_pool_sha256'], coverage_state_sha256=digest(run/'coverage_state.pt'),
            teacher_sha256=teacher_sha, metric_parameters=theta.numel()))
        write(run/'router_TRAIN.json', dict(events=events, CPU_seconds=seconds, TRAIN_ids=spec['TRAIN_ids'],
            theta_l2=float(theta.norm()), actual_nonzero_gradient_steps=sum(bool(torch.count_nonzero(g)) for g in gradients),
            actual_changed_theta_dimensions=int(torch.count_nonzero(theta)), teacher_generation_QA=60,
            teacher_sha256=teacher_sha, CPU_parameter_bytes=theta.numel()*theta.element_size(),
            no_TEST_input_to_fit=True, no_new_backbone_updates=True, source93updates_charged=True,
            image_exclusion_is_training_objective_not_validation=True))
        feedback_after = x.monitor.snapshot('TRAIN_feedback_after', model=x.model)
        write(run/'TRAIN_feedback_window.json', dict(before=feedback_before, after=feedback_after,
            scope='20 feature queries+60 TRAIN teacher generations+40 CPU metric updates; zero model optimizer updates',
            cuda_peak_allocated_bytes=feedback_after['gpu_peak_allocated_bytes'],
            cuda_peak_reserved_bytes=feedback_after['gpu_peak_reserved_bytes']))
        for which in VIEWS:
            x.set_feedback(state, scores, theta if which.endswith('_metric') else torch.zeros_like(theta), which)
            x.evaluate(which, tests, generation)
            assert case_complete(run, which, tests)
    else:
        assert complete('build', None, spec)
        build = ROOT/read(REPORT/'build_complete.json')['run']
        state = torch.load(build/'coverage_state.pt', map_location='cpu', weights_only=True)
        trained = torch.load(build/'feedback_state.pt', map_location='cpu', weights_only=True)
        assert trained['coverage_state_sha256'] == digest(build/'coverage_state.pt')
        assert trained['source_pool_sha256'] == spec['fixed_pool_sha256']
        assert previous.parent.same(state['pool'][19], source['pool'][19])
        theta = trained['theta'] if view.endswith('_metric') else torch.zeros_like(trained['theta'])
        x.set_feedback(state, trained['scores'], theta, view)
        x.evaluate('candidate_reload', tests, generation)
        previous.compare(run, 'candidate_reload', build, view, tests)
        write(run/'coverage_reload_exact.json', dict(passed=True, QA=500, generated=155, original_route_exact=True))
        write(run/'original_feedback_exact.json', dict(passed=True,
            inherited_build_gate_sha256=digest(build/'original_feedback_exact.json')))
        x.evaluate('final_full_input_router', full, full)
        assert case_complete(run, 'final_full_input_router', full)
        stratify_full_results(run, full, {r['id'] for rows in tests.values() for r in rows})
    assert digest(source_build/'final_state.pt') == spec['fixed_pool_sha256']
    for key in source['pool']:
        assert previous.parent.same(state['pool'][key], source['pool'][key])
        assert torch.equal(state['prototypes'][key], source['prototypes'][key])
    assert not any(p.requires_grad for p in x.model.parameters())
    x.monitor.snapshot('coverage_process_after', model=x.model)
    finish_resources(x, run, 'coverage_process_before', 'coverage_process_after',
        'samePID original control+60 TRAIN generation+CPU fit+two diagnostics or full; historical93 model updates charged')
    artifacts = {str(p.relative_to(run)): digest(p) for p in run.rglob('*')
                 if p.is_file() and p.suffix in ['.json', '.jsonl', '.pt']}
    write(REPORT/(name+'_complete.json'), dict(complete=True, spec=spec, phase=phase, view=view,
        run=str(run.relative_to(ROOT)), artifact_sha256=artifacts, full_QA=21454 if phase == 'full' else None,
        new_backbone_updates=0, original_backbone_updates_charged=93, full_FWT=None, full_BWT=None))
    assert complete(phase, view, spec)


def main(check=False, phase=None, view=None):
    REPORT.mkdir(parents=True, exist_ok=True)
    if check:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
        spec, values, source, old = plan()
        train = {r['id']: r for t in range(1, 20) for r in values[1][t, 'train']}
        mask = core.image_mask([tuple(train[i]['images']) for i in spec['TRAIN_ids']])
        assert mask.shape == (20, 20) and mask.any(1).all()
        write(REPORT/'CPU_WORKER_PLAN.json', dict(CPU_only=True, spec=spec,
            actual_TRAIN_mask_checked=True, GPU_started=False))
    else:
        run_trial(phase, view)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--phase', choices=['build', 'full'])
    p.add_argument('--view', choices=VIEWS)
    a = p.parse_args()
    main(a.check, a.phase, a.view)
