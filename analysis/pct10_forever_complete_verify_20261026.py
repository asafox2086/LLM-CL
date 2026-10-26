"""One completed pct10 FOREVER run: original F1 validation + replay evidence.

Stdlib only; no model/Torch import, deserialization, launch or Markdown writer.
Uses an isolated, SHA-pinned native verifier with exactly two provenance-path
substitutions and one source-map substitution. Its score/schedule bodies remain
unchanged. Separate per-run O_EXCL cache; never joins the native14 observer.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from types import ModuleType, SimpleNamespace

BASE_SHA = '2b2a3202fa049c911324081aacb31d0e7c7508b1020a2727893f1f85ab15c376'
READY_SHA = 'af6c26f756a51adfd1e909ef90e10ab5dd0be743dc73b469d37e0a19d818381f'
LEGACY_SHA = '2d9d5be87d81df80eab6ddf0095ce399c9a73d9c8c29909a5022d2364fdeeed8'
REPLAY_SHA = '8024931496a032ea666e44376d79614d7c223b17c73ff11a8074494f87cd3683'
FRAME_REL = 'exp/two_backbone_pct10_forever_dynamic_20261025'
READY_REL = 'summary/two_backbone_pct10_20261025/FOREVER_DYNAMIC_DISPATCH_READY.json'
OLD_TRANSPORT = 'summary/two_backbone_pct10_20261025/OLD_TRANSPORT_RECEIPT.json'
DAYS = (1, 2, 4, 7, 15, 30, 60, 90, 120)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def inherited(root):
    """Private metadata verifier; no source file or live observer is modified."""
    path = root / 'analysis/pct10_verified_baseline_report_20261025.py'
    assert sha(path) == BASE_SHA
    tree = ast.parse(path.read_bytes())

    class Provenance(ast.NodeTransformer):
        paths = maps = 0

        def visit_Constant(self, node):
            if node.value == OLD_TRANSPORT:
                self.paths += 1
                return ast.copy_location(ast.Constant(READY_REL), node)
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'frozen_sources':
                assert isinstance(node.value, ast.DictComp)
                self.maps += 1
                node.value = ast.parse("transport['dynamic_sources']", mode='eval').body
            return self.generic_visit(node)

    patch = Provenance()
    tree = patch.visit(tree)
    assert (patch.paths, patch.maps) == (2, 1), 'Pinned provenance boundary changed'
    module = ModuleType('isolated_pct10_forever_complete')
    module.__file__ = str(path)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    assert module.ROOT == root and module.SOURCE_SHA == BASE_SHA
    module.FRAME, module.METHODS, module.TRANSPORT_SHA = root / FRAME_REL, ['FOREVER'], READY_SHA
    return module


class Artifacts:
    def __init__(self, root):
        self.root, self.hashes, self.stamps = root, {}, {}

    def checked(self, path, expected=None):
        path = Path(path)
        value = sha(path)
        assert expected is None or value == expected, str(path)
        key = str(path.relative_to(self.root))
        assert key not in self.hashes or self.hashes[key] == value
        self.hashes[key] = value
        st = path.stat(); self.stamps[path] = (st.st_size, st.st_mtime_ns)
        return path

    def json(self, path, expected=None):
        return read(self.checked(path, expected))

    def unchanged(self):
        assert all((p.stat().st_size, p.stat().st_mtime_ns) == stamp
                   for p, stamp in self.stamps.items()), 'Extension artifacts changed'


def selector(root, files):
    # Reuse only these pure metadata definitions. The helper's torch import,
    # kernels, parameter operations and model class are never executed.
    path = files.checked(root / 'analysis/forever_multimodal_core_20261019.py', LEGACY_SHA)
    wanted = {'source', 'DatasetBridge', 'select_history'}
    nodes = [n for n in ast.parse(path.read_bytes()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    namespace = dict(ROOT=root, Path=Path, ast=ast, hashlib=hashlib, json=json,
                     os=os, Counter=Counter, random=random, SimpleNamespace=SimpleNamespace,
                     REVISION='bbeb549a21e59d9490f20a2b02099ee766a67f48')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['select_history']


def replay(folder, request, ids, files, stage):
    assert request['memory_QA'] == len(ids) and request['author_memory_epochs'] == 2
    assert request['author_replay_warmup_steps'] == 5
    ratio = request['mu'] / (request['mu0'] + 1e-12)
    beta = min(3.0, max(0.5, 1.0 + (ratio - 1.0)))
    assert request['beta'] == beta and request['regularizer_coefficient'] == .001 * beta
    assert request['theta_star_scope'] == 'fixed task-entry trainable parameters'
    closed = files.json(folder / 'REPLAY_CLOSED.json')
    assert closed['request'] == request and closed['source_complete']
    after = closed['scheduler_after']
    assert after['completed_current_updates'] == request['current_step']
    assert all(after[x] == request[x] for x in ('tau', 'mu', 'mu0'))
    assert after['final_complete'] == (request['kind'] == 'final')
    if request['kind'] == 'dynamic':
        assert after['next_trigger_idx'] == request['trigger_index'] + 1
    actual = folder / 'actual_author_replay'
    args = files.json(actual / 'REQUESTED_ARGUMENTS.json')
    expected = dict(num_train_epochs=2, warmup_steps=5, gradient_accumulation_steps=16,
        per_device_train_batch_size=1, learning_rate=1e-4, lr_scheduler_type='linear',
        optim='adamw_torch', weight_decay=0., adam_beta1=.9, adam_beta2=.999,
        adam_epsilon=1e-8, max_grad_norm=1., fp16=True, bf16=False, use_cpu=False,
        CPU_control_not_formal=False, stage=stage, beta=request['beta'], memory_QA=len(ids))
    assert all(args[k] == v for k, v in expected.items())
    assert files.json(actual / 'SHARED_FP16_INITIAL_SCALE.json')['actual_before_first_backward'] == 1.
    done = files.json(actual / 'REPLAY_COMPLETE.json')
    assert done == closed['actual'] and done['status'] == 'ACTUAL_REPLAY_COMPLETE'
    assert not done['CPU_control_not_formal'] and done['model_time_unchanged_by_replay']
    assert done['stage'] == stage and done['beta'] == request['beta']
    assert done['requested_epochs'] == 2
    clean = files.json(actual / 'AMP_UNWRAP_EXACT.json')
    assert not clean['keep_fp32_wrapper'] and clean['model_identity_storage_versions_exact']
    assert clean['common_explicit_autocast_NLL_exact'] and clean['NLL_before'] == clean['NLL_after']
    assert math.isfinite(clean['NLL_before']) and clean['TRAIN_probe_id'] == ids[0]
    assert clean['extra_TRAIN_NLL_forwards'] == 2 and clean['extra_optimizer_updates'] == 0
    trace = [json.loads(x) for x in files.checked(actual / 'trace/updates.jsonl').read_text().splitlines()]
    assert len(trace) == done['actual_optimizer_updates'] > 1
    exposures, positive, zero, previous = 0, 0, 0, None
    selected = set(ids)
    for step, row in enumerate(trace, 1):
        assert row['step'] == step and math.isfinite(row['raw_gradient_l2']) and row['raw_gradient_l2'] > 0
        assert math.isfinite(row['parameter_delta_l2'])
        before, after = row['parameter_before_sha256'], row['parameter_after_sha256']
        assert previous is None or previous == before
        previous = after
        lr = row['learning_rate_used']
        assert lr and all(math.isfinite(x) and x >= 0 for x in lr)
        moving = any(x > 0 for x in lr)
        assert row['author_zero_warmup_step'] == (not moving)
        if moving:
            assert before != after and row['parameter_delta_l2'] > 0
            positive += 1
        else:
            assert step == 1 and before == after and row['parameter_delta_l2'] == 0
            zero += 1
        name = row['raw_gradient_file']; assert Path(name).name == name
        files.checked(actual / 'trace' / name, row['raw_gradient_sha256'])  # Stream, never load tensor.
        for micro in row['micro_losses']:
            assert micro['TRAIN_id'] in selected
            assert all(math.isfinite(micro[k]) for k in
                       ('base_loss', 'unscaled_regularizer', 'regularizer_coefficient', 'total_loss'))
            assert micro['unscaled_regularizer'] >= 0
            assert math.isclose(micro['regularizer_coefficient'], .001 * request['beta'], abs_tol=1e-15, rel_tol=0)
            exposures += 1
        assert row['cumulative_QA'] == exposures
    assert exposures == done['actual_QA_exposures']
    assert positive == done['positive_LR_updates'] > 0
    assert zero == done['author_zero_warmup_updates'] == 1
    resource = files.json(folder / 'resources/memory_summary.json')
    assert resource['before']['pid'] == resource['after']['pid']
    return dict(kind=request['kind'], stage=stage, current_step=request['current_step'],
        memory_QA=len(ids), actual_QA=exposures, actual_updates=len(trace), positive_LR=positive,
        author_zero_warmup=zero, unwrap_extra_TRAIN_NLL=2, seconds=done['seconds'],
        requested_epochs=2, HF_final_epoch=done['HF_final_epoch'], resource=resource)


def extension(root, run, host, base, files):
    ready = files.json(root / READY_REL, READY_SHA)
    frame = root / FRAME_REL
    build = files.json(frame / 'BUILD_RECEIPT.json', ready['build_receipt_sha256'])
    assert build['destination_py_sha256'] == ready['dynamic_sources']
    config = files.json(run / 'config.json')
    assert config['execution_source_sha256'] == ready['dynamic_sources'] and config['rolling_checkpoint']
    assert config['model_dtype'] == config['autocast_dtype'] == 'torch.float16'
    files.checked(frame / 'execution_policy.json', ready['dynamic_policy_sha256'])
    assert config['external_helper_sha256'] == {
        'analysis/forever_multimodal_core_20261019.py':LEGACY_SHA,
        'analysis/forever_multimodal_replay_v2_20261019.py':REPLAY_SHA}
    for name, value in config['external_helper_sha256'].items():
        files.checked(root / name, value); files.checked(run / 'runner_source' / name, value)
    for name, value in ready['supervisor_sources'].items():
        files.checked(root / 'exp/two_backbone_pct10_20261025' / name, value)
    queue = read(root / 'summary/two_backbone_pct10_20261025' /
                 ('queue_old.json' if host == 'old2080' else 'queue_new.json'))
    job = queue['jobs'][config['backbone'] + ':FOREVER']
    gate_run = root / job['preflight_run']
    assert gate_run.resolve().is_relative_to(frame / 'runs') and gate_run != run
    gate = files.json(gate_run / 'GATE_VERIFIED.json')
    assert gate['passed'] and gate['sources'] == ready['dynamic_sources']
    assert gate['backbone'] == config['backbone'] and gate['manifest_sha256'] == base.MANIFEST_SHA
    assert gate['seconds'] <= 2400 and gate['unix'] <= gate['deadline_unix']
    for phase in ('PREFLIGHT', 'RELOAD'):
        ex = files.json(gate_run / f'GATE_{phase}_EXIT.json')
        assert ex['returncode'] == 0 and not ex['timeout'] and ex['identity_absent']
    for name, value in gate['proof_files'].items():
        files.checked(gate_run / name, value)
    manifest = files.json(base.MANIFEST, base.MANIFEST_SHA)
    tasks = []
    for task in manifest['tasks']:
        f = task['files']['train']
        rows = [json.loads(x) for x in files.checked(root / f['path'], f['sha256']).read_text().splitlines()]
        assert len(rows) == f['count']
        tasks.append(rows)
    choose = selector(root, files)
    author = root / 'code/FOREVER'
    assert files.json(author / 'source_manifest.json')['revision'] == 'bbeb549a21e59d9490f20a2b02099ee766a67f48'
    official = {x['path']: x['sha256'] for x in files.json(author / 'files_sha256.json')}
    for name in ('utils/load_data.py', 'src/finetune_cl_our_qwen.py'):
        files.checked(author / 'source' / name, official[name])
    all_reports, stages = [], []
    for stage, rows in enumerate(tasks, 1):
        n = math.ceil(len(rows) / 16); k = min(24, n)
        folder = run / 'method_state' / f'stage_{stage:02d}'
        cal = files.json(folder / 'calibration.json')
        final = files.json(folder / 'SCHEDULER_COMPLETE.json')
        assert final['stage'] == stage and final['scheduler_revision'] == 'pct10-author-triggers-common-current-loop-v1'
        assert final['source_sha256'] == cal['source_sha256'] == official['src/finetune_cl_our_qwen.py']
        assert final['main_updates'] == final['completed_current_updates'] == n
        assert final['calibrated'] and final['final_complete']
        assert final['calibration_steps_per_day'] == 24 and final['actual_calibration_steps'] == k
        assert all(math.isfinite(final[x]) and final[x] >= 0 for x in ('tau', 'mu', 'mu0', 'calibration_tau'))
        assert final['model_day'] == max(final['calibration_tau'], 1e-8)
        assert final['mu0'] == final['calibration_tau'] / k
        assert final['thresholds'] == [d * final['model_day'] for d in DAYS]
        assert final['author_memory_epochs'] == 2 and final['author_replay_warmup_steps'] == 5
        assert not final['author_HF_phase_restarts_reproduced']
        assert cal['completed_current_updates'] == n and not cal['final_complete']
        assert not (folder / 'REPLAY_REQUEST.json').exists()
        memory, selection = choose(tasks, stage)
        ids = [r['id'] for r in memory]
        assert final['memory_QA'] == cal['memory_QA'] == len(ids)
        if stage == 1:
            assert not ids and not final['final_replay_required'] and not final['completed_replays']
            assert files.json(folder / 'REPLAY_SKIPPED.json')['reason'] == 'no historical TRAIN'
        else:
            assert files.json(folder / 'memory_selection.json') == selection
            assert selection['history_QA'] == sum(map(len, tasks[:stage-1])) and ids
            assert len(ids) == len(set(ids)) and set(ids) <= {r['id'] for tt in tasks[:stage-1] for r in tt}
        events = [files.json(folder / f'main_update_{i:03d}.json') for i in range(1, n+1)]
        completed, index, previous_tau = final['completed_replays'], 0, 0.
        for i, event in enumerate(events, 1):
            assert event['stage'] == stage and event['current_step'] == i and event['main_updates'] == n
            assert math.isfinite(event['tau']) and event['tau'] >= previous_tau
            previous_tau = event['tau']
            assert event['calibrated'] == (i >= k) and event['next_trigger_idx'] == index
            assert event['model_day'] == (final['model_day'] if i >= k else None)
            if i >= k:
                assert event['mu0'] == final['mu0'] and math.isfinite(event['mu'])
            due = bool(i >= k and ids and i < n and index < len(DAYS)
                       and event['tau'] >= final['thresholds'][index])
            assert event['dynamic_replay_due'] == due
            while due:
                expected = completed[index]
                request = files.json(folder / f'dynamic_{index:02d}/REPLAY_REQUEST.json')
                assert request['kind'] == 'dynamic' and request['trigger_index'] == index
                assert request['sequence'] == index + 1
                assert request['current_step'] == i and request['trigger_threshold'] == final['thresholds'][index]
                assert all(request[x] == event[x] for x in ('tau', 'mu', 'mu0'))
                assert expected == dict(request, model_time_unchanged_by_replay=True, drift_reference_updated=True)
                all_reports.append(replay(folder / f'dynamic_{index:02d}', request, ids, files, stage))
                index += 1
                due = index < len(DAYS) and event['tau'] >= final['thresholds'][index]
        assert index == final['dynamic_replay_triggers'] == final['next_trigger_idx']
        assert cal['completed_replays'] == completed[:index]
        if ids:
            request = files.json(folder / 'final/REPLAY_REQUEST.json')
            assert request['kind'] == 'final' and request['trigger_index'] is None
            assert request['sequence'] == index + 1
            assert request['current_step'] == n and request['tau'] == final['tau']
            assert completed[-1] == dict(request, model_time_unchanged_by_replay=True, drift_reference_updated=True)
            assert len(completed) == index + 1 and final['final_replay_required']
            all_reports.append(replay(folder / 'final', request, ids, files, stage))
        expected_dirs = {f'dynamic_{i:02d}' for i in range(index)} | ({'final'} if ids else set())
        actual_dirs = {p.name for p in folder.iterdir() if p.is_dir() and (p.name.startswith('dynamic_') or p.name == 'final')}
        assert actual_dirs == expected_dirs
        stages.append(dict(stage=stage, current_QA=len(rows), current_updates=n,
            memory_QA=len(ids), dynamic_replays=index, final_replays=int(bool(ids))))
    latest = files.json(run / 'checkpoints/latest.json')
    assert files.json(run / 'checkpoints/committed_latest.json') == latest and latest['stage'] == 19
    assert sorted(p.name for p in (run / 'checkpoints').glob('*.pt')) == [Path(latest['path']).name]
    for stage in range(1,20):
        tx = files.json(run / f'rolling_checkpoint_commits/stage_{stage:02d}.json')
        assert tx['stage'] == tx['committed_checkpoint']['stage'] == stage
        files.checked(run / f'logs/stage_{stage:02d}.events.jsonl')
    resource = files.json(run / 'memory_summary.json')
    assert resource['no_cross_process_memory_subtraction']
    if resource['resume_segment']:
        assert resource['full_run_growth_bytes'] is None and not resource['full_run_growth_available']
    return dict(stage_schedule=stages, replay=all_reports,
        extra_replay_QA_exposures=sum(x['actual_QA'] for x in all_reports),
        extra_replay_optimizer_updates=sum(x['actual_updates'] for x in all_reports),
        extra_replay_positive_LR_updates=sum(x['positive_LR'] for x in all_reports),
        author_replay_zero_warmup_updates=sum(x['author_zero_warmup'] for x in all_reports),
        unwrap_extra_TRAIN_NLL_forwards=2 * len(all_reports),
        original_process_resource_summary=resource,
        complete_training_GPU_peak_not_certified='Replay resets allocator peaks; original per-scope evidence only.',
        final_full_checkpoint_retained_on_origin=True, NEW_durable_archive_not_certified_here=True,
        gate_run=str(gate_run.relative_to(root)), gate_verified_sha256=sha(gate_run / 'GATE_VERIFIED.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='ROOT-relative completed dynamic FOREVER training run')
    parser.add_argument('--host', choices=['old2080', 'new3090'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--write', action='store_true', help='Explicitly permit the unique original-F1 validation')
    args = parser.parse_args()
    assert args.write and os.environ.get('LLMCL_ROOT')
    root = Path(os.environ['LLMCL_ROOT']).resolve()
    assert root.is_absolute() and root.is_dir() and args.output.is_absolute()
    relative = Path(args.run); assert not relative.is_absolute() and '..' not in relative.parts
    run = root / relative
    base = inherited(root)
    candidate = base.candidate(args.run, args.host)  # Requires final19 and real queue training exit0.
    own = sha(Path(__file__))
    report = root / 'summary/two_backbone_pct10_20261025/completion_observer_forever/completed'
    report.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((args.host + ':' + args.run).encode()).hexdigest()[:20]
    proof_path, start_path, complete_path = [report / (key + suffix) for suffix in ('.json', '.VERIFY_STARTED.json', '.COMPLETE.json')]
    if proof_path.exists() or complete_path.exists():
        receipt = read(complete_path); proof = read(proof_path)
        assert receipt['proof_sha256'] == sha(proof_path) and proof['candidate'] == candidate
        assert proof['verifier_sha256'] == own and receipt['status'] == 'COMPLETE_VERIFIED'
    else:
        write(start_path, dict(unix=time.time(), pid=os.getpid(), candidate=candidate,
            verifier_sha256=own, planned_scored_QA=80316, no_automatic_retry=True))
        started = time.monotonic()
        try:
            files = Artifacts(root)
            extra = extension(root, run, args.host, base, files)
            shared = base._verify_uncached(args.run, args.host)  # Only scoring call; no old run is selected.
            shared['FOREVER_in_scope'] = True
            files.unchanged()
            assert base.candidate(args.run, args.host) == candidate and sha(Path(__file__)) == own
            proof = dict(status='PCT10_FOREVER_FORMAL_COMPLETE_VERIFIED', candidate=candidate,
                verifier_sha256=own, shared_pct10_proof=shared, **extra,
                extension_source_and_artifact_sha256=files.hashes,
                base_provenance_AST_adaptation=dict(base_sha256=BASE_SHA,
                    old_transport_path_replacements=2, frozen_source_map_replacements=1,
                    replacement_READY_sha256=READY_SHA, scoring_and_schedule_body_changed=False),
                CPU_only=True, models_started=0, new_generations=0, scored_QA=80316,
                native14_observer_unmodified=True, Markdown_tables_written=False)
            write(proof_path, proof)
            write(complete_path, dict(status='COMPLETE_VERIFIED', candidate=candidate,
                proof_sha256=sha(proof_path), unix=time.time(), seconds=time.monotonic()-started))
        except BaseException:
            write(report / (key + '.FAILURE.json'), dict(unix=time.time(), traceback=traceback.format_exc(),
                no_automatic_retry=True, models_started=0))
            raise
    if args.output.resolve() != proof_path.resolve():
        write(args.output, proof)  # Exclusive copy only; never re-score cached results.


if __name__ == '__main__':
    main()
