"""One 2400-second original GPU preflight + fresh-process reload; stdlib only.

Called by dispatch_dynamic, never an independent GPU allocator. No import of a
runner/model, no CPU model, no scoring, no formal training or checkpoint reuse.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

CONTROLLER = Path(__file__).resolve().parent
HERE = CONTROLLER.parent / 'two_backbone_pct10_forever_dynamic_20261025'
BUDGET = 2400
MANIFEST = '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def identity(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), startticks=fields[19], uid=path.stat().st_uid,
                    pgid=int(fields[2]))
    except (OSError, ValueError):
        return None


def run_phase(name, command, run, env, deadline):
    remaining = deadline - time.time()
    assert remaining > 0, 'Original shared gate deadline exhausted'
    with (run / 'supervisor.log').open('a') as log:
        child = subprocess.Popen(command, env=env, cwd=env['LLMCL_ROOT'],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        who = identity(child.pid)
        assert who and who['uid'] == os.getuid() and who['pgid'] == child.pid
        write(run / ('GATE_' + name + '_STARTED.json'),
              dict(unix=time.time(), identity=who, command=command, deadline_unix=deadline))
        try:
            code = child.wait(timeout=max(.001, deadline - time.time()))
        except subprocess.TimeoutExpired:
            # Only this Popen child's still-matching private process group.
            if identity(child.pid) == who:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if identity(child.pid) == who:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            write(run / ('GATE_' + name + '_EXIT.json'), dict(
                unix=time.time(), identity=who, returncode=child.returncode,
                timeout=True, identity_absent=identity(child.pid) is None))
            raise RuntimeError('Gate deadline exceeded in ' + name)
    write(run / ('GATE_' + name + '_EXIT.json'), dict(unix=time.time(),
        identity=who, returncode=code, timeout=False,
        identity_absent=identity(child.pid) is None))
    assert code == 0 and identity(child.pid) is None, name + ' failed'


def verify(run, sources, backbone):
    config = read(run / 'config.json')
    assert config['backbone'] == backbone and config['method'] == 'FOREVER'
    assert config['preflight_only'] and config['manifest_sha256'] == MANIFEST
    assert config['execution_source_sha256'] == sources
    assert config['model_dtype'] == config['autocast_dtype'] == 'torch.float16'
    public = read(run / 'gpu_preflight.json')
    reload = read(run / 'checkpoint_reload_proof.json')
    for value in (public, reload):
        assert value['passed'] and value['manifest_sha256'] == MANIFEST
        assert value['execution_source_sha256'] == sources
    assert public['task_ids'] == list(range(1, 20)) and len(public['cases']) == 19
    assert all(math.isfinite(x['loss']) and math.isfinite(x['gradient_l2_preclip'])
               for x in public['cases'])
    if backbone == 'onevision':
        assert len(public['native_forward_cases']) == 19
    assert reload['fresh_process'] and reload['finite_backward']
    view = read(run / 'FOREVER_PREFLIGHT_VIEW.json')
    ids = view['task_ids']
    assert len(ids) == 19 and all(len(x) == 1 for x in ids[:12])
    assert len(ids[12]) == 385 and not any(ids[13:])
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == view['view_ids_sha256']
    discard = read(run / 'FOREVER_COMMON_PREFLIGHT_DISCARD.json')
    assert discard['discarded_scheduler_state_only'] and discard['not_a_real_S1_prefix']
    assert discard['drift_observed_current_updates'] == 0
    assert [x['source_id'] for x in public['cases']] == [
        x['source_id'] for x in discard['inherited_common_diagnostic_cases_by_source_contract']]
    gate = read(run / 'FOREVER_REPLAY_GPU_GATE.json')
    assert gate['extra_current_QA'] == 385 and gate['extra_current_updates'] == 25
    assert gate['extra_current_TRAIN_ids'] == ids[12]
    assert len(gate['all_current_gradient_l2']) == len(gate['learning_rate_used']) == 25
    assert all(math.isfinite(v) for v in gate['all_current_gradient_l2'])
    assert all(any(v > 0 for v in group) for group in gate['learning_rate_used'])
    assert gate['actual_history_QA'] == 12 and gate['actual_memory_QA'] > 0
    assert gate['parameter_before_sha256'] != gate['parameter_after_sha256']
    assert gate['dynamic_branch_GPU_validated']
    assert not gate['longer_than_24_branch_GPU_validation_still_required']
    assert all(gate[k] == 0 for k in ('new_TEST_QA', 'new_generations', 'new_F1_scores'))
    events = gate['ordered_execution']
    current = [x['current_step'] for x in events if x['event'] == 'current_update_complete']
    assert current == list(range(1, 26))
    dynamic = [i for i, x in enumerate(events)
               if x['event'] == 'dynamic_replay_complete' and x['current_step'] == 24]
    last = next(i for i, x in enumerate(events)
                if x == dict(event='current_update_complete', current_step=25))
    assert dynamic and max(dynamic) < last < len(events) - 2
    assert events[-2:] == [dict(event='final_replay_start', current_step=25),
                          dict(event='final_replay_complete', current_step=25)]
    for check in gate['dynamic_outer_state_checks']:
        assert check['external_optimizer_scheduler_scaler_objects_and_state_exact']
        assert check['parameter_objects_storage_requires_grad_exact'] and check['checkpoint_flags_exact']
    schedule = gate['actual_scheduler']
    assert schedule['completed_current_updates'] == schedule['main_updates'] == 25
    assert schedule['final_complete'] and schedule['dynamic_replay_triggers'] >= 1
    assert schedule['next_trigger_idx'] == schedule['dynamic_replay_triggers']
    reports = gate['actual_dynamic_replays'] + [gate['actual_replay']]
    assert len(reports) == schedule['dynamic_replay_triggers'] + 1
    assert len(gate['dynamic_outer_state_checks']) == len(reports) - 1
    for report in reports:
        assert report['status'] == 'ACTUAL_REPLAY_COMPLETE'
        assert report['requested_epochs'] == 2 and report['positive_LR_updates'] > 0
        assert report['model_time_unchanged_by_replay'] and not report['CPU_control_not_formal']
    assert gate['replay_TRAIN_QA'] == sum(x['actual_QA_exposures'] for x in reports)
    assert gate['replay_unwrap_extra_TRAIN_NLL_forwards'] == 2 * len(reports)
    stage = run / 'method_state/stage_13'
    assert read(stage / 'SCHEDULER_COMPLETE.json') == schedule
    for name in [f'dynamic_{i:02d}' for i in range(len(reports) - 1)] + ['final']:
        folder = stage / name / 'actual_author_replay'
        unwrap = read(folder / 'AMP_UNWRAP_EXACT.json')
        assert unwrap['common_explicit_autocast_NLL_exact']
        assert unwrap['model_identity_storage_versions_exact']
        assert unwrap['extra_TRAIN_NLL_forwards'] == 2 and unwrap['extra_optimizer_updates'] == 0
    saved = read(run / 'checkpoints/latest.json')
    assert saved['full_architecture_and_runtime'] and saved['stage'] == 1
    assert saved['manifest_sha256'] == MANIFEST and saved['method'] == 'FOREVER'
    assert not list((run / 'checkpoints').glob('*.pt')), 'Original probe must discard diagnostic weights'
    assert not (run / 'evaluations').exists(), 'No TEST evaluation or formal score in gate'
    proofs = ['config.json', 'gpu_preflight.json', 'checkpoint_reload_proof.json',
              'FOREVER_PREFLIGHT_VIEW.json', 'FOREVER_COMMON_PREFLIGHT_DISCARD.json',
              'FOREVER_REPLAY_GPU_GATE.json', 'checkpoints/latest.json',
              'method_state/stage_13/SCHEDULER_COMPLETE.json']
    return dict(common_diagnostic_QA=19, current_QA=385, current_updates=25,
        actual_dynamic_replays=len(reports) - 1, replay=reports,
        replay_TRAIN_QA=gate['replay_TRAIN_QA'], replay_unwrap_extra_TRAIN_NLL_forwards=2 * len(reports),
        original_tail_probe_task=19, reload_finite_backward=True,
        checkpoint_saved_and_original_probe_deleted=saved,
        diagnostic_weights_never_formal_start=True,
        proof_files={p: sha(run / p) for p in proofs},
        raw_gradient_files_bytes=sum(p.stat().st_size for p in run.rglob('*raw_gradient.pt')),
        total_retained_run_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),
        complete_gate_GPU_peak_available=False,
        resource_limit='Original replay resets peaks; only its recorded per-scope peaks are available.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', type=Path, required=True)
    args = ap.parse_args()
    root = Path(os.environ['LLMCL_ROOT']).resolve()
    admission = read(root / 'summary/two_backbone_pct10_20261025/FOREVER_DYNAMIC_DISPATCH_READY.json')
    assert sha(Path(__file__)) == admission['supervisor_sources'][Path(__file__).name]
    request = read(args.request)
    assert request['phase'] == 'preflight' and request['job']['method'] == 'FOREVER'
    assert request['manifest_sha256'] == MANIFEST
    backbone = request['job']['backbone']
    assert backbone in ('onevision', 'qwen')
    assert HERE == root / 'exp/two_backbone_pct10_forever_dynamic_20261025'
    sources = {p.name: sha(p) for p in HERE.glob('*.py')}
    assert sources == request['sources']
    build = read(HERE / 'BUILD_RECEIPT.json')
    assert all(sources[name] == value for name, value in build['destination_py_sha256'].items())
    assert all(sha(root / name) == value['sha256'] for name, value in build['author_helpers'].items())
    run = root / request['run_dir']
    assert run.resolve().is_relative_to(HERE / 'runs') and run.is_dir()
    started = time.time()
    deadline = started + BUDGET
    write(run / 'GATE_STARTED.json', dict(unix=started, deadline_unix=deadline,
        budget_seconds=BUDGET, identity=identity(os.getpid()), GPU=request['gpu'],
        backbone=backbone, request_sha256=sha(args.request), sources=sources))
    env = dict(os.environ, LLMCL_ROOT=str(root),
        LLMCL_HOST=request['machine'] + '_pct10_' + backbone, LLMCL_BACKBONE=backbone,
        CUDA_VISIBLE_DEVICES=str(request['gpu']), PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    python = root / '.vision-env/bin/python'
    try:
        run_phase('PREFLIGHT', [str(python), '-u', str(HERE / 'run.py'),
            '--method', 'FOREVER', '--condition', 'pct10', '--run-dir', str(run),
            '--preflight-only'], run, env, deadline)
        run_phase('RELOAD', [str(python), '-u', str(HERE / 'checkpoint_probe.py'),
            '--run-dir', str(run)], run, env, deadline)
        result = verify(run, sources, backbone)
        assert time.time() <= deadline, 'Gate verification belongs to shared budget'
        write(run / 'GATE_VERIFIED.json', dict(passed=True, unix=time.time(),
            seconds=time.time() - started, deadline_unix=deadline,
            backbone=backbone, GPU=request['gpu'], manifest_sha256=MANIFEST,
            sources=sources, new_TEST_generations=0, new_F1_scores=0,
            **result))
    except BaseException as error:
        write(run / 'GATE_FAILURE.json', dict(unix=time.time(), seconds=time.time()-started,
            deadline_unix=deadline, error=repr(error), no_retry=True))
        raise


if __name__ == '__main__':
    main()
