"""Build new pct10 checkpoint/pause runners; stdlib metadata and copying only.

Required CLI: --root OUTPUT_ROOT --source-root REVIEWED_SOURCE_ROOT.
Never overwrite a runner, import runner/model code, or launch a process. The
actual frozen dynamic FOREVER is the copy source, not the pct01 adapter.
"""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import json
import time

NATIVE = 'two_backbone_pct10_native_checkpoint_20261027'
FOREVER = 'two_backbone_pct10_forever_checkpoint_20261027'
MANIFEST = 'data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json'
MANIFEST_SHA = '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'
HELPERS = ('analysis/forever_multimodal_core_20261019.py',
           'analysis/forever_multimodal_replay_v2_20261019.py')


def sha(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf8')


def patch_function(source_root):
    """Execute only the old builder's reviewed stdlib patch_run definition."""
    path = source_root/'exp/two_backbone_pct10_20261025/build_runners.py'
    raw = path.read_bytes(); tree = ast.parse(raw)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'patch_run')
    namespace = {'ast': ast}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['patch_run'], dict(path=str(path), sha256=sha(raw))


def function_bytes(raw, name):
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return b''.join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def replace_once(raw, before, after):
    assert raw.count(before) == 1, before.decode('utf8')
    changed = raw.replace(before, after, 1)
    assert changed.count(after) == 1 and changed.replace(after, before, 1) == raw
    return changed


def apply_edits(raw, edits, template):
    result = raw
    for before, after in edits:
        assert after in template, 'The reviewed template must contain the transplanted interface'
        result = replace_once(result, before, after)
    restored = result
    for before, after in reversed(edits):
        restored = replace_once(restored, after, before)
    assert restored == raw
    return result


def transplant_run(raw, template):
    old_initial = b"    if not resume_obj: evaluate(method, processor, manifest['tasks'], data, run, 0)\n"
    start = template.index(b'    if not resume_obj:\n        scores = evaluate(')
    stop = template.index(b'    gc.collect()\n', start)
    edits = [
        (b'from rolling_io import load as load_checkpoint, commit_stage, ProcessResourceMonitor as ResourceMonitor\n',
         b'from rolling_io import (load as load_checkpoint, commit_stage, pause_requested,\n                        pause_after_commit, ProcessResourceMonitor as ResourceMonitor)\n'),
        (b"'checkpoint_all_stages': args.method in policy['checkpoint_all_stages_methods']", b"'checkpoint_all_stages': True"),
        (b"    config['rolling_checkpoint'] = True\n", b"    config['rolling_checkpoint'] = True\n    config['checkpoint_retention'] = 'all_completed_stages'\n"),
        (old_initial, template[start:stop]),
        (b'        commit_stage(run, stage, manifest)\n',
         b'        commit_stage(run, stage, manifest)\n        if pause_after_commit(run, stage, manifest, config):\n            monitor.finish(model)\n            return\n'),
    ]
    result = apply_edits(raw, edits, template)
    # Extra dynamic run changes, if present, remain outside these exact edits.
    def training_loop(blob):
        return next(n for n in ast.walk(ast.parse(blob)) if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name) and n.target.id == 'stage'
            and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                    and x.func.id == 'commit_stage' for x in ast.walk(n)))
    new = training_loop(result); old = training_loop(raw)
    assert isinstance(new.body[-1], ast.If) and isinstance(new.body[-1].test, ast.Call)
    assert new.body[-1].test.func.id == 'pause_after_commit'
    new.body.pop()
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)
    return result


def transplant_pipeline(raw, template):
    helper = function_bytes(template, 'resume_authorized')
    old_branch = (b"                elif j['method']=='SeqFT_LoRA' and (run/'checkpoints/latest.json').exists():\n"
                  b"                    j.update(status='ready_resume',resume_count=j.get('resume_count',0)+1)\n"
                  b"                    if j['resume_count']>3:j['status']='failed_repeated_resume'\n")
    start = template.index(b"                elif s.get('status')=='paused_checkpoint':\n")
    stop = template.index(b"                else:j['status']='failed_training'", start)
    insertion = (b"            processes.pop(key,None)\n"
        b"        for j in state['jobs'].values():\n"
        b"            if j.get('host')==HOST and j['status']=='paused_checkpoint' and resume_authorized(j):\n"
        b"                j.update(status='ready_resume',resume_count=j.get('resume_count',0)+1)\n")
    return apply_edits(raw, [
        (b'def idle(gpu,minimum):', helper+b'\n\ndef idle(gpu,minimum):'),
        (old_branch, template[start:stop]),
        (b'            processes.pop(key,None)\n', insertion),
    ], template)


def transplant_rolling(raw, template):
    old_commit = function_bytes(raw, 'commit_stage'); new_commit = function_bytes(template, 'commit_stage')
    helpers = b'\n\n'.join(function_bytes(template, n) for n in
                           ('pause_requested', 'pause_after_commit', 'authorize_resume'))
    result = apply_edits(raw, [
        (old_commit, new_commit),
        (b'def prepare_resume(run, manifest):\n    run = Path(run)\n',
         helpers+b'\n\ndef prepare_resume(run, manifest):\n    run = Path(run)\n    authorize_resume(run)\n'),
        (b"    assert 1 <= stage <= len(manifest['tasks'])", b"    assert 0 <= stage <= len(manifest['tasks'])"),
    ], template)
    return result


def hosts():
    return {f'{machine}_pct10_{backbone}': dict(backbone=backbone,
        reserved_gpus=[] if machine == 'old' else [0],
        conditions={'pct10': [0, 1, 2, 3] if machine == 'old' else [1, 2, 3]},
        minimum_free_mib=8500 if machine == 'old' else 16000)
        for machine in ('old', 'new') for backbone in ('onevision', 'qwen')}


def prepare(root, source_root):
    patch_run, patch_source = patch_function(source_root)
    manifest_path = root/MANIFEST; manifest_raw = manifest_path.read_bytes()
    assert sha(manifest_raw) == MANIFEST_SHA
    manifest = json.loads(manifest_raw)
    assert manifest['condition'] == 'pct10' and manifest['seed'] == 42
    assert manifest['formal_ready'] and not manifest['assets_pending']
    assert [t['task'] for t in manifest['tasks']] == list(range(1, 20))
    assert sum(t['files']['train']['count'] for t in manifest['tasks']) == 13105
    assert sum(t['files']['test']['count'] for t in manifest['tasks']) == 21454
    lock_path = manifest_path.parent.parent/'split_lock.json'; lock_raw = lock_path.read_bytes()
    lock = json.loads(lock_raw)['conditions']['pct10']
    assert lock['manifest_sha256'] == MANIFEST_SHA
    assert lock['frozen_ids_sha256'] == sha((manifest_path.parent/'frozen_ids.json').read_bytes())
    for task in manifest['tasks']:
        for item in task['files'].values(): assert lock['files'][item['path']] == item['sha256']
    plans = []
    for family, original_name, target_name in (
        ('native', 'two_backbone_native_cl_20261017_v2', NATIVE),
        ('forever', 'two_backbone_pct10_forever_dynamic_20261025', FOREVER)):
        original = source_root/'exp'/original_name; target = root/'exp'/target_name
        assert original.is_dir(), 'Missing actual source runner: '+str(original)
        assert not target.exists(), 'Refusing an existing target: '+str(target)
        source = {p.name: p.read_bytes() for p in sorted(original.glob('*.py'))}
        assert {'run.py', 'pipeline.py', 'checkpoints.py', 'methods.py'} <= set(source)
        blobs = dict(source); extra = {}
        if family == 'native':
            blobs['run.py'], extra['pct10_metadata_patch'] = patch_run(source['run.py'], 'native')
        else:
            assert {'core.py', 'forever_adapter.py', 'rolling_io.py'} <= set(source)
            template = source_root/'exp/two_backbone_forever_20261019_v3'
            for name, transplant in (('run.py', transplant_run), ('pipeline.py', transplant_pipeline), ('rolling_io.py', transplant_rolling)):
                reviewed = (template/name).read_bytes()
                blobs[name] = transplant(source[name], reviewed)
                extra.setdefault('interface_template_sha256', {})[name] = sha(reviewed)
            # This diagnoses the original build's metadata scope without replacing
            # any additional dynamic changes by the ordinary FOREVER runner.
            template_run = (template/'run.py').read_bytes()
            metadata_template, _ = patch_run(template_run, 'forever')
            extra['run_matches_template_after_pct10_and_interface_edits'] = blobs['run.py'] == metadata_template
            prior_build = original/'BUILD_RECEIPT.json'
            if prior_build.exists():
                extra['original_build_receipt_sha256'] = sha(prior_build.read_bytes())
                declared = json.loads(prior_build.read_bytes())['destination_py_sha256']
                assert declared == {n: sha(v) for n, v in source.items()}, 'Dynamic source differs from its build receipt'
            extra['author_helpers'] = {}
            for name in HELPERS:
                raw = (source_root/name).read_bytes()
                assert (root/name).read_bytes() == raw, 'Output overlay helper must retain the original bytes: '+name
                extra['author_helpers'][name] = sha(raw)
            assert blobs['core.py'] == source['core.py'] and blobs['forever_adapter.py'] == source['forever_adapter.py']
        root_expr = next(n.value for n in ast.parse(source['methods.py']).body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in n.targets))
        assert ast.dump(root_expr, include_attributes=False) == ast.dump(ast.parse('Path(__file__).resolve().parents[2]', mode='eval').body, include_attributes=False)
        for name, value in blobs.items(): compile(ast.parse(value), str(target/name), 'exec')
        policy_raw = (original/'execution_policy.json').read_bytes(); policy = json.loads(policy_raw)
        methods = policy['methods']
        assert methods == (['FOREVER'] if family == 'forever' else
            ['SeqFT_LoRA', 'CL_MoE', 'SMoLoRA', 'PASs_MoE', 'PaRSP', 'OSFT', 'SpaRTA'])
        policy.update(experiment_id=target_name, checkpoint_all_stages_methods=methods,
            checkpoint_other_methods='all completed stage checkpoints retained; final must be retained',
            save_final_parameters=True, task_count=19,
            conditions={'pct10': dict(train_fraction=.1, test_fraction=1.0, allowed_gpus=[0, 1, 2, 3],
                ready=True, manifest=MANIFEST, manifest_sha256=MANIFEST_SHA)}, hosts=hosts())
        policy['ownership'] = {host: ['pct10:'+method for method in methods
            if not (host.startswith('old_') and method == 'SMoLoRA')] for host in policy['hosts']}
        policy.pop('future_full', None)
        policy['stage_checkpoint_scope'] = dict(retention='all completed stages; no automatic pruning',
            pause_request='PAUSE_REQUEST.json', resume_request='RESUME_REQUEST.json',
            resume='same immutable source version and exact pause receipt; explicit authorization only',
            old_running_sources_modified=False, standalone_pipeline_main_invoked=False)
        blobs['execution_policy.json'] = encoded(policy)
        changed = sorted(n for n in source if blobs[n] != source[n])
        assert set(changed) <= ({'run.py'} if family == 'native' else {'run.py', 'pipeline.py', 'rolling_io.py'})
        differences = {name: ''.join(difflib.unified_diff(source[name].decode().splitlines(keepends=True),
            blobs[name].decode().splitlines(keepends=True), fromfile=str(original/name), tofile=str(target/name))) for name in changed}
        plans.append(dict(family=family, original=original, target=target, blobs=blobs,
            receipt=dict(source=str(original), target=str(target), source_py_sha256={n: sha(v) for n, v in source.items()},
                source_policy_sha256=sha(policy_raw), destination_sha256={n: sha(v) for n, v in blobs.items()},
                destination_py_sha256={n: sha(v) for n, v in blobs.items() if n.endswith('.py')},
                author_helpers={n: {'sha256': h} for n, h in extra.get('author_helpers', {}).items()},
                changed_py=changed, py_differences=differences, metadata_patch=extra,
                all_other_py_byte_identical=True, training_loop_unchanged=True, methods=methods)))
    return plans, dict(manifest=MANIFEST, manifest_sha256=MANIFEST_SHA,
        split_lock_sha256=sha(lock_raw), patch_source=patch_source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    assert args.root.is_absolute() and args.source_root.is_absolute()
    root = args.root.resolve(); source_root = args.source_root.resolve()
    assert root.is_dir() and source_root.is_dir()
    receipt_path = root/'summary/stage_checkpoint_dispatch_20261027/BUILD_RECEIPT.json'
    assert not receipt_path.exists(), 'Refusing an existing build receipt'
    started = time.monotonic(); plans, inputs = prepare(root, source_root)
    # Both complete plans and all input checks precede creation of either clone.
    for plan in plans:
        plan['target'].mkdir(parents=True, exist_ok=False)
        for name, value in plan['blobs'].items():
            with (plan['target']/name).open('xb') as stream: stream.write(value)
        with (plan['target']/'BUILD_RECEIPT.json').open('xb') as stream: stream.write(encoded(plan['receipt']))
    result = dict(complete=True, scope='Source construction only; no GPU or model admission',
        root=str(root), source_root=str(source_root), unix=time.time(), seconds=time.monotonic()-started,
        builder_sha256=sha(Path(__file__).read_bytes()), inputs=inputs, runners=[p['receipt'] for p in plans],
        original_sources_modified=False, author_helpers_modified=False, models=0, scores=0, jobs_launched=0)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('xb') as stream: stream.write(encoded(result))
    print(json.dumps(dict(complete=True, receipt=str(receipt_path), sha256=sha(receipt_path.read_bytes()), jobs_launched=0)))


if __name__ == '__main__': main()
