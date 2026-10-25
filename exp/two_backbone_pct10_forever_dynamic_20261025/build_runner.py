"""Metadata-only builder for the independent WIP pct10 dynamic FOREVER runner.

LLMCL_ROOT and LLMCL_SOURCE_ROOT are required absolute repository roots.
The output already owns core.py and forever_adapter.py; neither is overwritten.
No runner import, model execution, transport, admission or launch occurs here.
"""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import json
import os
import time

NAME = 'two_backbone_pct10_forever_dynamic_20261025'
DEFAULT_SOURCE = 'exp/two_backbone_forever_20261019_v3'
HELPERS = ('analysis/forever_multimodal_core_20261019.py',
           'analysis/forever_multimodal_replay_v2_20261019.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def patch_run(raw):
    """Accept the original v3 or frozen pct10 metadata, not arbitrary runners."""
    variants = [
        (b"ap.add_argument('--condition', choices=['pct01'], required=True)",
         b"ap.add_argument('--condition', choices=['pct10'], required=True)"),
        (b"'train_fraction': 1.0 if args.condition == 'full' else .01",
         b"'train_fraction': policy['conditions'][args.condition]['train_fraction']"),
        (b"'primary_score': 'rougeL'", b"'primary_score': 'token_f1'"),
    ]
    result, changes = raw, []
    for original, wanted in variants:
        assert result.count(original) + result.count(wanted) == 1
        before = original if original in result else wanted
        result = result.replace(before, wanted, 1)
        changes.append(dict(before=before.decode(), after=wanted.decode(), changed=before != wanted))
    inverse = result
    for item in reversed(changes):
        if item['changed']:
            old, new = item['before'].encode(), item['after'].encode()
            assert inverse.count(new) == 1
            inverse = inverse.replace(new, old, 1)
    assert inverse == raw
    compile(ast.parse(result), '<dynamic-run-metadata-only>', 'exec')
    return result, dict(edits=changes, exact_inverse_bytes_equal=True,
                        main_training_loop_unchanged=True, scoring_functions_unchanged=True)


def difference(old, new, old_name, new_name):
    return dict(source=str(old_name), source_sha256=sha(old), target=str(new_name),
        target_sha256=sha(new), unified_diff=''.join(difflib.unified_diff(
            old.decode('utf-8').splitlines(keepends=True), new.decode('utf-8').splitlines(keepends=True),
            fromfile=str(old_name), tofile=str(new_name))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-runner', default=DEFAULT_SOURCE)
    parser.add_argument('--manifest', default=os.environ.get('LLMCL_PCT10_MANIFEST') or
        'data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json')
    args = parser.parse_args()
    for variable in ('LLMCL_ROOT', 'LLMCL_SOURCE_ROOT'):
        assert os.environ.get(variable) and Path(os.environ[variable]).is_absolute(), variable + ' is required'
    root = Path(os.environ['LLMCL_ROOT']).resolve()
    source_root = Path(os.environ['LLMCL_SOURCE_ROOT']).resolve()
    assert root.is_dir() and source_root.is_dir()
    source = Path(args.source_runner)
    source = (source if source.is_absolute() else source_root / source).resolve()
    assert source.name in ('two_backbone_forever_20261019_v3', 'two_backbone_pct10_forever_20261025')
    target = root / 'exp' / NAME
    assert source != target.resolve() and target.is_dir()
    owned = {n: (target / n).read_bytes() for n in ('core.py', 'forever_adapter.py')}
    self_bytes = Path(__file__).read_bytes()
    builder_target = target / 'build_runner.py'
    if builder_target.exists():
        assert builder_target.read_bytes() == self_bytes, 'Never overwrite a different builder'
    receipt_path = target / 'BUILD_RECEIPT.json'
    assert not receipt_path.exists(), 'Refusing to overwrite a prior build receipt'
    manifest_path = Path(args.manifest)
    manifest_path = (manifest_path if manifest_path.is_absolute() else root / manifest_path).resolve()
    relative = manifest_path.relative_to(root).as_posix()
    manifest_bytes = manifest_path.read_bytes(); manifest = json.loads(manifest_bytes)
    assert manifest['condition'] == 'pct10' and manifest['seed'] == 42
    assert manifest['experiment_id'] == 'five_domain_native_cl_20261016_sparse'
    assert manifest['formal_ready'] and not manifest['assets_pending']
    assert [t['task'] for t in manifest['tasks']] == list(range(1, 20))
    assert sum(t['files']['test']['count'] for t in manifest['tasks']) == 21454
    lock_path = manifest_path.parent.parent / 'split_lock.json'
    lock = json.loads(lock_path.read_bytes())['conditions']['pct10']
    assert lock['manifest_sha256'] == sha(manifest_bytes)
    assert lock['frozen_ids_sha256'] == sha((manifest_path.parent / 'frozen_ids.json').read_bytes())
    for task in manifest['tasks']:
        for item in task['files'].values():
            assert lock['files'][item['path']] == item['sha256']
    helpers = {}
    for name in HELPERS:
        original, installed = source_root / name, root / name
        original_bytes, installed_bytes = original.read_bytes(), installed.read_bytes()
        assert original_bytes == installed_bytes, 'Overlay must expose the same pinned helper: ' + name
        helpers[name] = dict(source=str(original), runtime=str(installed), sha256=sha(installed_bytes))
    core_tree = ast.parse(owned['core.py'])
    pinned = next(n.value.value for n in core_tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'LEGACY_CORE_SHA256' for t in n.targets))
    assert helpers[HELPERS[0]]['sha256'] == pinned
    source_files = sorted(source.glob('*.py'))
    source_map = {p.name: p.read_bytes() for p in source_files}
    assert {'run.py', 'pipeline.py', 'methods.py', 'forever_adapter.py'} <= set(source_map)
    assert not {'core.py', 'build_runner.py'} & set(source_map)
    root_assignment = next(n for n in ast.parse(source_map['methods.py']).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in n.targets))
    assert ast.dump(root_assignment.value, include_attributes=False) == ast.dump(
        ast.parse('Path(__file__).resolve().parents[2]', mode='eval').body, include_attributes=False)
    pipeline_sources = next(n for n in ast.parse(source_map['pipeline.py']).body
                            if isinstance(n, ast.FunctionDef) and n.name == 'sources')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == 'glob' and n.args and isinstance(n.args[0], ast.Constant)
               and n.args[0].value == '*.py' for n in ast.walk(pipeline_sources))
    blobs = {n: raw for n, raw in source_map.items() if n != 'forever_adapter.py'}
    blobs['run.py'], run_diff = patch_run(blobs['run.py'])
    if not builder_target.exists():
        blobs['build_runner.py'] = self_bytes
    original_policy_bytes = (source / 'execution_policy.json').read_bytes()
    policy = json.loads(original_policy_bytes)
    assert policy['methods'] == ['FOREVER']
    policy.update(experiment_id=NAME, methods=['FOREVER'], task_count=19,
        conditions={'pct10': dict(train_fraction=.1, test_fraction=1.0, ready=True,
            allowed_gpus=[0, 1, 2, 3], manifest=relative, manifest_sha256=sha(manifest_bytes))})
    policy['hosts'] = {}
    for machine in ('old', 'new'):
        for backbone in ('onevision', 'qwen'):
            policy['hosts'][machine + '_pct10_' + backbone] = dict(backbone=backbone,
                reserved_gpus=[] if machine == 'old' else [0],
                conditions={'pct10': [0, 1, 2, 3] if machine == 'old' else [1, 2, 3]},
                minimum_free_mib=8500 if machine == 'old' else 16000)
    policy['ownership'] = {host: ['pct10:FOREVER'] for host in policy['hosts']}
    policy.pop('future_full', None); policy.pop('pct10_runner_scope', None)
    policy['dynamic_runner_scope'] = dict(status='WIP build only; not admission or activation',
        primary_score='token_f1', main_protocol='shared task-level optimizer/cosine/order',
        author_current_HF_phase_restarts_reproduced=False,
        dynamic_over24_actual_GPU_validation='pending', original_sources_modified=False,
        standalone_pipeline_main_not_invoked=True, upstream_revision='bbeb549a21e59d9490f20a2b02099ee766a67f48')
    blobs['execution_policy.json'] = encoded(policy)
    for name, raw in {**blobs, **owned}.items():
        if name.endswith('.py'):
            compile(ast.parse(raw), str(target / name), 'exec')
    assert all(not (target / name).exists() for name in blobs), 'Refusing existing copied runner files'
    science_diffs = dict(
        core=difference((source_root / HELPERS[0]).read_bytes(), owned['core.py'],
                        source_root / HELPERS[0], target / 'core.py'),
        adapter=difference(source_map['forever_adapter.py'], owned['forever_adapter.py'],
                           source / 'forever_adapter.py', target / 'forever_adapter.py'))
    # All validation is complete before creating any previously missing file.
    for name, raw in blobs.items():
        with (target / name).open('xb') as stream:
            stream.write(raw)
    assert all((target / name).read_bytes() == raw for name, raw in owned.items())
    all_py = {p.name: sha(p.read_bytes()) for p in sorted(target.glob('*.py'))}
    assert all_py['core.py'] == sha(owned['core.py'])
    assert all_py['forever_adapter.py'] == sha(owned['forever_adapter.py'])
    assert all_py['build_runner.py'] == sha(self_bytes)
    report = dict(complete=True, status='WIP metadata build; no admission', unix=time.time(),
        root=str(root), source_root=str(source_root), source_runner=str(source), target=str(target),
        manifest=relative, manifest_sha256=sha(manifest_bytes), split_lock_sha256=sha(lock_path.read_bytes()),
        train_QA=sum(t['files']['train']['count'] for t in manifest['tasks']), test_QA=21454,
        source_py_sha256={n: sha(raw) for n, raw in source_map.items()},
        source_policy_sha256=sha(original_policy_bytes), destination_py_sha256=all_py,
        destination_policy_sha256=sha(blobs['execution_policy.json']), run_metadata_diff=run_diff,
        scientific_source_diffs=science_diffs, author_helpers=helpers,
        byte_identical_copied_py=sorted(n for n in source_map if n not in ('run.py', 'forever_adapter.py')),
        existing_core_adapter_preserved=True, sources_function_includes_core_and_builder=True,
        old1pct_and_existing_pct10_clones_modified=False, no_runner_or_torch_import=True,
        models_executed=0, scores=0, transports=0, jobs_launched=0,
        dynamic_over24_actual_GPU_validation='pending')
    with receipt_path.open('xb') as stream:
        stream.write(encoded(report))
    print(json.dumps(dict(complete=True, status=report['status'], receipt=str(receipt_path),
                         receipt_sha256=sha(receipt_path.read_bytes()), jobs_launched=0)))


if __name__ == '__main__':
    main()
