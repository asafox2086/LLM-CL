"""Build pct10 runner copies without importing or executing any model code.

Required: LLMCL_ROOT (absolute output/overlay root).
Optional: LLMCL_SOURCE_ROOT (original repository, defaults to LLMCL_ROOT).
Input: --manifest ROOT-relative pct10 manifest, or LLMCL_PCT10_MANIFEST.
Existing runner targets are always refused; this script never launches jobs.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import time

NATIVE_METHODS = ['SeqFT_LoRA', 'CL_MoE', 'SMoLoRA', 'PASs_MoE', 'PaRSP', 'OSFT', 'SpaRTA']
RUNNERS = (
    ('two_backbone_native_cl_20261017_v2', 'two_backbone_pct10_native_20261025', NATIVE_METHODS),
    ('two_backbone_forever_20261019_v3', 'two_backbone_pct10_forever_20261025', ['FOREVER']),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def patch_run(raw, family):
    """Three allowed metadata/entry edits; exact inverse preserves original bytes."""
    condition = (b"ap.add_argument('--condition', choices=['full', 'pct01'], required=True)"
                 if family == 'native' else
                 b"ap.add_argument('--condition', choices=['pct01'], required=True)")
    edits = [
        (condition, b"ap.add_argument('--condition', choices=['pct10'], required=True)"),
        (b"'train_fraction': 1.0 if args.condition == 'full' else .01",
         b"'train_fraction': policy['conditions'][args.condition]['train_fraction']"),
        (b"'primary_score': 'rougeL'" if family == 'native' else b"'primary_score': 'token_f1'",
         b"'primary_score': 'token_f1'"),
    ]
    transformed = raw
    audit = []
    for before, after in edits:
        assert transformed.count(before) == 1, (family, before.decode())
        transformed = transformed.replace(before, after, 1)
        audit.append(dict(before=before.decode(), after=after.decode(), changed=before != after))
    restored = transformed
    for before, after in reversed(edits):
        if before != after:
            assert restored.count(after) == 1
            restored = restored.replace(after, before, 1)
    assert restored == raw
    compile(ast.parse(transformed), '<pct10-runner-source-only>', 'exec')
    return transformed, dict(edits=audit, exact_inverse_bytes_equal=True,
                             training_and_method_math_unchanged=True,
                             scoring_function_unchanged=True)


def patch_forever_guards(raw):
    """Generalize only the two pct01-size guards, preserving author math."""
    total = b"sum(t['files']['train']['count'] for t in manifest['tasks'])"
    edits = [
        (b"assert len(self.train_tasks) == 19 and sum(map(len, self.train_tasks)) == 1313",
         b"assert len(self.train_tasks) == 19 and sum(map(len, self.train_tasks)) == " + total),
        (b"assert len(self.task_by_id) == 1313",
         b"assert len(self.task_by_id) == " + total),
    ]
    transformed = raw
    for before, after in edits:
        assert transformed.count(before) == 1, before.decode()
        transformed = transformed.replace(before, after, 1)
    restored = transformed
    for before, after in reversed(edits):
        assert restored.count(after) == 1
        restored = restored.replace(after, before, 1)
    assert restored == raw
    compile(ast.parse(transformed), '<pct10-forever-guards-only>', 'exec')
    return transformed, dict(
        edits=[dict(before=a.decode(), after=b.decode(), changed=True) for a, b in edits],
        exact_inverse_bytes_equal=True, scope='Two training-cardinality assertions only',
        training_and_method_math_unchanged=True, replay_and_sampling_unchanged=True)


def hosts():
    result = {}
    for machine in ('old', 'new'):
        for backbone in ('onevision', 'qwen'):
            result[machine + '_pct10_' + backbone] = dict(
                backbone=backbone, reserved_gpus=[] if machine == 'old' else [0],
                conditions={'pct10': [0, 1, 2, 3] if machine == 'old' else [1, 2, 3]},
                minimum_free_mib=8500 if machine == 'old' else 16000)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=os.environ.get('LLMCL_PCT10_MANIFEST') or
                        'data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json')
    args = parser.parse_args()
    assert 'LLMCL_ROOT' in os.environ and os.environ['LLMCL_ROOT'], 'LLMCL_ROOT is required'
    root = Path(os.environ['LLMCL_ROOT'])
    source_root = Path(os.environ.get('LLMCL_SOURCE_ROOT') or str(root))
    assert root.is_absolute() and source_root.is_absolute()
    root, source_root = root.resolve(), source_root.resolve()
    assert root.is_dir() and source_root.is_dir()
    assert args.manifest, '--manifest or LLMCL_PCT10_MANIFEST is required'
    manifest_path = Path(args.manifest)
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    manifest_path = manifest_path.resolve()
    manifest_relative = manifest_path.relative_to(root).as_posix()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest['condition'] == 'pct10' and manifest['seed'] == 42
    assert manifest['experiment_id'] == 'five_domain_native_cl_20261016_sparse'
    assert manifest['formal_ready'] and not manifest['assets_pending']
    assert len(manifest['tasks']) == 19
    assert [task['task'] for task in manifest['tasks']] == list(range(1, 20))
    assert sum(task['files']['test']['count'] for task in manifest['tasks']) == 21454
    assert all(task['files']['train']['count'] > 0 for task in manifest['tasks'])
    # Validate the frozen ledger consumed unchanged by both original load_protocol functions.
    lock_path = manifest_path.parent.parent / 'split_lock.json'
    lock = json.loads(lock_path.read_bytes())['conditions']['pct10']
    assert lock['manifest_sha256'] == sha(manifest_bytes)
    assert lock['frozen_ids_sha256'] == sha((manifest_path.parent / 'frozen_ids.json').read_bytes())
    for task in manifest['tasks']:
        for item in task['files'].values():
            assert lock['files'][item['path']] == item['sha256']
    targets = [root / 'exp' / target for _, target, _ in RUNNERS]
    assert all(not target.exists() for target in targets), 'Refusing existing runner target'
    report_path = root / 'summary/two_backbone_pct10_20261025/RUNNER_BUILD_RECEIPT.json'
    assert not report_path.exists(), 'Refusing existing build receipt'
    plans = []
    for index, (original_name, target_name, methods) in enumerate(RUNNERS):
        original, target = source_root / 'exp' / original_name, targets[index]
        source_files = sorted(original.glob('*.py'))
        assert source_files and {'run.py', 'pipeline.py', 'methods.py'} <= {p.name for p in source_files}
        # The cloned depth is identical: methods.ROOT remains output_root, including overlays.
        method_tree = ast.parse((original / 'methods.py').read_bytes())
        root_assignment = next(n for n in method_tree.body if isinstance(n, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in n.targets))
        assert ast.dump(root_assignment.value, include_attributes=False) == ast.dump(
            ast.parse('Path(__file__).resolve().parents[2]', mode='eval').body, include_attributes=False)
        blobs, differences, source_hashes = {}, {}, {}
        for path in source_files:
            raw = path.read_bytes(); source_hashes[path.name] = sha(raw)
            if path.name == 'run.py':
                value, difference = patch_run(raw, 'native' if index == 0 else 'forever')
                differences[path.name] = difference
            elif index == 1 and path.name == 'forever_adapter.py':
                value, difference = patch_forever_guards(raw)
                differences[path.name] = difference
            else:
                value = raw
            compile(ast.parse(value), str(target / path.name), 'exec')
            blobs[path.name] = value
        policy_raw = (original / 'execution_policy.json').read_bytes()
        policy = json.loads(policy_raw)
        policy['experiment_id'] = target_name
        policy['methods'] = methods
        policy['conditions'] = {'pct10': dict(train_fraction=.10, test_fraction=1.0,
            allowed_gpus=[0, 1, 2, 3], ready=True, manifest=manifest_relative,
            manifest_sha256=sha(manifest_bytes))}
        policy['hosts'] = hosts()
        policy['ownership'] = {host: ['pct10:' + method for method in methods
            if not (host.startswith('old_') and method == 'SMoLoRA')]
            for host in policy['hosts']}
        policy['task_count'] = 19
        policy.pop('future_full', None)
        policy['pct10_runner_scope'] = dict(primary_score='token_f1', test_QA=21454,
            source_runner=original_name, standalone_pipeline_main_not_invoked=True,
            coordination='One external coordinator owns all physical GPUs and calls cloned pipeline.launch.',
            SMoLoRA='Original native-BF16 CLI guard remains; old2080 ownership excludes this method.',
            source_root=str(source_root), output_root=str(root))
        blobs['execution_policy.json'] = encoded(policy)
        plans.append(dict(original=original, target=target, source_hashes=source_hashes,
                          source_policy_sha256=sha(policy_raw), blobs=blobs,
                          differences=differences, methods=methods))
    # Nothing above imports runners, initializes CUDA or executes model code.
    # Both complete plans and existing-target refusal are resolved before writing either clone.
    receipts = []
    for plan in plans:
        target = plan['target']; target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=False)
        for name, blob in plan['blobs'].items():
            with (target / name).open('xb') as stream:
                stream.write(blob)
        receipt = dict(source=str(plan['original']), target=str(target),
            source_py_sha256=plan['source_hashes'], source_policy_sha256=plan['source_policy_sha256'],
            destination_sha256={n: sha(v) for n, v in plan['blobs'].items()},
            changed_py=sorted(plan['differences']), run_diff=plan['differences']['run.py'],
            additional_py_diffs={n: d for n, d in plan['differences'].items() if n != 'run.py'},
            byte_identical_py=sorted(n for n in plan['source_hashes'] if n not in plan['differences']),
            all_other_py_byte_identical=True, methods=plan['methods'],
            methods_ROOT_unmodified_and_resolves_to_output_root=True)
        with (target / 'BUILD_RECEIPT.json').open('xb') as stream:
            stream.write(encoded(receipt))
        receipts.append(receipt)
    report = dict(complete=True, unix=time.time(), builder_sha256=sha(Path(__file__).read_bytes()),
        root=str(root), source_root=str(source_root), manifest=manifest_relative,
        manifest_sha256=sha(manifest_bytes), split_lock_sha256=sha(lock_path.read_bytes()),
        conditions=['pct10'], train_fraction=.10, test_fraction=1.0, test_QA=21454,
        train_QA=sum(t['files']['train']['count'] for t in manifest['tasks']),
        runners=receipts, old_gpu_ids=[0, 1, 2, 3], new_gpu_ids=[1, 2, 3],
        new_reserved_gpu_ids=[0], no_model_import=True, new_models=0, new_scores=0,
        jobs_launched=0, original_sources_modified=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('xb') as stream:
        stream.write(encoded(report))
    print(json.dumps(dict(complete=True, report=str(report_path),
                         report_sha256=sha(report_path.read_bytes()), jobs_launched=0)))


if __name__ == '__main__':
    main()
