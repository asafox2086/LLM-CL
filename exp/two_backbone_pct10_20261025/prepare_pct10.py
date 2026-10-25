"""Freeze a nested10% native-task dataset from the two existing sealed pools.

Standard library only. Run explicitly with LLMCL_ROOT set on a host containing
the original source assets. This file does not train, load models or launch jobs.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path


FULL_SHA = 'cf737555dab3646c9916dc63a66f1d52285eebbeed4aa414e807ae5b08a0d952'
PCT01_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
EXPERIMENT = 'five_domain_native_cl_20261016_sparse'
DESTINATION = 'data/FiveDomainCL/five_domain_native_cl_pct10_20261025'
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
SEED, FRACTION = 42, .10


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')


def rooted(root, relative):
    relative = Path(relative)
    assert not relative.is_absolute(), 'Source data paths must be ROOT-relative'
    assert '..' not in relative.parts, 'Source path escapes LLMCL_ROOT'
    # Existing read-only source trees may be symlinks; their bound SHA, not
    # physical storage placement, defines the source data identity.
    return root / relative


def object_digest(row):
    raw = json.dumps(row, sort_keys=True, ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False).encode('utf8')
    return hashlib.sha256(raw).hexdigest()


def kind(row):
    # Keep the original subset_one_percent priority, including explicit None.
    metadata = row.get('metadata', {})
    return str(metadata.get('question_type', metadata.get('content_type', 'unspecified')))


def source_metadata(root, info, task, split, lock):
    """Read immutable QA once, retain only IDs/type/content digests, never bodies."""
    path = rooted(root, info['path'])
    assert lock['files'][info['path']] == info['sha256']
    hasher, entries = hashlib.sha256(), []
    with path.open('rb') as stream:
        for raw in stream:
            hasher.update(raw)
            assert raw.strip(), 'Blank source JSONL line'
            row = json.loads(raw)
            assert isinstance(row['id'], str) and row['id']
            assert isinstance(row['dataset'], str) and row['dataset']
            answers = row['answers']
            assert isinstance(answers, list) and answers
            assert all(isinstance(answer, str) for answer in answers)
            assert any(answer.strip() for answer in answers), 'Source has empty gold: ' + row['id']
            if split == 'train':
                assert isinstance(row['answer'], str) and row['answer'].strip(), \
                    'Source training target is empty: ' + row['id']
            entries.append(dict(id=row['id'], dataset=row['dataset'], kind=kind(row),
                                task=task, object_sha256=object_digest(row)))
    assert hasher.hexdigest() == info['sha256'], info['path']
    assert len(entries) == info['count'] and len({x['id'] for x in entries}) == len(entries)
    return entries


def load_pool(root, condition, info, expected_sha):
    manifest_path = rooted(root, info['manifest'])
    assert info['manifest_sha256'] == expected_sha == sha(manifest_path)
    manifest = read(manifest_path)
    assert manifest['experiment_id'] == EXPERIMENT and manifest['condition'] == condition
    assert manifest['seed'] == SEED and manifest['formal_ready'] and not manifest['assets_pending']
    assert [t['task'] for t in manifest['tasks']] == list(range(1, 20))
    assert all(t['task_id'] == f'T{t["task"]:02}' for t in manifest['tasks'])
    assert tuple(dict.fromkeys(t['domain'] for t in manifest['tasks'])) == DOMAINS
    lock = read(manifest_path.parent.parent / 'split_lock.json')['conditions'][condition]
    assert lock['manifest_sha256'] == expected_sha
    frozen_path = manifest_path.parent / 'frozen_ids.json'
    assert sha(frozen_path) == lock['frozen_ids_sha256']
    frozen = read(frozen_path)
    data, seen = {}, {'train': set(), 'test': set()}
    for task in manifest['tasks']:
        tid = task['task']
        for split in ('train', 'test'):
            rows = source_metadata(root, task['files'][split], tid, split, lock)
            ids = [x['id'] for x in rows]
            assert not seen[split].intersection(ids), 'Duplicate source QA ID across native tasks'
            seen[split].update(ids)
            ordered = sorted(ids, key=lambda rid: hashlib.sha256(f'42:{tid}:{rid}'.encode()).digest()) \
                if split == 'train' else ids
            assert ordered == frozen[f'T{tid:02}/{split}']
            if split == 'train':
                assert hashlib.sha256(''.join(rid + '\n' for rid in ordered).encode()).hexdigest() \
                    == task['training_order_sha256']
            data[tid, split] = rows
    assert seen['train'].isdisjoint(seen['test'])
    return manifest, data, seen


def select_ten_percent(entries):
    """Original per-dataset ceil/proportional largest-remainder/hash sampler."""
    strata = defaultdict(list)
    for row in entries:
        strata[row['dataset'], row['kind']].append(row)
    selected, audit = set(), []
    for dataset in sorted({key[0] for key in strata}):
        keys = sorted(key for key in strata if key[0] == dataset)
        count = sum(len(strata[key]) for key in keys)
        target = math.ceil(FRACTION * count)
        quotas = {key: math.floor(target * len(strata[key]) / count) for key in keys}
        priority = sorted(keys, key=lambda key: (
            -(target * len(strata[key]) / count - quotas[key]), key))
        for key in priority[:target - sum(quotas.values())]:
            quotas[key] += 1
        assert sum(quotas.values()) == target
        detail = []
        for key in keys:
            ordered = sorted(strata[key], key=lambda row: hashlib.sha256(('42:' + row['id']).encode()).digest())
            chosen = [row['id'] for row in ordered[:quotas[key]]]
            assert len(chosen) == quotas[key] and not selected.intersection(chosen)
            selected.update(chosen)
            detail.append(dict(type=key[1], full_count=len(ordered), quota=quotas[key]))
        audit.append(dict(dataset=dataset, effective_full_count=count, target_count=target, strata=detail))
    return selected, audit


def main():
    root = Path(os.environ['LLMCL_ROOT']).expanduser().resolve(strict=True)
    assert root.is_dir()
    destination = rooted(root, DESTINATION)
    assert not destination.exists(), 'Refuse existing destination or frozen files: ' + str(destination)
    policy_path = root / 'exp/two_backbone_native_cl_20261017_v2/execution_policy.json'
    policy = read(policy_path)
    assert policy['seed'] == SEED and policy['epochs'] == 1 and policy['task_count'] == 19
    full, full_data, full_ids = load_pool(root, 'full', policy['conditions']['full'], FULL_SHA)
    small, small_data, small_ids = load_pool(root, 'pct01', policy['conditions']['pct01'], PCT01_SHA)
    assert len(full_ids['train']) == 131018 and len(small_ids['train']) == 1313
    assert len(full_ids['test']) == len(small_ids['test']) == 21454
    assert full.get('authorized_exclusions') == ['SLAKE:1622']
    assert 'SLAKE:1622' not in full_ids['train'] | small_ids['train'] | full_ids['test']
    all_full = [row for task in full['tasks'] for row in full_data[task['task'], 'train']]
    selected, sampling = select_ten_percent(all_full)
    assert small_ids['train'] <= selected <= full_ids['train'], \
        'Original sampler did not produce nested IDs; do not silently alter quotas or select replacements'
    assert len(selected) == sum(x['target_count'] for x in sampling) == 13105
    assert selected.isdisjoint(full_ids['test'])
    for large_task, small_task in zip(full['tasks'], small['tasks']):
        tid = large_task['task']
        for key in ('task', 'task_id', 'domain', 'native_label', 'instruction', 'max_new_tokens'):
            assert large_task.get(key) == small_task.get(key), (tid, key)
        assert large_task['files']['test']['sha256'] == small_task['files']['test']['sha256']
        assert full_data[tid, 'test'] == small_data[tid, 'test']
        large_by_id = {row['id']: row for row in full_data[tid, 'train']}
        assert all(large_by_id[row['id']] == row for row in small_data[tid, 'train'])
        assert {row['id'] for row in small_data[tid, 'train']} <= selected.intersection(large_by_id)

    # All source and nesting checks finish before the first output is created.
    destination.mkdir(parents=True, exist_ok=False)
    folder = destination / 'pct10'
    folder.mkdir()
    manifest = deepcopy(full)
    manifest.update(condition='pct10', seed=SEED, train_fraction=FRACTION, test_fraction=1.0,
        variant_id=destination.name, parent_manifest_sha256=FULL_SHA,
        source_full_manifest_sha256=FULL_SHA, source_pct01_manifest_sha256=PCT01_SHA,
        source_effective_full_train_count=131018, effective_train_count=len(selected),
        parent_1pct_ids_preserved=True, formal_ready=True, assets_pending=False,
        sampling_rule='Original per-dataset ceil(0.10*N), proportional native-type largest remainder; SHA256(42:ID) stratum prefix',
        no_qa_rewriting=True, original_source_split_identity_preserved=True)
    frozen, files, stages = {}, {}, []
    output_paths = []
    for task in manifest['tasks']:
        tid = task['task']
        chosen = [row for row in full_data[tid, 'train'] if row['id'] in selected]
        assert chosen and {row['id'] for row in small_data[tid, 'train']} <= {row['id'] for row in chosen}
        original_info = deepcopy(task['files']['train'])
        source_path = rooted(root, original_info['path'])
        output_path = folder / f'T{tid:02}_train.jsonl'
        check, written = hashlib.sha256(), []
        with source_path.open('rb') as source, output_path.open('xb') as target:
            for raw in source:
                check.update(raw)
                rid = json.loads(raw)['id']
                if rid in selected:
                    # Retain each selected original serialized QA line unchanged.
                    target.write(raw)
                    written.append(rid)
        assert check.hexdigest() == original_info['sha256']
        assert written == [row['id'] for row in chosen]
        relative = output_path.relative_to(root).as_posix()
        train_info = deepcopy(original_info)
        train_info.update(path=relative, count=len(chosen), sha256=sha(output_path),
                          dataset_counts=dict(Counter(row['dataset'] for row in chosen)))
        if 'original_type_counts' in train_info:
            train_info['original_type_counts'] = dict(Counter(row['kind'] for row in chosen))
        task['files']['train'] = train_info
        # Test path/hash/count are the source full manifest's original objects.
        test_info = task['files']['test']
        ordered = sorted(written, key=lambda rid: hashlib.sha256(f'42:{tid}:{rid}'.encode()).digest())
        task['training_order_sha256'] = hashlib.sha256(''.join(rid + '\n' for rid in ordered).encode()).hexdigest()
        task['training_order_rule'] = 'SHA256(42:stage:original_QA_ID), ascending digest'
        frozen[f'T{tid:02}/train'] = ordered
        frozen[f'T{tid:02}/test'] = [row['id'] for row in full_data[tid, 'test']]
        files[relative] = train_info['sha256']
        files[test_info['path']] = test_info['sha256']
        stages.append(dict(task=tid, domain=task['domain'], train_QA=len(chosen),
                           pct01_QA=len(small_data[tid, 'train']), test_QA=test_info['count'],
                           optimizer_updates_one_epoch=math.ceil(len(chosen) / 16),
                           training_order_sha256=task['training_order_sha256']))
        output_paths.append(output_path)
    assert sum(x['train_QA'] for x in stages) == 13105
    assert sum(x['test_QA'] for x in stages) == 21454
    manifest_path, frozen_path = folder / 'manifest.json', folder / 'frozen_ids.json'
    for path, value in ((frozen_path, frozen), (manifest_path, manifest)):
        with path.open('xb') as stream:
            stream.write(encode(value))
        output_paths.append(path)
    lock = dict(experiment_id=EXPERIMENT, variant_id=destination.name, seed=SEED,
                method_independent=True, conditions=dict(pct10=dict(
                    manifest_sha256=sha(manifest_path), frozen_ids_sha256=sha(frozen_path), files=files)))
    lock_path = destination / 'split_lock.json'
    with lock_path.open('xb') as stream:
        stream.write(encode(lock))
    output_paths.append(lock_path)
    audit = dict(status='frozen', experiment_id=EXPERIMENT, variant_id=destination.name,
        condition='pct10', seed=SEED, train_fraction=FRACTION, test_fraction=1.0,
        source_policy_sha256=sha(policy_path), source_full_manifest_sha256=FULL_SHA,
        source_pct01_manifest_sha256=PCT01_SHA, source_full_train_QA=131018,
        source_pct01_train_QA=1313, train_QA=13105, test_QA=21454, stages=stages,
        total_optimizer_updates_one_epoch=sum(x['optimizer_updates_one_epoch'] for x in stages),
        dataset_sampling=sampling, pct01_nested_per_task=True, train_test_disjoint=True,
        all_QA_have_nonempty_gold=True, original_selected_QA_lines_byte_identical=True,
        test_files_referenced_without_copy_or_changes=True, images_modified=False,
        inherited_full_exclusion=['SLAKE:1622'], additional_exclusions=[],
        stratum_order='dataset,type ascending; remainder descending then key ascending',
        stratum_ID_order='SHA256(42:original_QA_ID), ascending digest',
        train_order='SHA256(42:stage:original_QA_ID), ascending digest',
        manifest=manifest_path.relative_to(root).as_posix(), manifest_sha256=sha(manifest_path),
        frozen_ids_sha256=sha(frozen_path), split_lock_sha256=sha(lock_path),
        QA_bodies_in_audit=False, new_models=0, new_scores=0,
        interpretation='Data preparation only; no10% training/results or transfer of1% scores.')
    audit_path = destination / 'SAMPLING_AUDIT.json'
    with audit_path.open('xb') as stream:
        stream.write(encode(audit))
    output_paths.append(audit_path)
    for path in output_paths:
        path.chmod(0o444)
    print(json.dumps(dict(manifest=audit['manifest'], manifest_sha256=audit['manifest_sha256'],
        train_QA=13105, test_QA=21454, stages=19,
        updates=audit['total_optimizer_updates_one_epoch'], nested_pct01=True,
        new_models=0, new_scores=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
