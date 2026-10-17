"""Preserve source data; derive the single user-authorized missing-label exclusion."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse'
DEST = ROOT / 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse_labeled_20261017'
REPORT = ROOT / 'summary/two_backbone_native_cl_20261017_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == raw, f'Refuse inconsistent overwrite: {path}'
    else:
        path.write_bytes(raw)


def main():
    parent = SOURCE / 'full/manifest.json'
    assert sha(parent) == '3dac40b318c4bc7c792eeebc4e5667b172cfdf42ae835b92f4fe56998ef44504'
    manifest = json.loads(parent.read_text())
    frozen = json.loads((SOURCE / 'full/frozen_ids.json').read_text())
    original = {}
    blanks = []
    for task in manifest['tasks']:
        for split, info in task['files'].items():
            path = ROOT / info['path']
            assert sha(path) == info['sha256']
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == info['count']
            original[task['task'], split] = rows
            for row in rows:
                if not any(str(a).strip() for a in row['answers']):
                    blanks.append((task['task'], split, row['id']))
    assert blanks == [(4, 'train', 'SLAKE:1622')], blanks
    task = manifest['tasks'][3]
    source_file = ROOT / task['files']['train']['path']
    lines = source_file.read_bytes().splitlines(keepends=True)
    excluded = [json.loads(line) for line in lines if json.loads(line)['id'] == 'SLAKE:1622']
    kept = [line for line in lines if json.loads(line)['id'] != 'SLAKE:1622']
    assert len(excluded) == 1 and len(kept) == 2542
    out = DEST / 'full/T04_train.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = b''.join(kept)
    if out.exists():
        assert out.read_bytes() == raw
    else:
        out.write_bytes(raw)
    task['files']['train'].update(path=out.relative_to(ROOT).as_posix(), count=2542,
                                sha256=sha(out), dataset_counts={'SLAKE': 2542})
    frozen['T04/train'].remove('SLAKE:1622')
    ordered = sorted((json.loads(line)['id'] for line in kept),
                     key=lambda ident: hashlib.sha256(f'42:4:{ident}'.encode()).hexdigest())
    assert ordered == frozen['T04/train']
    task['training_order_sha256'] = hashlib.sha256(''.join(i+'\n' for i in ordered).encode()).hexdigest()
    manifest.update(variant_id=DEST.name, parent_manifest_sha256=sha(parent),
                    authorized_exclusions=['SLAKE:1622'], effective_train_count=131018,
                    exclusion_reason='Only original empty gold answer; explicitly authorized by user 2026-10-17')
    dump(DEST/'full/frozen_ids.json', frozen)
    dump(DEST/'full/manifest.json', manifest)
    lock = {'experiment_id': manifest['experiment_id'], 'variant_id': DEST.name, 'seed':42,
            'method_independent':True, 'conditions':{'full':{
                'manifest_sha256':sha(DEST/'full/manifest.json'),
                'frozen_ids_sha256':sha(DEST/'full/frozen_ids.json'),
                'files':{info['path']: info['sha256'] for t in manifest['tasks'] for info in t['files'].values()}}}}
    dump(DEST/'split_lock.json', lock)
    pct = json.loads((SOURCE/'pct01/frozen_ids.json').read_text())
    for key, ids in pct.items():
        assert set(ids) <= set(frozen[key]) if key.endswith('/train') else ids == frozen[key]
    assert sum(t['files']['train']['count'] for t in manifest['tasks']) == 131018
    assert sum(t['files']['test']['count'] for t in manifest['tasks']) == 21454
    assert sha(parent) == manifest['parent_manifest_sha256']
    assert sha(SOURCE/'pct01/manifest.json') == '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    audit = {'status':'verified', 'user_authorization':'只有这一个有问题？那就把这个直接踢掉就可以了',
             'excluded_id':'SLAKE:1622', 'excluded_original_record':excluded[0],
             'source_file_sha256':sha(source_file), 'parent_manifest_sha256':sha(parent),
             'derived_manifest':(DEST/'full/manifest.json').relative_to(ROOT).as_posix(),
             'derived_manifest_sha256':lock['conditions']['full']['manifest_sha256'],
             'original_full_train':131019, 'effective_full_train':131018,
             'pct01_train_unchanged':1313,'test_unchanged':21454,
             'blank_answers_audit':'All 131019 original train and 21454 test rows: exactly this one blank; no other exclusion.',
             'integrity':'Only T04 train loses this ID. Retained serialized QA lines byte-identical; other 37 QA files reused unchanged. Original manifest/files untouched; 1% nested IDs unchanged.',
             'legacy_full_failure':'stage4 empty answer tokens; repeated incomplete legacy run is not a completed result; old queue disabled',
             'resumption':'New framework full jobs start fresh; no incompatible checkpoint/prediction reuse.'}
    dump(REPORT/'FULL_MISSING_LABEL_EXCLUSION.json', audit)
    for path in DEST.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
