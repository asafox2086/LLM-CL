"""Partition immutable native records; never resample, rewrite QA, or alter source splits."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/FiveDomainCL/six_domain_cl_20261016_finmme_coda'
DEST = ROOT / 'data/FiveDomainCL' / HERE.name


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    assert not (DEST/'split_lock.json').exists(), 'Frozen split exists; do not regenerate or overwrite it'
    policy = json.loads((HERE/'execution_policy.json').read_text())
    originals = {c: json.loads((SOURCE/c/'manifest.json').read_text()) for c in ['full', 'pct01']}
    # These are source labels. Order is our preregistered order, NOT an author-specified curriculum.
    layout = [(1, None, [None]),
              (2, 'content_type', ['Modality','Position','Organ','Size','Abnormality','Quantity','Plane','Shape','Color','KG']),
              (3, 'question_type', ['rural_urban','presence','count','comp']),
              (4, None, [None]),
              (6, 'question_type', ['general','regional','suggestion'])]
    audit = {'experiment_id': HERE.name, 'parent': SOURCE.name,
             'removed_dataset': 'AQUA', 'order_is_research_defined': True, 'conditions': {}}
    for condition, original in originals.items():
        tasks = []; source_ids = {'train': set(), 'test': set()}; covered = {'train': set(), 'test': set()}
        for parent_id, field, categories in layout:
            parent = original['tasks'][parent_id-1]
            pools = {}
            for split in ['train','test']:
                f = parent['files'][split]; path = ROOT/f['path']
                assert sha(path) == f['sha256']
                pools[split] = [json.loads(x) for x in path.read_text().splitlines()]
                source_ids[split].update(x['id'] for x in pools[split])
            if field:
                assert {x['metadata'][field] for x in pools['train']} == set(categories)
                assert {x['metadata'][field] for x in pools['test']} == set(categories)
            for category in categories:
                tid = len(tasks)+1
                task = {k:v for k,v in parent.items() if k not in ['files','training_order_sha256','training_order_rule']}
                task.update(task=tid, task_id=f'T{tid:02d}', parent_domain_task=parent_id,
                            native_label_field=field, native_label=category,
                            max_new_tokens={1:128,2:128,3:512,4:512,6:512}[parent_id], files={})
                # Explicit native task descriptor, no rewriting of original questions or answers.
                task['instruction'] = parent['instruction'] + (f' Native task: {category}.' if category else '')
                for split, pool in pools.items():
                    rows = [x for x in pool if not field or x['metadata'][field] == category]
                    assert rows, (condition, tid, split)
                    ids = {x['id'] for x in rows}; assert not covered[split] & ids
                    covered[split].update(ids)
                    path = DEST/condition/f'T{tid:02d}_{split}.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
                    # Serialization changes only; record objects must remain byte-for-byte equivalent in meaning.
                    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
                    task['files'][split] = {'path':str(path.relative_to(ROOT)), 'count':len(rows), 'sha256':sha(path),
                        'dataset_counts':dict(collections.Counter(x['dataset'] for x in rows))}
                    if split == 'train':
                        order = sorted(rows,key=lambda x:hashlib.sha256(f"42:{tid}:{x['id']}".encode()).digest())
                        task['training_order_sha256'] = hashlib.sha256(''.join(x['id']+'\n' for x in order).encode()).hexdigest()
                        task['training_order_rule'] = 'SHA256(42:stage:original_QA_ID), ascending digest'
                tasks.append(task)
        assert covered == source_ids
        assert not covered['train'] & covered['test']
        manifest = {'experiment_id':HERE.name,'condition':condition,'seed':42,'formal_ready':True,
                    'assets_pending':False,'tasks':tasks,'source_manifest_sha256':sha(SOURCE/condition/'manifest.json'),
                    'native_task_order':'Domain order, then explicit label order in prepare_native.py; no claim of official curriculum',
                    'no_qa_rewriting':True,'original_source_split_identity_preserved':True,'parent_1pct_ids_preserved':True}
        path = DEST/condition/'manifest.json'; write(path, manifest)
        policy['conditions'][condition].update(ready=True,manifest=str(path.relative_to(ROOT)),manifest_sha256=sha(path))
        audit['conditions'][condition] = {'tasks':len(tasks),'train':len(covered['train']),'test':len(covered['test']),
            'counts':[{'task':t['task'],'domain':t['domain'],'label':t['native_label'],
                       'train':t['files']['train']['count'],'test':t['files']['test']['count']} for t in tasks]}
    a=json.loads((DEST/'full/manifest.json').read_text()); b=json.loads((DEST/'pct01/manifest.json').read_text())
    for x,y in zip(a['tasks'],b['tasks']):
        assert x['files']['test']['sha256']==y['files']['test']['sha256']
        fullids={json.loads(l)['id'] for l in (ROOT/x['files']['train']['path']).read_text().splitlines()}
        smallids={json.loads(l)['id'] for l in (ROOT/y['files']['train']['path']).read_text().splitlines()}
        assert smallids <= fullids
    policy.update(training_enabled=True,task_count=len(a['tasks']))
    write(HERE/'execution_policy.json',policy);write(DEST/'partition_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False))


if __name__ == '__main__': main()
