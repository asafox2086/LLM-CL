"""Pin, download and select real medical QA; never use external probes for training."""
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path
import requests
import pyarrow.parquet as pq
from common import ROOT, digest_bytes, write_json

BASE = ROOT / 'data/medical_qa_med'
COUNTS = {'train': 1000, 'dev': 100, 'test': 200}
SOURCES = {
    'medmcqa': ('openlifescienceai/medmcqa', '91c6572c454088bf71b679ad90aa8dffcd0d5868',
                 ['data/train-00000-of-00001.parquet', 'data/validation-00000-of-00001.parquet']),
    'medqa': ('GBaker/MedQA-USMLE-4-options', '0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18',
               ['phrases_no_exclude_train.jsonl', 'phrases_no_exclude_test.jsonl']),
}

def key(row):
    return ' '.join(row['question'].lower().split())

def prepare():
    session = requests.Session(); session.trust_env = False
    sources = {}; pools = {}
    for dataset, (repo, revision, names) in SOURCES.items():
        pools[dataset] = []
        for name in names:
            url = f'https://hf-mirror.com/datasets/{repo}/resolve/{revision}/{name}'
            path = BASE / 'raw' / dataset / Path(name).name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                response = session.get(url, timeout=180); response.raise_for_status()
                path.with_suffix('.tmp').write_bytes(response.content)
                path.with_suffix('.tmp').replace(path)
            raw = path.read_bytes()
            sources[f'{dataset}/{name}'] = {'url': url, 'revision': revision, 'sha256': digest_bytes(raw)}
            rows = pq.read_table(io.BytesIO(raw)).to_pylist() if name.endswith('.parquet') else [json.loads(x) for x in raw.splitlines()]
            good = []
            for i, r in enumerate(rows):
                if dataset == 'medmcqa':
                    if r.get('choice_type') != 'single':
                        continue
                    choices = [r['op'+letter] for letter in 'abcd']; answer = int(r['cop'])
                else:
                    choices = [r['options'][letter] for letter in 'ABCD']; answer = 'ABCD'.index(r['answer_idx'])
                if not r.get('question') or any(not x for x in choices):
                    continue
                good.append({'instance_id': f'{dataset}_{Path(name).stem}_{i}', 'question': r['question'],
                             'choices': choices, 'answer': answer, 'subject': r.get('subject_name'),
                             'source_row': i, 'source_file': name, 'domain': 'medical'})
            random.Random(42).shuffle(good)
            pools[dataset].append(good)
            print(json.dumps({'event': 'download_med', 'dataset': dataset, 'file': name, 'eligible': len(good)}), flush=True)
    # Test/dev are reserved first across BOTH datasets. Exact question deduplication prevents
    # their reuse in any training task. Near duplicates remain a documented limitation.
    selected = {d: {} for d in pools}
    # Prospective knowledge panel is excluded from the new selected QA datasets.
    reserved = {key(json.loads(x)) for x in (ROOT/'data/knowledge_probe/probes.jsonl').read_text().splitlines()}
    for dataset, (_, heldout) in pools.items():
        unique = []
        for row in heldout:
            if key(row) not in reserved:
                reserved.add(key(row)); unique.append(row)
            if len(unique) == (300 if dataset == 'medmcqa' else 200):
                break
        selected[dataset]['test'] = unique[:200]
        if dataset == 'medmcqa':
            selected[dataset]['dev'] = unique[200:300]
    # MedQA mirror has train/test only: reserve dev from official training pool.
    dev = []
    for row in pools['medqa'][0]:
        if key(row) not in reserved:
            reserved.add(key(row)); dev.append(row)
        if len(dev) == 100:
            break
    selected['medqa']['dev'] = dev
    used = set(reserved)
    for dataset, (training, _) in pools.items():
        selected[dataset]['train'] = []
        for row in training:
            if key(row) in used:
                continue
            used.add(key(row)); selected[dataset]['train'].append(row)
            if len(selected[dataset]['train']) == 1000:
                break
    manifest = {'seed': 42, 'sources': sources, 'tasks': {},
                'selection': 'single-answer only; seeded shuffle; global exact-question dedup; no repeated sampling',
                'split_policy': {'medmcqa': 'train from official train; disjoint dev/test from labeled official validation (not unlabeled official test)',
                                 'medqa': 'train/dev from official train; test from official test; mirror has no separate dev'},
                'limitation': 'Exact-question deduplication is not semantic or near-duplicate decontamination.'}
    for dataset, splits in selected.items():
        task = 'medical_' + dataset
        manifest['tasks'][task] = {}
        for split, rows in splits.items():
            assert len(rows) == COUNTS[split]
            path = BASE / dataset / f'{split}.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
            manifest['tasks'][task][split] = {'path': str(path.relative_to(ROOT)), 'count': len(rows),
                                            'sha256': digest_bytes(path.read_bytes()),
                                            'answers': dict(Counter('ABCD'[r['answer']] for r in rows))}
    write_json(BASE/'manifest.json', manifest)

if __name__ == '__main__':
    prepare()
