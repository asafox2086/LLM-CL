"""Fixed equal-budget views. Original questions/answers/annotations stay untouched."""
import hashlib
import heapq
import io
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/TenTaskCL/equal_budget_20261012'
SEED = 42
COUNTS = {'train': 2000, 'dev': 100, 'test': 200}
INSTRUCTIONS = [
    'Answer the question using the supplied passage. Give the answer directly.',
    'Answer the question using the image. Give the answer directly.',
    'Answer the medical question accurately and directly.',
    'Answer the medical question using the image. Give the answer directly.',
    'Answer the contract question using the supplied text. Give all necessary answer parts.',
    'Answer the contract question using the supplied document images. Give all necessary answer parts.',
    'Answer the financial question using the supplied text and table. Give the answer directly.',
    'Answer the financial question using the supplied document images. Give the answer directly.',
    'Answer the art history question directly.',
    'Answer the art question using the painting. Give the answer directly.',
]
DOMAINS = ['natural','natural','medical','medical','legal','legal','finance','finance','art','art']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def rank(key):
    return hashlib.sha256(f'{SEED}:{key}'.encode()).hexdigest()


def group(row):
    if row.get('images'):
        return '|'.join(row['images'])
    meta = row.get('metadata', {})
    return meta.get('document_title') or meta.get('filename') or hashlib.sha256(row.get('context', row['id']).encode()).hexdigest()


def cohort(row):
    return int(rank(group(row))[:8], 16) % 10


def read_rows(dataset, source_split=None, heldout=None):
    directory = ROOT/'data'/dataset/'open_ended'
    path = directory/f'{source_split}.jsonl' if (directory/f'{source_split}.jsonl').is_file() else directory/'all.jsonl'
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            actual = row.get('source_split') or row.get('metadata', {}).get('source_split')
            if source_split is not None and actual != source_split and path.name == 'all.jsonl':
                continue
            if heldout is not None and (cohort(row) == 0) != heldout:
                continue
            yield row


def take(rows, count):
    chosen = heapq.nsmallest(count, rows, key=lambda row: rank(row.get('dataset','')+':'+row['id']))
    if len(chosen) != count:
        raise ValueError(f'need {count} unique eligible QA, found {len(chosen)}')
    keys = [row.get('dataset','')+':'+row['id'] for row in chosen]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate source QA IDs')
    return chosen


def canonical(row):
    result = dict(row)
    dataset = row['dataset']
    result['source_id'] = row['id']
    result['id'] = dataset+':'+row['id']
    result['images'] = [name if name.startswith('data/') else f'data/{dataset}/{name}' for name in row.get('images', [])]
    result.setdefault('context', '')
    result.setdefault('answers', [row['answer']])
    return result


def squad(split):
    source = ROOT/'data/SQuAD11/raw'/('train-v1.1.json' if split == 'train' else 'dev-v1.1.json')
    for article in json.loads(source.read_text())['data']:
        for paragraph in article['paragraphs']:
            for question in paragraph['qas']:
                answers = [answer['text'] for answer in question['answers']]
                yield {'id':question['id'], 'dataset':'SQuAD11', 'question':question['question'],
                       'context':paragraph['context'], 'answer':answers[0], 'answers':answers,
                       'images':[], 'source_split':split, 'metadata':{'article_title':article['title']}}


def cuad_splits():
    with zipfile.ZipFile(ROOT/'data/CUAD/raw/source.zip') as source:
        member = next(name for name in source.namelist() if name.endswith('/data.zip'))
        packed = source.read(member)
    result = {}
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        for split, suffix in [('train','train_separate_questions.json'), ('test','test.json')]:
            member = next(name for name in archive.namelist() if name.endswith(suffix))
            ids = set()
            for article in json.loads(archive.read(member))['data']:
                for paragraph in article['paragraphs']:
                    for question in paragraph['qas']:
                        ids.add(re.sub(r'_\d+$', '', question['id']) if split == 'train' else question['id'])
            result[split] = ids
    assert not result['train'] & result['test']
    return result


def aqua(split, knowledge):
    raw = json.loads((ROOT/'data/AQUA/raw'/({'dev':'val'}.get(split,split)+'.json')).read_text())
    for index, row in enumerate(raw):
        if row['need_external_knowledge'] != knowledge:
            continue
        # Existing dataset questions/answers remain byte-for-byte equal as strings.
        # Knowledge text-only evaluation follows ArtQuest's closed-book ablation.
        yield {'id':f'{split}-{index}', 'dataset':'AQUA', 'question':row['question'],
               'answer':row['answer'], 'answers':[row['answer']], 'context':'',
               'images':[] if knowledge else [f'data/AQUA/selected_images/{row["image"]}'],
               'source_split':split, 'metadata':{'original_image':row['image'],
                    'need_external_knowledge':knowledge, 'aqua_original_row_index':index}}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = {}
    selected[1] = {'train':take((row for row in squad('train') if cohort(row)!=0),2000),
                   'dev':take((row for row in squad('train') if cohort(row)==0),100),
                   'test':take(squad('dev'),200)}
    selected[2] = {split:[] for split in COUNTS}
    # Fixed mixture ensures the largest source does not drown out TextVQA.
    for dataset, train, dev, test in [('VQAv2',800,40,80), ('GQA',800,40,80), ('TextVQA',400,20,40)]:
        official_dev = dataset == 'GQA'
        selected[2]['train'] += take(read_rows(dataset,'train', None if official_dev else False),train)
        selected[2]['dev'] += take(read_rows(dataset,'validation' if official_dev else 'train', None if official_dev else True),dev)
        selected[2]['test'] += take(read_rows(dataset,'testdev' if official_dev else 'validation'),test)
    selected[3] = {}
    for split, count in COUNTS.items():
        def rows():
            with (ROOT/'data/MedQuAD/paper_split_seed42'/f'{split}.jsonl').open() as stream:
                for line in stream:
                    yield json.loads(line)
        selected[3][split] = take(rows(),count)
    selected[4] = {split:[] for split in COUNTS}
    for dataset, train, dev, test in [('VQA_RAD',300,0,50), ('PathVQA',1000,60,100), ('SLAKE',700,40,50)]:
        for split, count in [('train',train),('dev',dev),('test',test)]:
            if count:
                selected[4][split] += take(read_rows(dataset,'validation' if split=='dev' else split),count)
    legal = cuad_splits()
    selected[5], selected[6] = {}, {}
    for split, count in COUNTS.items():
        ids = legal['test' if split == 'test' else 'train']
        def legal_rows():
            for row in read_rows('CUAD'):
                if row['id'] in ids and (split=='test' or (cohort(row)==0)==(split=='dev')):
                    yield row
        combined = take(legal_rows(),count*2)
        selected[5][split], selected[6][split] = combined[::2], combined[1::2]
    for task, dataset in [(7,'FinQA'),(8,'TAT_DQA')]:
        selected[task] = {split:take(read_rows(dataset, split), count) for split,count in COUNTS.items()}
    for task, knowledge in [(9,True),(10,False)]:
        selected[task] = {split:take(aqua(split,knowledge),count) for split,count in COUNTS.items()}
    tasks = []
    roles = {split:set() for split in COUNTS}
    image_roles = {split:set() for split in COUNTS}
    for task in range(1,11):
        files, ids = {}, {}
        for split, count in COUNTS.items():
            rows = [canonical(row) for row in selected[task][split]]
            assert len(rows)==count
            if task == 6:
                for row in rows:
                    row['context'] = ''
                    row['metadata']['image_protocol'] = 'CUAD original PDF; question-only BM25 top-2 pages; new Qwen2-VL adaptation'
                    row['metadata']['requires_pdf_render'] = True
            for row in rows:
                roles[split].add(row['id'])
                image_roles[split].update(row['images'])
            path = OUT/f'T{task:02d}_{split}.jsonl'
            path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
            files[split] = {'path':str(path.relative_to(ROOT)), 'count':len(rows), 'sha256':sha(path)}
            ids[split] = [row['id'] for row in rows]
        tasks.append({'task':task,'task_id':f'T{task:02d}','domain':DOMAINS[task-1],
                      'modality':'text' if task%2 else 'image','instruction':INSTRUCTIONS[task-1],
                      'files':files,'sample_ids':ids,'ready_for_formal_training':False})
        print('SELECTED',task, {split:files[split]['count'] for split in COUNTS},flush=True)
    assert not roles['train'] & roles['dev'] and not roles['train'] & roles['test'] and not roles['dev'] & roles['test']
    overlaps = {f'{left}_{right}':len(image_roles[left]&image_roles[right])
                for left,right in [('train','dev'),('train','test'),('dev','test')]}
    manifest = {'protocol':'five_domain_ten_task_equal_budget_v1','sampling_seed':SEED,
                'counts_per_task':COUNTS,'tasks':tasks,'qa_role_overlap':0,
                'source_image_path_overlap_by_role':overlaps,
                'image_overlap_note':'Native medical VQA can share images across official QA splits. This is reported, not silently re-split. No QA ID crosses roles.',
                'sampling':'Deterministic SHA256 rank within source splits; grouped train holdout only where official dev absent; explicit mixture quotas.',
                'legal_pairing':'Disjoint QA IDs between T5 and T6; same official contract partitions; 10% hash-cohort of train contracts reserved for project dev.',
                'formal_ready':False,'assets_pending':['CUAD question-only selected PDF pages','AQUA fixed selected original images','SMoLoRA MiniLM instruction encoder']}
    (OUT/'selection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'data/TenTaskCL/final_task_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    prepare()
