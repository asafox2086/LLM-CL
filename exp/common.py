import hashlib
import json
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest_bytes(content):
    return hashlib.sha256(content).hexdigest()


def fingerprint(value):
    return digest_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def normalize_input(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def prepare_data(config):
    manifest = json.loads((ROOT / config['data_manifest']).read_text())
    if config['protocol'] != manifest['protocol']:
        raise ValueError('Data protocol differs from configuration')
    entries = {task['task_id']: task for task in manifest['tasks']}
    if len(config['tasks']) != 7 or set(config['tasks']) != set(entries):
        raise ValueError('This protocol requires exactly the seven locked tasks')
    bounds = {'train': (0, 1000), 'dev': (1000, 1200), 'test': (1200, 1700)}
    datasets, split_manifest = {}, {}
    all_inputs = {split: set() for split in bounds}
    for task_id in config['tasks']:
        entry = entries[task_id]
        content = (ROOT / entry['source_path']).read_bytes()
        if digest_bytes(content) != entry['source_sha256']:
            raise ValueError(f'Source checksum mismatch: {task_id}')
        source = json.loads(content)
        rows = source['Instances']
        if len(rows) != entry['total_instances']:
            raise ValueError(f'Unexpected instance count: {task_id}')
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError(f'Duplicate IDs: {task_id}')
        if len({normalize_input(row['input']) for row in rows}) != len(rows):
            raise ValueError(f'Duplicate inputs: {task_id}')
        definition = source['Definition'][0].strip()
        if not definition:
            raise ValueError(f'Empty definition: {task_id}')
        indexed, rejected = [], []
        for source_index, row in enumerate(rows):
            references = row['output'] if isinstance(row['output'], list) else [row['output']]
            if not isinstance(row['input'], str) or not row['input'].strip() or not references or any(
                    not isinstance(answer, str) or not answer.strip() for answer in references):
                rejected.append(row['id'])
            else:
                indexed.append((source_index, row))
        if rejected != entry['excluded_instance_ids'] or len(indexed) != entry['valid_instances']:
            raise ValueError(f'Invalid-instance audit differs from manifest: {task_id}')
        indexed.sort(key=lambda item: (
            digest_bytes((manifest['split_seed'] + '\n' + task_id + '\n' + item[1]['id']).encode('utf-8')),
            item[1]['id'],
        ))
        datasets[task_id], split_manifest[task_id] = {}, {}
        for split, (start, end) in bounds.items():
            selected = indexed[start:end]
            ids = '\n'.join(row['id'] for _, row in selected) + '\n'
            if len(selected) != entry['splits'][split]['count'] or digest_bytes(ids.encode('utf-8')) != entry['splits'][split]['ordered_ids_sha256']:
                raise ValueError(f'Split checksum mismatch: {task_id}/{split}')
            examples, records = [], []
            for source_index, row in selected:
                references = row['output'] if isinstance(row['output'], list) else [row['output']]
                if not references or any(not isinstance(answer, str) or not answer.strip() for answer in references):
                    raise ValueError(f'Invalid reference: {task_id}/{row["id"]}')
                examples.append({
                    'task_id': task_id, 'instance_id': row['id'],
                    'prompt': f'Instruction: {definition}\n\nInput: {row["input"].strip()}\n\nResponse:\n',
                    'references': [answer.strip() for answer in references],
                })
                records.append({'instance_id': row['id'], 'source_index': source_index, 'content_sha256': fingerprint(row)})
                all_inputs[split].add(normalize_input(row['input']))
            datasets[task_id][split] = examples
            split_manifest[task_id][split] = records
    for left, right in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        if all_inputs[left] & all_inputs[right]:
            raise ValueError(f'Input leakage between {left} and {right}')
    return datasets, split_manifest


def tokenize_examples(tokenizer, examples, config):
    tokenized, stats = [], {'count': len(examples), 'prompt_truncated': 0, 'target_truncated': 0}
    for example in examples:
        prompt = [tokenizer.bos_token_id] + tokenizer.encode(example['prompt'], add_special_tokens=False)
        target = tokenizer.encode(example['references'][0], add_special_tokens=False)
        stats['prompt_truncated'] += int(len(prompt) > config['prompt_tokens'])
        stats['target_truncated'] += int(len(target) + 1 > config['target_tokens'])
        prompt = prompt[:config['prompt_tokens']]
        target = target[:config['target_tokens'] - 1] + [tokenizer.eos_token_id]
        tokenized.append({**example, 'prompt_ids': prompt, 'input_ids': prompt + target,
                          'labels': [-100] * len(prompt) + target})
    return tokenized, stats


def normalize_answer(value):
    value = value.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', value).split())


def answer_f1(prediction, reference):
    prediction = normalize_answer(prediction).split()
    reference = normalize_answer(reference).split()
    if not prediction or not reference:
        return float(prediction == reference)
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    return 2 * overlap / (len(prediction) + len(reference))


class Scorer:
    def __init__(self):
        from rouge_score.rouge_scorer import RougeScorer
        self.rouge = RougeScorer(['rougeL'], use_stemmer=True)

    def score(self, prediction, references):
        return {
            'rougeL': 100 * max(self.rouge.score(reference, prediction)['rougeL'].fmeasure for reference in references),
            'exact_match': 100 * max(float(normalize_answer(prediction) == normalize_answer(reference)) for reference in references),
            'token_f1': 100 * max(answer_f1(prediction, reference) for reference in references),
        }


def continual_metrics(matrix, singles=None):
    task_count = len(matrix[0])
    if task_count < 2 or len(matrix) != task_count + 1 or any(len(row) != task_count for row in matrix):
        raise ValueError('Expected a complete (T+1) by T matrix with T >= 2')
    if any(value is None for row in matrix for value in row):
        raise ValueError('Cannot compute complete metrics with missing scores')
    diagonal = [matrix[index + 1][index] for index in range(task_count)]
    scores = {
        'AP': sum(matrix[-1]) / task_count,
        'F.Rate': sum(max(matrix[stage][task] for stage in range(task + 1, task_count)) - matrix[-1][task]
                      for task in range(task_count - 1)) / (task_count - 1),
        'BWT': sum(matrix[-1][task] - diagonal[task] for task in range(task_count - 1)) / (task_count - 1),
        'FWT': None,
        'FWT_zero_shot': sum(matrix[task][task] - matrix[0][task] for task in range(1, task_count)) / (task_count - 1),
    }
    if singles is not None:
        if len(singles) != task_count or any(score is None for score in singles):
            raise ValueError('Incomplete single-task controls')
        scores['FWT'] = sum(diagonal[task] - singles[task] for task in range(task_count)) / task_count
    return scores
