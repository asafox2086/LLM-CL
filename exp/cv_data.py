"""Locked nine-task multimodal pilot and frozen Qwen visual-feature cache."""
import argparse
import json
import time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from common import ROOT, digest_bytes, fingerprint, prepare_data, write_json
from recovery import atomic_save

MODEL = ROOT / 'model/Qwen2-VL-2B-Instruct'
CACHE = ROOT / 'exp/CV_result/shared'
COUNTS = {'train': 1000, 'dev': 100, 'test': 200}

def datasets():
    old = json.loads((ROOT / 'exp/configs/seq_lora_t5large.json').read_text())
    original, _ = prepare_data(old)
    data = {task: {s: rows[:COUNTS[s]] for s, rows in splits.items()}
            for task, splits in original.items() if task in old['tasks'][:5]}
    medical = json.loads((ROOT / 'rules/003_medical_manifest.json').read_text())
    for entry in medical['tasks']:
        data[entry['task_id']] = {}
        for split, meta in entry['splits'].items():
            raw = (ROOT / meta['path']).read_bytes()
            assert digest_bytes(raw) == meta['sha256']
            data[entry['task_id']][split] = [json.loads(line) for line in raw.splitlines()][:COUNTS[split]]
    for folder, task in [('natural/vqav2', 'natural_vqav2'), ('medical/vqa_rad', 'medical_vqa_rad')]:
        base = ROOT / 'CV_data' / folder
        manifest = json.loads((base / 'manifest.json').read_text())
        images = json.loads((base / 'image_manifest.json').read_text())
        data[task] = {}
        for split, count in COUNTS.items():
            raw = (base / f'{split}.jsonl').read_bytes()
            assert digest_bytes(raw) == manifest['splits'][split]['sha256']
            selected = [json.loads(line) for line in raw.splitlines()][:count]
            data[task][split] = []
            for row in selected:
                meta = images[row['image']]
                assert digest_bytes((ROOT / 'CV_data' / row['image']).read_bytes()) == meta['sha256']
                data[task][split].append({**row, 'task_id': task, 'instance_id': str(row['id']),
                    'prompt': 'Answer the question briefly using the image.\n' + row['question'],
                    'references': row['references'], 'training_answer': row['answer'], 'image_metadata': meta})
    return data

def processor():
    return AutoProcessor.from_pretrained(MODEL, local_files_only=True,
        min_pixels=4*28*28, max_pixels=128*28*28)

def tokenize(row, proc, grid=None):
    # Truncate user text before applying template; preserve image markers and assistant header.
    tok = proc.tokenizer
    content = ([{'type': 'image'}] if row.get('image') else []) + [{'type': 'text', 'text': row['prompt']}]
    text = proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    image_count = int(torch.tensor(grid).prod().item()) // 4 if grid else 0
    text_ids = tok.encode(row['prompt'], add_special_tokens=False)
    overhead = len(tok.encode(text, add_special_tokens=False)) - len(text_ids) + max(0, image_count - 1)
    budget = 768 - overhead
    assert budget > 0
    if len(text_ids) > budget:
        content[-1]['text'] = tok.decode(text_ids[:budget], skip_special_tokens=False)
        text = proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    if grid:
        text = text.replace('<|image_pad|>', '<|image_pad|>' * image_count)
    prompt_ids = tok.encode(text, add_special_tokens=False)
    assert len(prompt_ids) <= 768, len(prompt_ids)
    target = tok.encode(row.get('training_answer', row['references'][0]), add_special_tokens=False)
    target_ids = target[:255] + [tok.eos_token_id]
    return {**row, 'rendered_prompt': text, 'prompt_ids': prompt_ids,
            'input_ids': prompt_ids + target_ids, 'labels': [-100]*len(prompt_ids) + target_ids,
            'image_grid_thw': grid, 'image_tokens': image_count,
            'prompt_truncated': len(text_ids) > budget, 'target_truncated': len(target) > 255}

def feature_path(row):
    return CACHE / 'features' / (row['image_metadata']['sha256'] + '.pt')

def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    data = datasets()
    proc = processor()
    images = {row['image']: row for splits in data.values() for rows in splits.values() for row in rows if row.get('image')}
    model = None
    for index, row in enumerate(images.values()):
        path = feature_path(row)
        if not path.exists():
            if model is None:
                model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
                    torch_dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa').eval()
            started = time.monotonic()
            with Image.open(ROOT / 'CV_data' / row['image']) as im:
                packed = proc.image_processor(images=[im.convert('RGB')], return_tensors='pt')
            with torch.inference_mode():
                features = model.visual(packed['pixel_values'].cuda().half(), grid_thw=packed['image_grid_thw'].cuda())
            atomic_save({'features': features.cpu(), 'grid': packed['image_grid_thw'][0].tolist(),
                'image_sha256': row['image_metadata']['sha256'], 'seconds': time.monotonic()-started}, path)
        if index % 100 == 0:
            print(json.dumps({'event': 'visual_cache', 'done': index+1, 'total': len(images)}), flush=True)
    for task, splits in data.items():
        for split, rows in splits.items():
            encoded = []
            for row in rows:
                grid = torch.load(feature_path(row), weights_only=True)['grid'] if row.get('image') else None
                encoded.append(tokenize(row, proc, grid))
            data[task][split] = encoded
    atomic_save(data, CACHE / 'data.pt')
    manifest = {'tasks': list(data), 'counts': COUNTS, 'data_sha256': digest_bytes((CACHE/'data.pt').read_bytes()),
        'visual_max_tokens': 128, 'visual_encoder': 'frozen FP16; actual image features, reused across methods',
        'splits': {task: {split: {'ids': [r['instance_id'] for r in rows], 'content_sha256': fingerprint(rows),
                   'prompt_truncated': sum(r['prompt_truncated'] for r in rows),
                   'target_truncated': sum(r['target_truncated'] for r in rows)} for split, rows in splits.items()}
                   for task, splits in data.items()},
        'feature_sha256': {p.name: digest_bytes(p.read_bytes()) for p in sorted((CACHE/'features').glob('*.pt'))},
        'model_manifest': json.loads((MODEL/'download_manifest.json').read_text())}
    write_json(CACHE/'manifest.json', manifest)
    print(json.dumps({'event': 'prepared', 'tasks': list(data)}), flush=True)

if __name__ == '__main__':
    prepare()
