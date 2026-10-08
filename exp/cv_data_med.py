"""Isolated medical-only cache, sharing the original tokenizer and visual protocol."""
import json
import shutil
from collections import Counter
import torch
import cv_data
from common import ROOT, digest_bytes, write_json

CACHE = ROOT/'exp/CV_result_med/shared'
TASKS = ['medical_medmcqa', 'medical_medqa', 'medical_vqa_rad_chest',
         'medical_vqa_rad_head', 'medical_vqa_rad_abd']
# These imported functions resolve CACHE in cv_data's module globals. Setting it affects
# only this process, including train/runtime/reflection; original files/runs stay immutable.
cv_data.CACHE = CACHE

def datasets():
    source = json.loads((ROOT/'data/medical_qa_med/manifest.json').read_text())
    data = {}
    for task in TASKS[:2]:
        data[task] = {}
        for split in ['train','dev','test']:
            entry = source['tasks'][task][split]; raw = (ROOT/entry['path']).read_bytes()
            assert digest_bytes(raw) == entry['sha256']
            rows = [json.loads(line) for line in raw.splitlines()]
            data[task][split] = [{**r, 'task_id': task,
                'prompt': 'Answer this medical multiple-choice question. Respond only with the letter of the correct answer.\n\n'
                          +r['question']+'\n'+'\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(r['choices'])),
                'references': ['ABCD'[r['answer']]], 'training_answer': 'ABCD'[r['answer']]} for r in rows]
    base = ROOT/'CV_data/medical/vqa_rad'
    manifest = json.loads((base/'manifest.json').read_text())
    images = json.loads((base/'image_manifest.json').read_text())
    all_rows = {}
    for split in ['train','dev','test']:
        raw = (base/f'{split}.jsonl').read_bytes()
        assert digest_bytes(raw) == manifest['splits'][split]['sha256']
        # Preserve the old experiment's selected QA/image pool; divide it into organs.
        all_rows[split] = [json.loads(x) for x in raw.splitlines()][:cv_data.COUNTS[split]]
    organs = {}
    for rows in all_rows.values():
        for r in rows:
            organs.setdefault(r['image'], set()).add(r['organ'])
    for organ in ['CHEST','HEAD','ABD']:
        task = 'medical_vqa_rad_'+organ.lower(); data[task] = {}
        for split, rows in all_rows.items():
            chosen = [r for r in rows if r['organ'] == organ and len(organs[r['image']]) == 1][:cv_data.COUNTS[split]]
            data[task][split] = [{**r, 'task_id': task, 'instance_id': str(r['id']),
                'prompt': 'Answer the medical question briefly using the image.\n'+r['question'],
                'references': r['references'], 'training_answer': r['answer'], 'image_metadata': images[r['image']]} for r in chosen]
            assert chosen, (task,split)
    # Source image/case-group split is retained; neither images nor questions repeat across tasks/splits.
    sets = {s:{r['image'] for t in TASKS[2:] for r in data[t][s]} for s in ['train','dev','test']}
    assert not (sets['train']&sets['dev'] or sets['train']&sets['test'] or sets['dev']&sets['test'])
    assert list(data) == TASKS
    return data

def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    data = datasets()
    # Reuse byte-identical, frozen 128-token visual features. Validate original cache and
    # originals first; copy into isolated _med cache. No old manifest/data is rewritten.
    old = ROOT/'exp/CV_result/shared'
    old_manifest = json.loads((old/'manifest.json').read_text())
    for splits in data.values():
        for rows in splits.values():
            for row in rows:
                if not row.get('image'):
                    continue
                assert digest_bytes((ROOT/'CV_data'/row['image']).read_bytes()) == row['image_metadata']['sha256']
                src = old/'features'/(row['image_metadata']['sha256']+'.pt')
                if src.exists():
                    assert digest_bytes(src.read_bytes()) == old_manifest['feature_sha256'][src.name]
                    dst = cv_data.feature_path(row); dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists():
                        shutil.copy2(src,dst)
    # Preserve complete question/answer options. The shared tokenizer normally clips
    # the user prompt from the right; long MCQs instead shorten only the question.
    tokenize_original = cv_data.tokenize
    def tokenize_med(row, proc, grid=None):
        encoded = tokenize_original(row,proc,grid)
        if row.get('choices') and encoded['prompt_truncated']:
            question_ids = proc.tokenizer.encode(row['question'],add_special_tokens=False)
            instruction = row['prompt'].split('\n\n',1)[0]
            options = '\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(row['choices']))
            shortened = dict(row)
            while encoded['prompt_truncated']:
                question_ids = question_ids[:-16]
                assert len(question_ids)>32, 'Options alone exceed prompt budget'
                question = proc.tokenizer.decode(question_ids,skip_special_tokens=False)
                shortened['prompt'] = instruction+'\n\n'+question+'\n'+options
                encoded = tokenize_original(shortened,proc,grid)
            encoded.update(prompt_truncated=True,question_shortened=True,original_prompt=row['prompt'],
                           truncation_policy='shorten question tail only; preserve all four complete options')
        return encoded
    cv_data.tokenize = tokenize_med
    cv_data.datasets = lambda: data
    cv_data.prepare()
    manifest = json.loads((CACHE/'manifest.json').read_text())
    manifest['counts'] = {t:{s:len(rows) for s,rows in splits.items()} for t,splits in data.items()}
    manifest['answer_types'] = {t:{s:dict(Counter(r.get('answer_type','MCQ') for r in rows)) for s,rows in splits.items()} for t,splits in data.items()}
    manifest['medical_qa_source'] = json.loads((ROOT/'data/medical_qa_med/manifest.json').read_text())
    manifest['vqa_split_policy'] = 'existing custom seed42 connected image/case-group split; NOT published question-level split; ambiguous multi-organ images excluded'
    manifest['initialization'] = 'fresh official Qwen2-VL-2B-Instruct; no prior experimental LoRA weights'
    write_json(CACHE/'manifest.json',manifest)

if __name__ == '__main__':
    prepare()
