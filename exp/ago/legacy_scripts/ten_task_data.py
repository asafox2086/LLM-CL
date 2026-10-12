"""Input-only helpers for the locked five-domain, ten-task QA preparation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

def iter_samples(task_manifest,task_id,split='train'):
    if split not in {'train','validation','test'}:
        raise ValueError(split)
    manifest=json.loads(Path(task_manifest).read_text(encoding='utf-8'))
    task=next(t for t in manifest['tasks'] if t['task']==task_id)
    with (ROOT/task['files'][split]['path']).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)

def build_messages(sample):
    content=[]
    for name in sample.get('images',[]):
        path=(ROOT/name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        content.append({'type':'image','image':str(path)})
    context=sample.get('model_context',sample.get('context','')).strip()
    text='Answer the question'+(' using the supplied context' if context else ' using the image' if content else '')+'. Give the answer directly.'
    if context:text+='\nContext:\n'+context
    text+='\nQuestion: '+sample['question']
    content.append({'type':'text','text':text})
    return [{'role':'user','content':content}]

def visual_budget(sample):
    count=len(sample.get('images',[]))
    total=0 if count==0 else 512 if count==1 else 2048
    return {'image_count':count,'total_merged_tokens':total,'max_pixels_per_image':(total//count)*28*28 if count else None,'min_pixels':4*28*28 if count else None}
