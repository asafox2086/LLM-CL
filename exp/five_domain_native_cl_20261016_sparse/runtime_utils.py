"""Frozen-split sequential open QA: real images, complete results, no checkpoints."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import random
import re
import time
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration,get_cosine_schedule_with_warmup
from methods import ROOT,make_method as Method
from retrieval import retrieve



def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def tokens(text):
    text=text.lower()
    return re.findall(r'[a-z0-9]+|[\u3400-\u9fff]',text)


def score(prediction,references):
    a=tokens(prediction)
    values=[]
    for reference in references:
        b=tokens(reference)
        exact=float(a==b)
        if not a or not b:
            values.append({'exact_match':100*exact,'token_f1':100*exact,'rougeL':100*exact})
            continue
        common=sum((Counter(a)&Counter(b)).values())
        f1=2*common/(len(a)+len(b))
        masks={}
        for index,word in enumerate(b):
            masks[word]=masks.get(word,0)|(1<<index)
        state=0
        for word in a:
            union=masks.get(word,0)|state
            state=union&~(union-((state<<1)|1))
        values.append({'exact_match':100*exact,'token_f1':100*f1,'rougeL':200*state.bit_count()/(len(a)+len(b))})
    return {name:max(value[name] for value in values) for name in values[0]}


def input_batch(row,task,processor,device):
    context=row.get('context','')
    retrieval_indexes=[]
    if len(context)>6000:
        chunks=[context[start:start+1000] for start in range(0,len(context),800)]
        retrieval_indexes=sorted(retrieve(row['question'],chunks,5))
        context='\n'.join(chunks[index] for index in retrieval_indexes)
    content=[]
    images=[]
    for name in row.get('images',[]):
        with Image.open(ROOT/name) as image:
            images.append(image.convert('RGB'))
        content.append({'type':'image'})
    text=task['instruction']
    if context:
        text+='\nContext:\n'+context
    text+='\nQuestion: '+row['question']
    content.append({'type':'text','text':text})
    prompt=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    # Both methods preserve the author's effective frozen-visual setting.
    total_visual=512 if len(images)==1 else 1024
    pixel_budget=max(16,total_visual//max(1,len(images)))*28*28
    # Transformers 4.46 reads these limits from the image processor;
    # passing them to the combined processor forwards them to the tokenizer.
    processor.image_processor.min_pixels=4*28*28
    processor.image_processor.max_pixels=pixel_budget
    processor.image_processor.size={'min_pixels':4*28*28,'max_pixels':pixel_budget}
    batch=processor(text=[prompt],images=images or None,return_tensors='pt')
    if batch['input_ids'].shape[1]>2304:
        raise ValueError('prompt over budget; do not truncate image placeholder sequences silently')
    batch={key:value.to(device) for key,value in batch.items()}
    return batch,{'prompt_tokens':int(batch['input_ids'].shape[1]),'retrieved_chunk_indexes':retrieval_indexes}


def answer_loss(model,batch,answer,processor):
    target=processor.tokenizer(answer,add_special_tokens=False,truncation=True,max_length=512)['input_ids']
    if not target:
        raise ValueError('empty answer tokens')
    target.append(processor.tokenizer.eos_token_id)
    prompt_length=batch['input_ids'].shape[1]
    target_tensor=torch.tensor([target],device=batch['input_ids'].device)
    ids=torch.cat([batch['input_ids'],target_tensor],dim=1)
    mask=torch.ones_like(ids)
    embeddings=model.get_input_embeddings()(ids)
    grid=batch.get('image_grid_thw')
    if 'pixel_values' in batch:
        visual=model.visual(batch['pixel_values'].to(model.visual.get_dtype()),grid_thw=grid)
        image_mask=(ids==model.config.image_token_id).unsqueeze(-1).expand_as(embeddings)
        if image_mask.sum().item()!=visual.numel():
            raise ValueError('image feature/token count mismatch')
        embeddings=embeddings.masked_scatter(image_mask,visual.to(embeddings.dtype))
    positions,_=model.get_rope_index(ids,grid,attention_mask=mask)
    output=model.model(inputs_embeds=embeddings,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True)
    hidden=output.last_hidden_state[:,prompt_length-1:-1,:]
    logits=model.lm_head(hidden).float()
    return torch.nn.functional.cross_entropy(logits.flatten(0,1),target_tensor.flatten()),len(target)


def evaluate(method,processor,tasks,data,run,stage,split='test',only_task=None):
    method.model.eval()
    scores=[]
    limit={1:128,2:128,3:512,4:512,5:1024,6:512,7:128,8:512,9:512,10:256}
    for task in tasks:
        task_id=task['task']
        if only_task is not None and task_id!=only_task:
            continue
        method.set_instruction(task_id)
        rows=data[(task_id,split)]
        output=run/'evaluations'/f'stage_{stage:02d}'/f'T{task_id:02d}_{split}_predictions.jsonl'
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix('.jsonl.partial')
        existing=[]
        prior=output if output.exists() else temporary
        if prior.exists():
            for line in prior.read_text(encoding='utf-8').splitlines():
                try:
                    existing.append(json.loads(line))
                except json.JSONDecodeError:
                    break
            if [row['id'] for row in existing]!=[row['id'] for row in rows[:len(existing)]]:
                raise ValueError('prediction resume manifest mismatch')
        if not output.exists():
            temporary.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in existing),encoding='utf-8')
            with temporary.open('a',encoding='utf-8') as stream:
                for row in rows[len(existing):]:
                    batch,details=input_batch(row,task,processor,method.model.device)
                    if hasattr(method,'vision_original'):
                        from frozen_vision import prepare
                        details['visual_cache_hit']=prepare(batch,row,method.model,method.vision_original)
                        if hasattr(method,'prepare_row'):method.prepare_row(row,batch)
                    started=time.monotonic()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=method.model.dtype):
                        generated=method.model.generate(**batch,do_sample=False,num_beams=1,
                            max_new_tokens=task['max_new_tokens'],use_cache=True,
                            pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
                    response=generated[0,batch['input_ids'].shape[1]:]
                    prediction=processor.tokenizer.decode(response,skip_special_tokens=True)
                    record={'id':row['id'],'dataset':row.get('dataset',row['id'].split(':',1)[0]),
                            'prediction':prediction,'references':row['answers'],
                            'scores':score(prediction,row['answers']),'generation_seconds':time.monotonic()-started,
                            'output_tokens':len(response),'generation_cap':task['max_new_tokens'],
                            'native_label':task.get('native_label'),'metadata':row.get('metadata',{}),**details}
                    existing.append(record)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                    stream.flush()
                    if len(existing)%20==0:
                        print(json.dumps({'event':'evaluation','stage':stage,'task':task_id,'split':split,'done':len(existing),'total':len(rows)}),flush=True)
                os.fsync(stream.fileno())
            temporary.replace(output)
        if len(existing)!=len(rows):
            raise ValueError('incomplete predictions')
        result={name:sum(row['scores'][name] for row in existing)/len(existing)
                for name in ['rougeL','token_f1','exact_match']}
        result['count']=len(existing)
        result['exact_match_correct']=sum(x['scores']['exact_match']==100 for x in existing)
        datasets={}
        for record in existing:
            name=record['dataset'];item=datasets.setdefault(name,{'count':0,'exact_match_correct':0,'exact_match':0.,'token_f1':0.,'rougeL':0.})
            item['count']+=1;item['exact_match_correct']+=record['scores']['exact_match']==100
            for metric in ['exact_match','token_f1','rougeL']:item[metric]+=record['scores'][metric]
        for item in datasets.values():
            for metric in ['exact_match','token_f1','rougeL']:item[metric]/=item['count']
        write(output.with_suffix('.datasets.json'),datasets)
        write(output.with_suffix('.scores.json'),result)
        scores.append(result['rougeL'])
    return scores


def rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


def make_optimizer(method):
    groups={}
    for name,parameter in method.model.named_parameters():
        if not parameter.requires_grad:
            continue
        decay=method.name=='SMoLoRA' and parameter.ndim>=2 and not name.endswith('.bias')
        key=(getattr(method,'learning_rate',1e-4),.01 if decay else 0.)
        groups.setdefault(key,[]).append(parameter)
    from host_adam import HostAdamW
    optimizer=HostAdamW if (method.name=='PASs_MoE' or getattr(method,'host_optimizer',False)) else torch.optim.AdamW
    return optimizer([{'params':parameters,'lr':lr,'weight_decay':decay}
                              for (lr,decay),parameters in groups.items()],eps=1e-8,betas=(.9,.999))


def activation_storage(method):
    # Keep the original expert computation and parameter precision. Transfer saved
    # autograd tensors to host RAM until backward instead of retaining them on GPU.
    return torch.autograd.graph.save_on_cpu(pin_memory=True) if (method.name=='SMoLoRA' or getattr(method,'offload_activations',False)) else nullcontext()


def save_last(path,method,optimizer,scheduler,scaler,stage,offset,step,manifest_sha):
    temporary=path.with_suffix('.pt.tmp')
    torch.save({'weights':method.changed_state(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'scaler':scaler.state_dict(),'method_state':method.method_state(stage),'rng':rng(),
                'stage':stage,'next_offset':offset,'global_step':step,'manifest_sha':manifest_sha},temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
